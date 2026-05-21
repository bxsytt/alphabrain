"""
Fast ActionToken rollout — step-lock architecture.

Key design: all envs move in lockstep.
  1. Batch VLA forward for ALL active envs (one GPU call)
  2. Batch encoder + actor
  3. ALL envs execute chunk in parallel threads
  4. Collect results, repeat

No BatchInferenceServer needed. No async queuing. No batch fragmentation.

Speedup: ~50x vs original (env creation + batch fragmentation eliminated).
"""

"""
训练逻辑 — 包含 action_token_ppo_loss（PPO loss计算）、action_token_collect_group（episode收集）、push_episodes_to_buffer（回放缓冲区写入）、
action_token_td_critic_update 和 action_token_td_actor_update（TD3更新）
"""

import gc
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

import numpy as np
import torch

from AlphaBrain.training.reinforcement_learning.envs.persistent_env_pool import PersistentEnvPool
from AlphaBrain.training.reinforcement_learning.algos.RLActionToken.action_token_encoder_decoder import ActionTokenEncoderDecoder
from AlphaBrain.training.reinforcement_learning.algos.RLActionToken.action_token_actor_critic import ActionTokenActor, ActionTokenCritic
from AlphaBrain.training.reinforcement_learning.algos.RLActionToken.action_token_trainer import ActionTokenEpisode, ActionTokenStepRecord
from AlphaBrain.training.reinforcement_learning.common.rollout import _unnormalize, _postprocess_action, DUMMY_ACTION

logger = logging.getLogger(__name__)


def _env_step_chunk(env_pool, env_idx, action_chunk_unnorm, chunk_len, record_frames=False):
    """Execute chunk_len env steps in ONE pipe round-trip (8x fewer I/Os)."""
    actions = [_postprocess_action(action_chunk_unnorm[step]) for step in range(chunk_len)]
    try:
        obs, reward, done, steps_taken = env_pool.envs[env_idx].step_chunk(actions)
    except RuntimeError as e:
        print(f"  [WARNING] env {env_idx} step_chunk failed: {e}, marking as done", flush=True)
        # Return a fake "failed" result — episode will be marked as failure
        obs = {"primary_image": np.zeros((256,256,3), dtype=np.uint8),
               "wrist_image": np.zeros((256,256,3), dtype=np.uint8),
               "state": np.zeros(8, dtype=np.float32)}
        return obs, 0.0, True, 0, []
    return obs, reward, done, steps_taken, []


def _env_dummy_steps(env_pool, env_idx, n_steps):
    """Execute dummy actions (warmup). Returns final obs."""
    obs = None
    for _ in range(n_steps):
        obs, _, _ = env_pool.step_env(env_idx, DUMMY_ACTION)
    return obs

"""
 这是一个 step-lock（步锁）架构的并行 rollout 采集函数，是 RLActionToken 强化学习系统中核心的性能瓶颈突破点。
 它在同一 GPU 上并行运行 G 个环境实例，以 lockstep（齐步走） 的方式用 单次 GPU 批处理调用 同时为所有活跃环境推理，
 大幅提升 GPU 利用率。
 """
@torch.no_grad()
def action_token_collect_group_steplock(
    env_pool: PersistentEnvPool,
    frozen_vla,
    encoder: ActionTokenEncoderDecoder,
    actor: ActionTokenActor,
    critic: ActionTokenCritic,
    suite_name: str,
    task_id: int,
    n_initial_states: int,
    action_norm_stats: dict,
    max_steps: int,
    chunk_len: int,
    G: int = 64,
    seed: int = 42,
    num_steps_wait: int = 10,
    device: str = "cuda",
    video_dir: Optional[str] = None,
    group_idx: int = 0,
    store_images: bool = False,
    group_size: int = 1,
    reward_coef: float = 1.0,
    actor_chunk_len: int = None,
    env_offset: int = 0,
    warmup_mode: bool = False,
) -> List[ActionTokenEpisode]:
    """
    Collect G episodes using step-lock architecture.

    All envs move in lockstep:
      1. Batch VLA forward (one GPU call for ALL active envs)
      2. Batch encoder + actor (or skip actor if warmup_mode)
      3. All envs execute chunk in parallel
      4. Repeat

    Args:
        env_offset: starting env index in the pool
        actor_chunk_len: if set, actor outputs shorter chunk than VLA
        warmup_mode: if True, use VLA actions directly (skip actor). For buffer pre-fill.
    """
    if actor_chunk_len is None:
        actor_chunk_len = chunk_len
    exec_chunk_len = actor_chunk_len  # how many steps to execute per chunk

    frozen_vla.eval()
    encoder.eval()
    actor.eval()

    # Assign initial states (same-state grouping)
    num_unique = max(1, G // group_size)
    _rng = np.random.RandomState(seed + group_idx)
    unique_states = _rng.randint(0, n_initial_states, size=num_unique)
    state_ids = np.repeat(unique_states, group_size)[:G]

    n_workers = min(G, len(env_pool))

    # ── Phase 1: Reset all envs in parallel   使用 G 个线程并行重置所有环境 ──
    # 多线程并行重置（Parallel Reset）：利用 ThreadPoolExecutor 并行调用env_pool.reset_env，让G个环境同时初始化
    from concurrent.futures import as_completed as _as_completed
    obs_list = [None] * G
    with ThreadPoolExecutor(max_workers=G) as _pool:
        _futs = {_pool.submit(env_pool.reset_env, env_offset + g, suite_name, task_id, int(state_ids[g]), seed + g): g for g in range(G)}
        for _f in _as_completed(_futs):
            obs_list[_futs[_f]] = _f.result()
    print(f"  reset done: {G} envs (parallel)", flush=True)

    task_descriptions = [env_pool.envs[env_offset + g].task_description for g in range(G)]

    # ── Phase 2: Warmup dummy steps (parallel) ──
    # 所有环境并行执行 num_steps_wait 步零动作，让仿真的物理世界（如 MuJoCo 或 Isaac Gym）的刚体碰撞、重力过渡稳定下来
    if num_steps_wait > 0:
        with ThreadPoolExecutor(max_workers=G) as _pool:
            _futs = {_pool.submit(_env_dummy_steps, env_pool, env_offset + g, num_steps_wait): g for g in range(G)}
            for _f in _as_completed(_futs):
                obs_list[_futs[_f]] = _f.result()

    # ── Phase 3: Step-lock main loop ──
    episodes = [ActionTokenEpisode(task_id=task_id, state_idx=int(state_ids[g])) for g in range(G)]
    active = [True] * G  # which envs are still running
    env_steps = [0] * G
    all_frames = [[] for _ in range(G)]  # video frames

    max_chunks = max_steps // exec_chunk_len + 1

    # Timing accumulators
    _t_vla_forward = 0.0
    _t_encoder_actor = 0.0
    _t_store_records = 0.0
    _t_unnormalize = 0.0
    _t_env_step = 0.0
    _t_total_chunks = 0
    _t_rollout_start = time.time()

    # ── 显存监控：每次collect开始时打印显存状态 ──
    _mem_alloc_start = torch.cuda.memory_allocated(device) / 1024**3
    _mem_reserved_start = torch.cuda.memory_reserved(device) / 1024**3
    print(f"  [GPU MEM @ start] allocated={_mem_alloc_start:.2f}GB  reserved={_mem_reserved_start:.2f}GB", flush=True)

    # [环境状态/图像] ──> 1. VLA 模型推理 ──> 2. RL 模型微调/动作采样 ──> 3. 动作反归一化 ──> 4. 多环境并行执行并记录数据
    for chunk_idx in range(max_chunks):
        active_ids = [g for g in range(G) if active[g]]
        if not active_ids:
            break
        _t_chunk_start = time.time()

        # ── Step 1: Batch VLA forward for all active envs ──
        _t0 = time.time()
        batch_images = [[obs_list[g]["primary_image"], obs_list[g]["wrist_image"]] for g in active_ids]
        batch_instrs = [task_descriptions[g] for g in active_ids]
        batch_props = [np.array(obs_list[g]["state"], dtype=np.float32) for g in active_ids]

        print(f"  [VLA forward] batch={len(batch_images)}, active_envs={len(active_ids)}", flush=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            action_queries, vla_actions = frozen_vla.get_vla_action(    # [2,8,2048] [2,8,7]
                batch_images=batch_images, instructions=batch_instrs)
        torch.cuda.synchronize()
        _t1 = time.time()
        _t_vla_forward += _t1 - _t0
        print(f"  [VLA done] aq={action_queries.shape} va={vla_actions.shape} time={_t1-_t0:.3f}s", flush=True)

        # ── Step 2: Batch encoder + actor ──
        _t0 = time.time()
        rl_tokens = encoder.encode(action_queries)  # (N_active, 1, D)   (2,1,256)
        # 显存优化：action_queries 使用完毕后立即删除（释放QwenVL中间隐状态的最后一个引用）
        del action_queries

        props_t = torch.tensor(np.array(batch_props), dtype=torch.float32).to(device)

        # Slice VLA actions for actor if actor uses shorter chunk
        # 如果 actor_chunk_len < vla_actions.size(1)，先裁剪 VLA 动作到 actor 需要的长度
        if actor_chunk_len < vla_actions.size(1):
            vla_actions_for_actor = vla_actions[:, :actor_chunk_len, :]
        else:
            vla_actions_for_actor = vla_actions

        """
        warmup_mode=True：直接使用 VLA 原始动作，跳过 actor（回放缓冲区预填充）
        warmup_mode=False（正常 RL 训练）：
            Actor 以 RL token + VLA 输出的参考动作 + 本体感受为输入，输出编辑后的动作和对应的对数概率
            Critic 估计每个状态的 状态价值 V(s)
            deterministic=False 意味着动作采样的随机策略，用于探索
        """
        if warmup_mode:
            # Warmup: use VLA actions directly, skip actor (like BatchInferenceServer.warmup_mode)
            actions_t = vla_actions_for_actor
            log_probs = torch.zeros(len(active_ids), device=device)
            values = torch.zeros(len(active_ids), device=device)
        else:
            actions_t, log_probs = actor(rl_tokens, vla_actions_for_actor, props_t, deterministic=False)
            values = critic(rl_tokens)  # (N_active,)
        torch.cuda.synchronize()
        _t1 = time.time()
        _t_encoder_actor += _t1 - _t0

        # Convert to numpy
        actions_np = actions_t.cpu().numpy()  # (N_active, exec_chunk_len, action_dim)
        vla_actions_cpu = vla_actions_for_actor.cpu()

        # ── Store step records ──
        # 把刚刚在 GPU 里面由大模型（VLA）和强化学习网络（Actor-Critic）实时计算出来的各种高维张量（Tensor）
        # “打包”并“搬运”到 CPU 内存中，像写日记一样，记录下当前时间步机器人的所有“所思所想”和“所作所为”
        _t0 = time.time()
        for i, g in enumerate(active_ids):
            sr = ActionTokenStepRecord(
                rl_token=rl_tokens[i:i+1].cpu().squeeze(0),
                vla_action=vla_actions_cpu[i],
                action_taken=actions_t[i].detach().cpu(),
                old_log_prob=log_probs[i].item() if log_probs is not None else 0.0,
                value=values[i].item(),
                prop_state=torch.tensor(batch_props[i]),
                images=[obs_list[g]["primary_image"].copy(), obs_list[g]["wrist_image"].copy()] if store_images else None,
                instruction=task_descriptions[g] if store_images else None,
            )
            episodes[g].step_records.append(sr) # 打包好这个时间步的记录 sr 后，代码顺着环境索引 g，把它塞进了属于该环境的专属故事线 episodes[g] 的列表末尾
        _t1 = time.time()
        _t_store_records += _t1 - _t0

        # 显存优化：GPU张量已拷贝到CPU，删除GPU版本释放显存
        del vla_actions_for_actor
        del vla_actions  # 显存优化：显式删除原始vla_actions引用
        del rl_tokens
        del actions_t
        if not warmup_mode:
            del log_probs
            del values
        
        # ── Step 3: Unnormalize actions ──
        _t0 = time.time()
        action_chunks_unnorm = []
        for i in range(len(active_ids)):
            action_chunks_unnorm.append(_unnormalize(actions_np[i], action_norm_stats))
        _t1 = time.time()
        _t_unnormalize += _t1 - _t0

        # ── Step 4: All envs execute chunk in parallel ──
        _t0 = time.time()
        record_video = video_dir is not None
        # 拿到了真正的物理动作后，再次启动 ThreadPoolExecutor 线程池
        with ThreadPoolExecutor(max_workers=len(active_ids)) as _pool:
            _futs = {}
            for i, g in enumerate(active_ids):
                _futs[_pool.submit(
                    _env_step_chunk, env_pool, env_offset + g, action_chunks_unnorm[i],
                    exec_chunk_len, record_video
                )] = (i, g)
            for _f in _as_completed(_futs):
                i, g = _futs[_f]
                obs, reward, done, steps_taken, frames = _f.result()
                obs_list[g] = obs
                env_steps[g] += steps_taken
                if record_video:
                    all_frames[g].extend(frames)
                if done or env_steps[g] >= max_steps:
                    active[g] = False
                    ep = episodes[g]
                    ep.success = bool(done and reward > 0.5)
                    ep.reward = reward_coef if ep.success else 0.0
                    ep.done_cache_idx = steps_taken
                    ep.finish_step = len(ep.step_records)
                    ep.env_steps = env_steps[g]
        _t1 = time.time()
        _t_env_step += _t1 - _t0
        _t_total_chunks += 1

        print(
            f"[TIMING] chunk {chunk_idx}: active={len(active_ids)} | "
            f"vla={_t_vla_forward/_t_total_chunks:.3f}s  enc+act={_t_encoder_actor/_t_total_chunks:.3f}s  "
            f"store={_t_store_records/_t_total_chunks:.3f}s  unnorm={_t_unnormalize/_t_total_chunks:.3f}s  "
            f"env_step={_t_env_step/_t_total_chunks:.3f}s  "
            f"chunk_total={time.time()-_t_chunk_start:.3f}s"
        )

        # ── 显存优化：每5个chunk清理一次PyTorch缓存分配器 ──
        # QwenVL forward (output_hidden_states=True) 每次产生 ~290MB 中间隐状态。
        # PyTorch的缓存分配器不会自动归还显存，定期清理可避免碎片化累积导致OOM。
        if chunk_idx > 0 and chunk_idx % 5 == 0:
            _mem_before = torch.cuda.memory_allocated(device) / 1024**3
            gc.collect()
            torch.cuda.empty_cache()
            _mem_after = torch.cuda.memory_allocated(device) / 1024**3
            _freed = _mem_before - _mem_after
            if _freed > 0.1:  # 只打印释放超过100MB的情况
                print(f"  [GPU MEM] chunk {chunk_idx}: freed {_freed:.2f}GB "
                      f"(allocated: {_mem_before:.2f}GB → {_mem_after:.2f}GB)", flush=True)

    # ── 显存优化：collect结束后强制清理GPU缓存 ──
    # 避免跨collect的碎片化累积导致下一个collect开始时显存不足
    gc.collect()
    torch.cuda.empty_cache()
    _mem_alloc_end = torch.cuda.memory_allocated(device) / 1024**3
    _mem_reserved_end = torch.cuda.memory_reserved(device) / 1024**3
    _mem_delta = _mem_alloc_end - _mem_alloc_start
    print(f"  [GPU MEM @ end] allocated={_mem_alloc_end:.2f}GB  reserved={_mem_reserved_end:.2f}GB  "
          f"delta={_mem_delta:+.2f}GB", flush=True)

    # ── Timing summary ──
    _t_rollout_total = time.time() - _t_rollout_start
    if _t_total_chunks > 0:
        print(
            f"\n[TIMING SUMMARY] rollout group {group_idx} | G={G} | {_t_total_chunks} chunks | total={_t_rollout_total:.2f}s\n"
            f"  vla_forward:    {_t_vla_forward:.2f}s ({100*_t_vla_forward/_t_rollout_total:.1f}%)  avg={_t_vla_forward/_t_total_chunks:.3f}s/chunk\n"
            f"  encoder+actor:  {_t_encoder_actor:.2f}s ({100*_t_encoder_actor/_t_rollout_total:.1f}%)  avg={_t_encoder_actor/_t_total_chunks:.3f}s/chunk\n"
            f"  store_records:  {_t_store_records:.2f}s ({100*_t_store_records/_t_rollout_total:.1f}%)  avg={_t_store_records/_t_total_chunks:.3f}s/chunk\n"
            f"  unnormalize:    {_t_unnormalize:.2f}s ({100*_t_unnormalize/_t_rollout_total:.1f}%)  avg={_t_unnormalize/_t_total_chunks:.3f}s/chunk\n"
            f"  env_step:       {_t_env_step:.2f}s ({100*_t_env_step/_t_rollout_total:.1f}%)  avg={_t_env_step/_t_total_chunks:.3f}s/chunk\n"
            f"  other/overhead: {_t_rollout_total - _t_vla_forward - _t_encoder_actor - _t_store_records - _t_unnormalize - _t_env_step:.2f}s"
        )

    # ── Finalize episodes ──
    for g in range(G):
        ep = episodes[g]
        if ep.finish_step == 0:  # timeout, never set
            ep.finish_step = len(ep.step_records)
            ep.env_steps = env_steps[g]
            ep.reward = 0.0

        if all_frames[g] and video_dir is not None:
            from AlphaBrain.training.reinforcement_learning.common.rollout import _save_video
            os.makedirs(video_dir, exist_ok=True)
            status = "success" if ep.success else "fail"
            vpath = os.path.join(video_dir,
                                 f"g{group_idx:04d}_e{g:02d}_t{task_id}_s{int(state_ids[g]):02d}_{status}.mp4")
            ep.video_path = _save_video(all_frames[g], vpath)

    return episodes


@torch.no_grad()
def action_token_collect_multitask_steplock(
    env_pool: PersistentEnvPool,
    frozen_vla,
    encoder: ActionTokenEncoderDecoder,
    actor: ActionTokenActor,
    critic: ActionTokenCritic,
    suite_name: str,
    task_ids: List[int],
    n_initial_states: int,
    action_norm_stats: dict,
    max_steps: int,
    chunk_len: int,
    G_per_task: int = 8,
    seed: int = 42,
    num_steps_wait: int = 10,
    device: str = "cuda",
    group_idx: int = 0,
    store_images: bool = False,
    group_size: int = 1,
    reward_coef: float = 1.0,
    actor_chunk_len: int = None,
    warmup_mode: bool = False,
) -> List[ActionTokenEpisode]:
    """
    Collect episodes for MULTIPLE tasks on ONE GPU in a single step-lock loop.

    All tasks' envs are merged into one batch for VLA forward — no per-task
    threading, no CUDA concurrency issues, maximum GPU batch utilization.

    Args:
        task_ids: list of task IDs to run on this GPU
        G_per_task: episodes per task
    Returns:
        flat list of all episodes across all tasks
    """
    if actor_chunk_len is None:
        actor_chunk_len = chunk_len
    exec_chunk_len = actor_chunk_len

    frozen_vla.eval()
    encoder.eval()
    actor.eval()

    n_tasks = len(task_ids)
    total_G = G_per_task * n_tasks

    # Assign states per task
    _rng = np.random.RandomState(seed + group_idx)
    all_state_ids = []
    all_task_labels = []  # which task each episode belongs to
    for tid in task_ids:
        num_unique = max(1, G_per_task // group_size)
        states = _rng.randint(0, n_initial_states, size=num_unique)
        states = np.repeat(states, group_size)[:G_per_task]
        all_state_ids.extend(states)
        all_task_labels.extend([tid] * G_per_task)

    n_workers = min(total_G, len(env_pool))

    # ── Phase 1: Reset all envs in parallel ──
    from concurrent.futures import as_completed as _as_completed
    obs_list = [None] * total_G
    with ThreadPoolExecutor(max_workers=total_G) as _pool:
        _futs = {_pool.submit(env_pool.reset_env, g, suite_name, all_task_labels[g], int(all_state_ids[g]), seed + g): g for g in range(total_G)}
        for _f in _as_completed(_futs):
            obs_list[_futs[_f]] = _f.result()
    print(f"  reset done: {total_G} envs (parallel)", flush=True)

    task_descriptions = [env_pool.envs[g].task_description for g in range(total_G)]

    # ── Phase 2: Warmup (parallel) ──
    if num_steps_wait > 0:
        with ThreadPoolExecutor(max_workers=total_G) as _pool:
            _futs = {_pool.submit(_env_dummy_steps, env_pool, g, num_steps_wait): g for g in range(total_G)}
            for _f in _as_completed(_futs):
                obs_list[_futs[_f]] = _f.result()

    # ── Phase 3: Step-lock main loop (ALL tasks merged) ──
    episodes = [ActionTokenEpisode(task_id=all_task_labels[g], state_idx=int(all_state_ids[g]))
                for g in range(total_G)]
    active = [True] * total_G
    env_steps = [0] * total_G
    max_chunks = max_steps // exec_chunk_len + 1

    _t_vla = 0.0
    _t_env = 0.0
    _n_chunks = 0

    for chunk_idx in range(max_chunks):
        active_ids = [g for g in range(total_G) if active[g]]
        if not active_ids:
            break

        # ── ONE batched VLA forward for ALL active envs across ALL tasks ──
        _t0 = time.time()
        batch_images = [[obs_list[g]["primary_image"], obs_list[g]["wrist_image"]] for g in active_ids]
        batch_instrs = [task_descriptions[g] for g in active_ids]
        batch_props = [np.array(obs_list[g]["state"], dtype=np.float32) for g in active_ids]

        with torch.autocast("cuda", dtype=torch.bfloat16):
            action_queries, vla_actions = frozen_vla.get_vla_action(
                batch_images=batch_images, instructions=batch_instrs)

        rl_tokens = encoder.encode(action_queries)
        # 显存优化：action_queries 使用完毕后立即删除（释放QwenVL中间隐状态的最后一个引用）
        del action_queries

        props_t = torch.tensor(np.array(batch_props), dtype=torch.float32).to(device)

        if actor_chunk_len < vla_actions.size(1):
            vla_actions_for_actor = vla_actions[:, :actor_chunk_len, :]
        else:
            vla_actions_for_actor = vla_actions

        if warmup_mode:
            actions_t = vla_actions_for_actor
            log_probs = torch.zeros(len(active_ids), device=device)
            values = torch.zeros(len(active_ids), device=device)
        else:
            actions_t, log_probs = actor(rl_tokens, vla_actions_for_actor, props_t, deterministic=False)
            values = critic(rl_tokens)
        _t_vla += time.time() - _t0

        actions_np = actions_t.cpu().numpy()
        vla_actions_cpu = vla_actions_for_actor.cpu()

        # Store records
        for i, g in enumerate(active_ids):
            episodes[g].step_records.append(ActionTokenStepRecord(
                rl_token=rl_tokens[i:i+1].cpu().squeeze(0),
                vla_action=vla_actions_cpu[i],
                action_taken=actions_t[i].detach().cpu(),
                old_log_prob=log_probs[i].item() if log_probs is not None else 0.0,
                value=values[i].item(),
                prop_state=torch.tensor(batch_props[i]),
            ))

        # 显存优化：GPU张量已拷贝到CPU，删除GPU版本释放显存
        del vla_actions_for_actor
        del vla_actions  # 显存优化：显式删除原始vla_actions引用
        del rl_tokens
        del actions_t
        if not warmup_mode:
            del log_probs
            del values

        # Unnormalize
        action_chunks_unnorm = [_unnormalize(actions_np[i], action_norm_stats) for i in range(len(active_ids))]

        # ── ALL envs execute chunk in parallel ──
        _t0 = time.time()
        with ThreadPoolExecutor(max_workers=len(active_ids)) as _pool:
            _futs = {}
            for i, g in enumerate(active_ids):
                _futs[_pool.submit(_env_step_chunk, env_pool, g, action_chunks_unnorm[i],
                                   exec_chunk_len, False)] = (i, g)
            for _f in _as_completed(_futs):
                i, g = _futs[_f]
                obs, reward, done, steps_taken, _ = _f.result()
                obs_list[g] = obs
                env_steps[g] += steps_taken
                if done or env_steps[g] >= max_steps:
                    active[g] = False
                    ep = episodes[g]
                    ep.success = bool(done and reward > 0.5)
                    ep.reward = reward_coef if ep.success else 0.0
                    ep.done_cache_idx = steps_taken
                    ep.finish_step = len(ep.step_records)
                    ep.env_steps = env_steps[g]
        _t_env += time.time() - _t0
        _n_chunks += 1

        # ── 显存优化：每5个chunk清理一次PyTorch缓存分配器 ──
        # QwenVL forward (output_hidden_states=True) 每次产生 ~290MB 中间隐状态。
        # PyTorch的缓存分配器不会自动归还显存，定期清理可避免碎片化累积导致OOM。
        if chunk_idx > 0 and chunk_idx % 5 == 0:
            _mem_before = torch.cuda.memory_allocated(device) / 1024**3
            gc.collect()
            torch.cuda.empty_cache()
            _mem_after = torch.cuda.memory_allocated(device) / 1024**3
            _freed = _mem_before - _mem_after
            if _freed > 0.1:  # 只打印释放超过100MB的情况
                print(f"  [GPU MEM] chunk {chunk_idx}: freed {_freed:.2f}GB "
                      f"(allocated: {_mem_before:.2f}GB → {_mem_after:.2f}GB)", flush=True)

    # ── 显存优化：collect结束后强制清理GPU缓存 ──
    gc.collect()
    torch.cuda.empty_cache()

    # Finalize
    for g in range(total_G):
        ep = episodes[g]
        if ep.finish_step == 0:
            ep.finish_step = len(ep.step_records)
            ep.env_steps = env_steps[g]
            ep.reward = 0.0

    if _n_chunks > 0:
        print(f"[MULTITASK TIMING] {n_tasks} tasks × {G_per_task} eps = {total_G} total | "
                     f"{_n_chunks} chunks | vla={_t_vla:.1f}s env={_t_env:.1f}s "
                     f"total={_t_vla+_t_env:.1f}s")

    return episodes
