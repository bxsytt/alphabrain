# 修复计划：QwenGR00T.get_action_queries() 签名不兼容

## 问题描述

运行 [`scripts/run_rl_scripts/run_rlat_5traj_alltasks.sh`](scripts/run_rl_scripts/run_rlat_5traj_alltasks.sh) 时，在 Phase 1（pretraining）阶段报错：

```
TypeError: Qwen_GR00T.get_action_queries() got an unexpected keyword argument 'batch_images'
```

## 错误根源

调用方 [`extract_action_queries_from_obs()`](AlphaBrain/training/reinforcement_learning/algos/RLActionToken/action_token_trainer.py:351) 使用 `batch_images` 和 `instructions` 关键字参数调用 `get_action_queries()`：

```python
aq = frozen_vla.get_action_queries(
    batch_images=batch_imgs,     # ← 引起错误的参数
    instructions=batch_instr,
)
```

这与 [`QwenOFT.get_action_queries()`](AlphaBrain/model/framework/QwenOFT.py:239) 的签名兼容：

```python
def get_action_queries(self, batch_images, instructions) -> torch.Tensor:
```

但 [`Qwen_GR00T.get_action_queries()`](AlphaBrain/model/framework/QwenGR00T.py:396-397) 的签名不同：

```python
@torch.inference_mode()
def get_action_queries(self, observations: List[dict]) -> torch.Tensor:
```

## 调用方数据分析

调用方 [`collect_observations_fast()`](AlphaBrain/training/reinforcement_learning/algos/RLActionToken/action_token_trainer.py:253) 返回的数据格式为：

```python
# observations 是 list of (images, instruction) 元组
# 其中 images = [primary_image_numpy, wrist_image_numpy]  （2 张 numpy 图像）
# 注意：不是单张图像，而是一个包含2个 numpy 数组的列表
```

所以在 [`extract_action_queries_from_obs()`](AlphaBrain/training/reinforcement_learning/algos/RLActionToken/action_token_trainer.py:371-377) 中：

```python
batch_imgs = [observations[i][0] for i in range(start, end)]
# batch_imgs 的格式是 List[List[numpy.ndarray]] 
# 即每个样本包含 [primary_numpy, wrist_numpy]
```

## 前置兼容性分析（已完成验证 ✅）

| 工具函数 | 能否处理 `List[List[numpy.ndarray]]` |
|---------|-----------------------------------|
| [`to_pil_preserve()`](deployment/model_server/tools/image_tools.py:61) | ✅ 支持递归嵌套，保留结构，`List[List[numpy]]` → `List[List[PIL]]` |
| [`resize_images()`](AlphaBrain/training/trainer_utils/trainer_tools.py:146) | ✅ 支持递归嵌套列表，保留结构 |
| [`build_qwenvl_inputs()`](AlphaBrain/model/modules/vlm/qwen2_5.py:212) | ✅ 期望 `List[List[PIL.Image]]`，与转换后格式完全匹配 |

## 精确修改方案

### 修改文件

仅修改 [`AlphaBrain/model/framework/QwenGR00T.py`](AlphaBrain/model/framework/QwenGR00T.py) 中的 `get_action_queries()` 方法（第396-425行）。

### 修改内容（精确 diff）

**修改前（第396-425行）：**

```python
    @torch.inference_mode()
    def get_action_queries(self, observations: List[dict]) -> torch.Tensor:
        """
        专门为 RLActionToken 提取特征的方法。
        它模拟 predict_action 的前半部分，只返回 VLM 输出的 hidden_states。
        """
        batch_images = [to_pil_preserve(obs["image"]) for obs in observations]
        instructions = [obs["lang"] for obs in observations]

        # 1. 图像缩放处理（保持与训练一致）
        train_obs_image_size = getattr(self.config.datasets.vla_data, "image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        # 2. 构建输入并推断特征
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images, 
            instructions=instructions
        )
        
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            # 返回最后一层隐藏状态 [B, L, H]
            # 这就是 RL Token 训练器需要的 "Action Queries"
            return qwenvl_outputs.hidden_states[-1]
```

**修改后：**

```python
    @torch.inference_mode()
    def get_action_queries(
        self,
        batch_images: List = None,
        instructions: List[str] = None,
    ) -> torch.Tensor:
        """
        专门为 RLActionToken 提取特征的方法。
        它模拟 predict_action 的前半部分，只返回 VLM 输出的 hidden_states。

        Args:
            batch_images: List of image inputs. Each element is [primary_img, wrist_img]
                          (numpy arrays), already processed by the caller.
            instructions: List of instruction strings.

        Returns:
            action_queries: (B, L, H) tensor of VLM hidden states.
        """
        # batch_images 是 List[List[numpy.ndarray]]，内层为 [primary, wrist]
        # to_pil_preserve 递归转换所有 numpy 为 PIL，保留嵌套结构
        batch_images = to_pil_preserve(batch_images)  # → List[List[PIL.Image]]

        # 1. 图像缩放处理（保持与训练一致）
        train_obs_image_size = getattr(self.config.datasets.vla_data, "image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        # 2. 构建输入并推断特征
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images, 
            instructions=instructions
        )
        
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            # 返回最后一层隐藏状态 [B, L, H]
            # 这就是 RL Token 训练器需要的 "Action Queries"
            return qwenvl_outputs.hidden_states[-1]
```

### 关键变化总结

| 项目 | 修改前 | 修改后 |
|------|--------|--------|
| 方法签名 | `(self, observations: List[dict])` | `(self, batch_images=None, instructions=None)` |
| 图像格式 | `observations` 中 `obs["image"]`（单张 PIL） | `batch_images` 为 `List[List[numpy]]`（多张） |
| 转换方式 | 逐项 `to_pil_preserve(obs["image"])` | 整体 `to_pil_preserve(batch_images)`（递归） |
| 与 QwenOFT 兼容性 | ❌ 不兼容 | ✅ 完全兼容 |
| 调用方是否需要修改 | - | ❌ 不需要修改 |

## 影响范围分析

- **修改文件**：仅 [`AlphaBrain/model/framework/QwenGR00T.py`](AlphaBrain/model/framework/QwenGR00T.py) 一个文件
- **不受影响**：
  - [`action_token_trainer.py`](AlphaBrain/training/reinforcement_learning/algos/RLActionToken/action_token_trainer.py) — 不用改，已是正确调用方式
  - [`QwenOFT.py`](AlphaBrain/model/framework/QwenOFT.py) — 其他框架，不受影响
  - 其他所有调用 `get_action_queries` 的地方 — 都已使用 `batch_images`/`instructions` 签名
- **所有调用方**（共2处）均使用 `batch_images=batch_images, instructions=instructions` 模式，与修改后签名完全一致

## 验证方法

修复后重新运行：

```bash
cd /home/zlb/embody_project/AlphaBrain
bash scripts/run_rl_scripts/run_rlat_5traj_alltasks.sh
```

观察 Phase 1 的 pretraining 是否能正常执行通过，不再出现 `TypeError`。

## 数据流对比图

```mermaid
flowchart LR
    A[collect_observations_fast] -->|"(images, instruction) tuples"| B[extract_action_queries_from_obs]
    B -->|"batch_images=List[List[numpy]]<br>instructions=List[str]"| C[get_action_queries]
    C -->|"to_pil_preserve 递归转换"| D[List[List[PIL.Image]]]
    D -->|"resize_images 递归缩放"| E[build_qwenvl_inputs]
    E -->|"Qwen2.5-VL forward"| F[hidden_states[-1]]
    F -->|"(B, L, H)"| G[Encoder-Decoder pretrain]
```
