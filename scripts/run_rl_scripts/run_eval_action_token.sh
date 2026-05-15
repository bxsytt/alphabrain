#!/bin/bash
# Eval an RLActionToken run on libero_goal task(s). Supports both single-task
# and multi-task evaluation. Results are aggregated into
# <RUN_DIR>/eval_<ITER>/summary.json.
#
# Usage:
#   bash scripts/run_rl_scripts/run_eval_action_token.sh [RUN_DIR] [GPU_IDS] [TASK_IDS]
#
# Examples:
#   # Single-task eval (default: task 0 on GPU 0)
#   bash scripts/run_rl_scripts/run_eval_action_token.sh
#
#   # Single-task eval with explicit args
#   bash scripts/run_rl_scripts/run_eval_action_token.sh \
#       results/my_run/rl_offpolicy 0 0
#
#   # Multi-task eval (tasks 0,1,2 on GPU 0, tasks 3,4 on GPU 1)
#   TASK_IDS="0,1,2,3,4" GPU_IDS="0,1" \
#       bash scripts/run_rl_scripts/run_eval_action_token.sh
#
# Edit VLA_CKPT / RUN_DIR / ITER / GPU_IDS / TASK_IDS below, or override via argv.
set -euo pipefail
cd "${ALPHABRAIN_ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"  # e.g. /path/to/AlphaBrain

[ -f .env ] && { set -a; source .env; set +a; }

# AlphaBrain isn't pip-installed in the vla env; when launching a script by path
# (not `-m`), sys.path[0] is the script's dir, so the AlphaBrain package isn't
# importable. Prepend repo root to PYTHONPATH to fix.
export PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}"

export LIBERO_PYTHON="${LIBERO_PYTHON:-/path/to/envs/libero/bin/python}"
export LIBERO_HOME="${LIBERO_HOME:-/path/to/LIBERO}"
export TOKENIZERS_PARALLELISM=false
export MUJOCO_GL="${MUJOCO_GL:-egl}"

# ── Edit these before running ────────────────────────────────
VLA_CKPT="data/final_run"
RUN_DIR=${1:-"results/action_token_training_TD3/singletask_resume_v2_0514_1122/rl_offpolicy"}

ITER="iter_00140"
GPU_IDS=${2:-"1"}          # Use GPU 1 (GPU 0 may have lingering processes)
TASK_IDS=${3:-"0"}         # comma-separated list; default: only task 0 (single-task)
SAVE_EVAL_VIDEOS=${SAVE_EVAL_VIDEOS:-true}   # set to false to skip video saving
# ─────────────────────────────────────────────────────────────

ACTION_TOKEN_CKPT="${RUN_DIR}/checkpoints/rl_offpolicy_${ITER}"
OUT_DIR="${RUN_DIR}/eval_${ITER}"
mkdir -p "${OUT_DIR}"

# Remove stale shard JSONs from prior runs so aggregate never silently reads
# pre-refactor results if the fresh eval fails.
rm -f "${OUT_DIR}"/shard_*.json "${OUT_DIR}"/summary.json

# Parse task IDs into an array, then split across GPUs round-robin.
IFS=',' read -r -a TASK_ARRAY <<< "${TASK_IDS}"
N_TASKS=${#TASK_ARRAY[@]}

IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
N_GPUS=${#GPU_ARRAY[@]}

# Build comma-separated task groups per GPU (round-robin distribution).
declare -a GPU_TASK_GROUPS
for g in $(seq 0 $((N_GPUS - 1))); do
    GPU_TASK_GROUPS[g]=""
done
for i in "${!TASK_ARRAY[@]}"; do
    g=$(( i % N_GPUS ))
    if [ -z "${GPU_TASK_GROUPS[g]}" ]; then
        GPU_TASK_GROUPS[g]="${TASK_ARRAY[i]}"
    else
        GPU_TASK_GROUPS[g]="${GPU_TASK_GROUPS[g]},${TASK_ARRAY[i]}"
    fi
done

N_EPS=50
NUM_WORKERS=2             # fewer workers = lower peak GPU memory
SUITE=libero_goal
ARCH_ARGS="--bottleneck_dim 256 --encoder_layers 2 --encoder_heads 4 --actor_hidden_dim 512 --ref_dropout 0.5 --fixed_std 0.1 --prop_dim 8"

if [ ! -d "${ACTION_TOKEN_CKPT}" ]; then
    echo "ERROR: RLActionToken ckpt not found: ${ACTION_TOKEN_CKPT}" >&2
    exit 1
fi
if [ ! -d "${VLA_CKPT}" ]; then
    echo "ERROR: VLA ckpt not found: ${VLA_CKPT}" >&2
    exit 1
fi

echo "============================================================"
echo " Eval RLActionToken (${ITER}) | ${N_EPS} eps/task, ${NUM_WORKERS} workers/shard"
echo "   ckpt: ${ACTION_TOKEN_CKPT}"
echo "   vla:  ${VLA_CKPT}"
echo "   out:  ${OUT_DIR}"
for g in $(seq 0 $((N_GPUS - 1))); do
    tasks="${GPU_TASK_GROUPS[g]}"
    [ -z "$tasks" ] && continue
    echo "   GPU ${GPU_ARRAY[g]}: tasks [${tasks}]"
done
echo "============================================================"

# Cleanup on any exit (Ctrl-C, error, normal). Registered BEFORE launching
# anything so partial state is still cleaned up on early failures.
TAIL_PIDS=()
SHARD_PIDS=()
_cleanup () {
    trap - EXIT INT TERM   # avoid re-entry
    # Kill each shard's ENTIRE process group (python + spawned libero_env_worker
    # subprocesses). `setsid` below gives each shard PGID == shard's PID, so
    # `kill -- -PID` signals the whole group.
    for pid in "${SHARD_PIDS[@]}"; do
        kill -TERM -- "-${pid}" 2>/dev/null || true
    done
    # Kill log-mirror tails (pipeline subshells).
    [ "${#TAIL_PIDS[@]}" -gt 0 ] && kill "${TAIL_PIDS[@]}" 2>/dev/null || true
    # Belt-and-suspenders: kill any orphaned libero_env_worker owned by us.
    pkill -TERM -u "$(id -u)" -f "AlphaBrain/training/reinforcement_learning/envs/libero_env_worker" 2>/dev/null || true
}
trap _cleanup EXIT INT TERM

# ── run_shard: launch one eval_libero.py process in the background ──
run_shard() {
    local gpu_id="$1"
    local tasks="$2"
    local tag="$3"
    local log="${OUT_DIR}/shard_${tag}.log"
    local results_json="${OUT_DIR}/shard_${tag}.json"

    local video_args=""
    if [ "${SAVE_EVAL_VIDEOS}" = "true" ]; then
        video_args="--video_dir ${OUT_DIR}/videos"
    fi

    CUDA_VISIBLE_DEVICES="${gpu_id}" python AlphaBrain/training/reinforcement_learning/eval/eval_libero.py \
        --action_token_ckpt "${ACTION_TOKEN_CKPT}" \
        --vla_ckpt          "${VLA_CKPT}" \
        --suite             "${SUITE}" \
        --n_eps_per_task    "${N_EPS}" \
        --gpu               0 \
        --task_ids          "${tasks}" \
        --num_workers       "${NUM_WORKERS}" \
        --results_json      "${results_json}" \
        ${video_args} \
        ${ARCH_ARGS} \
        > "${log}" 2>&1 &

    local pid=$!
    # Declare globally so the wait loop below can read PID_<tag>.
    declare -g "PID_${tag}=${pid}"
    SHARD_PIDS+=("${pid}")
    echo "[shard ${tag}] launched PID ${pid} — log: ${log}"
}

SHARD_TAGS=()
for g in $(seq 0 $((N_GPUS - 1))); do
    tasks="${GPU_TASK_GROUPS[g]}"
    [ -z "$tasks" ] && continue
    gpu_id="${GPU_ARRAY[g]}"
    # Map gpu index → letter tag (a, b, c, ...)
    tag=$(echo "abcdefghijklmnopqrstuvwxyz" | cut -c$((g + 1)))
    SHARD_TAGS+=("$tag")
    run_shard "${gpu_id}" "${tasks}" "${tag}"
done

# Mirror each shard log to main stdout with a `[tag] ` prefix so progress is
# visible live. Tails are killed by _cleanup on any exit.
for tag in "${SHARD_TAGS[@]}"; do
    _log="${OUT_DIR}/shard_${tag}.log"
    : > "${_log}"   # create empty file so tail -F starts following immediately
    (tail -F -q -n +1 "${_log}" 2>/dev/null | sed -u "s/^/[${tag}] /") &
    TAIL_PIDS+=($!)
done

# Wait on each PID individually and abort if any shard failed — unlike
# bare `wait`, this propagates non-zero exit codes from background children.
fail=0
for tag in "${SHARD_TAGS[@]}"; do
    pid_var="PID_${tag}"
    pid=${!pid_var}
    if ! wait "${pid}"; then
        echo "ERROR: shard ${tag} (pid ${pid}) FAILED — see ${OUT_DIR}/shard_${tag}.log" >&2
        fail=1
    fi
done
# Let tails flush last lines, then stop them.
sleep 1
kill "${TAIL_PIDS[@]}" 2>/dev/null || true
[ "${fail}" -eq 0 ] || { echo "Aborting before aggregate (some shards failed)." >&2; exit 1; }
echo "[shards] all done."

echo "============================================================"
echo " Aggregating per-task SR"
echo "============================================================"
python AlphaBrain/training/reinforcement_learning/eval/aggregate_shards.py \
    --out_dir  "${OUT_DIR}" \
    --action_token_ckpt "${ACTION_TOKEN_CKPT}" \
    --vla_ckpt "${VLA_CKPT}" \
    --suite    "${SUITE}" \
    --n_eps    "${N_EPS}"
