"""
LIBERO environment proxy — runs in the VLA Python environment.

Spawns `libero_env_worker.py` as a subprocess using `LIBERO_PYTHON`
(the separate conda env that has `libero` installed), then communicates
via stdin/stdout with length-prefixed msgpack messages.

Usage matches the original direct API:
    env = LiberoEnv(suite_name, task_id, seed)
    obs = env.reset(initial_state_idx=0)
    obs, reward, done = env.step(action_7d)
    env.close()
"""

import io
import os
import struct
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple

import msgpack
import numpy as np
from PIL import Image


# ── Wire protocol helpers ───────────────────────────────────────────────────────

def _write_msg(proc: subprocess.Popen, obj: dict):
    data = msgpack.packb(obj, use_bin_type=True)
    proc.stdin.write(struct.pack("<I", len(data)))
    proc.stdin.write(data)
    proc.stdin.flush()


_READ_MSG_TIMEOUT = 30.0  # seconds; 防止子进程卡死导致父进程永久阻塞


def _read_msg(proc: subprocess.Popen, timeout: float = _READ_MSG_TIMEOUT) -> dict:
    """读取子进程的 msgpack 响应，带超时机制防止死锁。"""
    import select

    # 等待子进程 stdout 可读，最多 timeout 秒
    readable, _, _ = select.select([proc.stdout], [], [], timeout)
    if not readable:
        # 超时 —— 子进程可能卡在 stderr buffer 或其他原因
        _stderr_snapshot = ""
        try:
            if proc.stderr is not None:
                _stderr_snapshot = proc.stderr.read().decode(errors="replace")[:2000]
        except Exception:
            pass
        raise TimeoutError(
            f"LIBERO worker response timeout ({timeout}s). "
            f"stderr (last 2000 chars):\n{_stderr_snapshot}"
        )

    raw_len = proc.stdout.read(4)
    if not raw_len:
        _stderr_content = ""
        try:
            if proc.stderr is not None:
                _stderr_content = proc.stderr.read().decode(errors="replace")
        except Exception:
            pass
        raise RuntimeError(f"LIBERO worker exited unexpectedly.\nWorker stderr:\n{_stderr_content}")
    length = struct.unpack("<I", raw_len)[0]
    data = proc.stdout.read(length)
    return msgpack.unpackb(data, raw=False)


def _bytes_to_pil(b: bytes) -> Image.Image:
    return Image.open(io.BytesIO(b))


# ── Worker path ─────────────────────────────────────────────────────────────────

_WORKER_SCRIPT = str(Path(__file__).parent / "libero_env_worker.py")


# ── LiberoEnv ───────────────────────────────────────────────────────────────────

class LiberoEnv:
    """
    Proxy to a LIBERO environment running in a separate Python process.

    The worker process is started once per LiberoEnv instance and reused
    across reset() calls (different tasks can be loaded with reset).
    """

    def __init__(
        self,
        libero_python: Optional[str] = None,
    ):
        """
        Args:
            libero_python: Path to the LIBERO conda env Python binary.
                           Defaults to LIBERO_PYTHON env var, then 'python'.
        """
        python_bin = (
            libero_python
            or os.environ.get("LIBERO_PYTHON", "python")
        )

        # Inherit the current env so LIBERO can find its own packages.
        # Inject LIBERO_HOME into PYTHONPATH so the editable install is not required.
        worker_env = os.environ.copy()
        # Strip debugpy/pydevd env vars so the worker subprocess runs
        # normally (not intercepted by debugpy --multiprocess), which
        # would break the msgpack-based stdin/stdout protocol.
        for _key in list(worker_env.keys()):
            _key_upper = _key.upper()
            if "PYDEVD" in _key_upper or "DEBUGPY" in _key_upper:
                del worker_env[_key]
        libero_home = os.environ.get("LIBERO_HOME", "")
        if libero_home:
            existing = worker_env.get("PYTHONPATH", "")
            worker_env["PYTHONPATH"] = f"{libero_home}:{existing}" if existing else libero_home

        # ── 修复子进程 stderr pipe buffer 死锁 ──
        # 问题: LIBERO 环境初始化时输出大量日志到 stderr，pipe buffer 很快填满，
        #       导致子进程阻塞在 write(stderr) 上，永远无法写到 stdout 响应。
        #       父进程同时阻塞在 _read_msg(stdout) 上，形成死锁。
        # 解决: stderr 重定向到 DEVNULL（避免 buffer 死锁），
        #       同时给 _read_msg 添加超时防止永久阻塞。
        stderr_target = subprocess.DEVNULL
        self._proc = subprocess.Popen(
            [python_bin, _WORKER_SCRIPT],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr_target,
            env=worker_env,
        )

        self.task_description: str = ""
        self.max_steps: int = 300
        self._closed = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(
        self,
        suite_name: str,
        task_id: int,
        initial_state_idx: int = 0,
        seed: int = 42,
    ) -> dict:
        """
        Reset the environment to a specific task and initial state.

        Returns obs dict:
          - "primary_image"  : PIL.Image
          - "wrist_image"    : PIL.Image
          - "state"          : np.ndarray (8,)
        """
        _write_msg(self._proc, {
            "cmd":               "reset",
            "task_suite":        suite_name,
            "task_id":           task_id,
            "initial_state_idx": initial_state_idx,
            "seed":              seed,
        })
        resp = _read_msg(self._proc)
        _check_resp(resp)

        self.task_description = resp["task_description"]
        self.max_steps = resp["max_steps"]
        return _parse_obs(resp["obs"])

    def step(self, action_7d: np.ndarray) -> Tuple[dict, float, bool]:
        """
        Execute one env step.

        Returns:
            obs_dict  : parsed observation
            reward    : 0.0 / 1.0
            done      : episode termination flag
        """
        _write_msg(self._proc, {"cmd": "step", "action": action_7d.tolist()})
        resp = _read_msg(self._proc)
        _check_resp(resp)
        return _parse_obs(resp["obs"]), float(resp["reward"]), bool(resp["done"])

    def close(self):
        if not self._closed:
            try:
                _write_msg(self._proc, {"cmd": "close"})
            except Exception:
                pass
            try:
                self._proc.stdin.close()
            except Exception:
                pass
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait()
            self._closed = True

    def __del__(self):
        self.close()


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _check_resp(resp: dict):
    if resp.get("status") != "ok":
        raise RuntimeError(f"LIBERO worker error: {resp.get('message', resp)}")


def _parse_obs(obs_raw: dict) -> dict:
    return {
        "primary_image": _bytes_to_pil(obs_raw["primary"]),
        "wrist_image":   _bytes_to_pil(obs_raw["wrist"]),
        "state":         np.array(obs_raw["state"], dtype=np.float32),
    }


# ------------------------------------------------------------------
# Suite info helper (no env needed — just metadata)
# ------------------------------------------------------------------

MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object":  280,
    "libero_goal":    320,   # debug: 16 steps → 2 chunks (was 320 → 40 chunks)
    "libero_10":      520,
    "libero_90":      400,
}


def get_suite_info(suite_name: str, libero_python: Optional[str] = None) -> dict:
    """
    Query task count and task names from the LIBERO worker without
    opening an environment.

    Returns: {"n_tasks": int, "task_names": [str, ...]}
    """
    python_bin = libero_python or os.environ.get("LIBERO_PYTHON", "python")
    script = (
        "import sys; _real_stdout = sys.stdout; sys.stdout = sys.stderr; "
        "from libero.libero import benchmark; "
        f"s = benchmark.get_benchmark_dict()['{suite_name}'](); "
        "sys.stdout = _real_stdout; "
        "import json; "
        "json.dump({'n_tasks': s.n_tasks, "
        "'task_names': [s.get_task(i).language for i in range(s.n_tasks)]}, sys.stdout)"
    )
    run_env = os.environ.copy()
    libero_home = os.environ.get("LIBERO_HOME", "")
    if libero_home:
        existing = run_env.get("PYTHONPATH", "")
        run_env["PYTHONPATH"] = f"{libero_home}:{existing}" if existing else libero_home
    result = subprocess.run(
        [python_bin, "-c", script],
        capture_output=True, text=True, timeout=30,
        env=run_env,
    )
    if result.returncode != 0:
        raise RuntimeError(f"get_suite_info failed:\n{result.stderr}")
    import json
    return json.loads(result.stdout)
