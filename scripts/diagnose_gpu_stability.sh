#!/bin/bash
# ============================================================
# GPU 稳定性诊断脚本
# 用于检测 GPU 硬件/驱动问题，判断是否掉卡、显存错误等
#
# 用法：
#   bash scripts/diagnose_gpu_stability.sh          # 快速诊断
#   bash scripts/diagnose_gpu_stability.sh --stress  # 压力测试（30分钟）
#   bash scripts/diagnose_gpu_stability.sh --stress --duration 60  # 60分钟压力测试
# ============================================================

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

STRESS_MODE=false
DURATION=30  # 默认压力测试时长（分钟）

# 解析参数
for arg in "$@"; do
    case "$arg" in
        --stress) STRESS_MODE=true ;;
        --duration) ;;
        *) if [[ "$arg" =~ ^[0-9]+$ ]]; then DURATION=$arg; fi ;;
    esac
done

echo -e "${CYAN}============================================${NC}"
echo -e "${CYAN}  GPU 稳定性诊断脚本${NC}"
echo -e "${CYAN}============================================${NC}"
echo ""

# ── 1. 检查 nvidia-smi ──────────────────────────────────
echo -e "${YELLOW}[1/6] 检查 nvidia-smi...${NC}"
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}  ✗ nvidia-smi 未找到！驱动未安装或损坏${NC}"
    exit 1
fi

SMI_OUTPUT=$(nvidia-smi 2>&1)
if echo "$SMI_OUTPUT" | grep -qi "Unable to determine"; then
    echo -e "${RED}  ✗ GPU 掉卡检测到！${NC}"
    echo "$SMI_OUTPUT" | grep -i "unable"
    echo ""
    echo -e "${YELLOW}  建议：${NC}"
    echo "    1. sudo reboot 重启系统"
    echo "    2. 检查 GPU 电源线连接"
    echo "    3. 重新插拔 GPU"
    echo "    4. 检查电源供电是否充足（RTX 3090 峰值 350W）"
else
    echo -e "${GREEN}  ✓ nvidia-smi 正常${NC}"
fi
echo ""

# ── 2. 显示 GPU 信息 ────────────────────────────────────
echo -e "${YELLOW}[2/6] GPU 信息概览${NC}"
nvidia-smi --query-gpu=index,name,temperature.gpu,power.draw,power.limit,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null | while IFS=',' read -r idx name temp power power_limit mem_used mem_total util; do
    echo "  GPU $idx: $name"
    echo "    温度: ${temp}°C"
    echo "    功耗: ${power}/${power_limit}"
    echo "    显存: ${mem_used}/${mem_total}"
    echo "    利用率: ${util}"
    echo ""
done

# ── 3. 检查 PyTorch CUDA 可用性 ─────────────────────────
echo -e "${YELLOW}[3/6] 检查 PyTorch CUDA...${NC}"
python3 -c "
import torch
import sys

n_gpus = torch.cuda.device_count()
print(f'  PyTorch 检测到 {n_gpus} 块 GPU')
for i in range(n_gpus):
    name = torch.cuda.get_device_name(i)
    cap = torch.cuda.get_device_capability(i)
    mem = torch.cuda.get_device_properties(i).total_memory / 1e9
    print(f'  GPU {i}: {name} | Compute Capability: {cap[0]}.{cap[1]} | VRAM: {mem:.1f} GB')

# 简单 CUDA 内核测试
try:
    x = torch.randn(1000, 1000, device='cuda:0')
    y = torch.randn(1000, 1000, device='cuda:0')
    z = torch.mm(x, y)
    print(f'  ✓ CUDA 内核测试通过 (矩阵乘法结果 shape={z.shape})')
    del x, y, z
    torch.cuda.empty_cache()
except Exception as e:
    print(f'  ✗ CUDA 内核测试失败: {e}')
    sys.exit(1)
" 2>&1 || echo -e "${RED}  ✗ PyTorch CUDA 检测失败${NC}"
echo ""

# ── 4. 检查驱动和 CUDA 版本 ─────────────────────────────
echo -e "${YELLOW}[4/6] 驱动和 CUDA 版本${NC}"
DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
CUDA_VER=$(nvidia-smi | grep "CUDA Version" | awk '{print $NF}')
PYTORCH_CUDA_VER=$(python3 -c "import torch; print(torch.version.cuda)" 2>/dev/null || echo "N/A")
echo "  NVIDIA 驱动版本: $DRIVER_VER"
echo "  CUDA 运行时版本: $CUDA_VER"
echo "  PyTorch CUDA 版本: $PYTORCH_CUDA_VER"
echo ""

# ── 5. 检查系统日志中的 GPU 错误 ────────────────────────
echo -e "${YELLOW}[5/6] 检查系统日志中的 GPU 相关错误...${NC}"
if command -v dmesg &> /dev/null; then
    GPU_ERRORS=$(sudo dmesg 2>/dev/null | grep -iE "nvidia|gpu|pci.*error|card.*lost|unknown.*error" | tail -20 || true)
    if [ -n "$GPU_ERRORS" ]; then
        echo -e "${RED}  发现 GPU 相关错误：${NC}"
        echo "$GPU_ERRORS"
    else
        echo -e "${GREEN}  ✓ 系统日志中未发现 GPU 错误${NC}"
    fi
else
    echo "  (无法访问 dmesg，跳过)"
fi
echo ""

# ── 6. 压力测试（可选） ─────────────────────────────────
if [ "$STRESS_MODE" = true ]; then
    echo -e "${YELLOW}[6/6] GPU 压力测试（${DURATION} 分钟）...${NC}"
    echo -e "${YELLOW}  这将运行密集的 CUDA 计算来检测硬件稳定性${NC}"
    echo ""

    python3 -c "
import torch
import time
import sys

n_gpus = torch.cuda.device_count()
duration_sec = $DURATION * 60
start_time = time.time()
iter_count = 0
error_count = 0

print(f'  开始 {n_gpus} GPU 压力测试，持续 {duration_sec//60} 分钟...')
print()

while time.time() - start_time < duration_sec:
    try:
        for gpu_id in range(n_gpus):
            # 大矩阵乘法（密集计算）
            size = 4096
            a = torch.randn(size, size, device=f'cuda:{gpu_id}')
            b = torch.randn(size, size, device=f'cuda:{gpu_id}')
            for _ in range(5):
                c = torch.mm(a, b)
                d = torch.sigmoid(c)
                e = torch.tanh(d)
                _ = torch.mm(e, a)
            del a, b, c, d, e
            torch.cuda.synchronize(f'cuda:{gpu_id}')
        
        # 每 10 次迭代清理一次缓存
        iter_count += 1
        if iter_count % 10 == 0:
            torch.cuda.empty_cache()
            elapsed = time.time() - start_time
            remaining = duration_sec - elapsed
            print(f'    运行中... {int(elapsed//60)}分{int(elapsed%60)}秒 | '
                  f'剩余 {int(remaining//60)}分{int(remaining%60)}秒 | '
                  f'迭代 {iter_count} 次 | 错误 {error_count} 次', end='\r')
    
    except Exception as e:
        error_count += 1
        print()
        print(f'  [!] 错误 #{error_count}: {e}')
        torch.cuda.empty_cache()
        time.sleep(2)
        if error_count >= 5:
            print(f'  {RED}✗ 压力测试失败：连续 5 次错误${NC}')
            sys.exit(1)

print()
print()
elapsed_total = time.time() - start_time
if error_count == 0:
    print(f'  {GREEN}✓ 压力测试通过！${NC}')
    print(f'    运行时长: {int(elapsed_total//60)}分{int(elapsed_total%60)}秒')
    print(f'    总迭代: {iter_count} 次')
    print(f'    错误: 0 次')
else:
    print(f'  {YELLOW}⚠ 压力测试完成，但有 {error_count} 次错误${NC}')
" 2>&1 || echo -e "${RED}  ✗ 压力测试异常退出${NC}"
else
    echo -e "${YELLOW}[6/6] 跳过压力测试（使用 --stress 启用）${NC}"
    echo "  用法: bash $0 --stress [--duration 60]"
fi

echo ""
echo -e "${CYAN}============================================${NC}"
echo -e "${CYAN}  诊断完成${NC}"
echo -e "${CYAN}============================================${NC}"
echo ""
echo "快速修复建议："
echo "  1. 重启: sudo reboot"
echo "  2. 仅用 GPU1 训练: CUDA_VISIBLE_DEVICES=1"
echo "  3. 降级驱动: 安装 550 或 535 LTS 版本"
echo "  4. 检查电源/PCIe连接"
