#!/bin/bash
# ============================================
# GCN 图神经网络风控项目 - 环境安装脚本
# ============================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'
PASS=0
FAIL=0

pass_step() { echo -e "  ${GREEN}✅${NC} $1"; PASS=$((PASS+1)); }
fail_step() { echo -e "  ${RED}❌${NC} $1"; FAIL=$((FAIL+1)); }
warn_step() { echo -e "  ${YELLOW}⚠️${NC}  $1"; }

echo "======================================"
echo "  GCN 图神经网络风控项目 - 环境安装"
echo "======================================"
echo ""

# ========================================
# Step 1: 检查 Python
# ========================================
echo "[1/5] 检查 Python 环境..."
PY_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null)
PY_MAJOR=$(echo "$PY_VER" | cut -d. -f1)

if [ "$PY_MAJOR" = "3" ]; then
    pass_step "Python $PY_VER"
else
    fail_step "需要 Python 3.9+，当前: $PY_VER"
    exit 1
fi

# ========================================
# Step 2: 创建虚拟环境
# ========================================
echo ""
echo "[2/5] 创建虚拟环境..."
if [ -d ".venv" ]; then
    warn_step "虚拟环境已存在，删除并重建..."
    rm -rf .venv
fi

python3 -m venv .venv 2>/dev/null && pass_step "虚拟环境创建完成" || {
    fail_step "虚拟环境创建失败"
    exit 1
}
source .venv/bin/activate
pip install --upgrade pip -q 2>/dev/null

# ========================================
# Step 3: 安装 PyTorch
# ========================================
echo ""
echo "[3/5] 安装 PyTorch..."
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cpu -q 2>/dev/null && {
    TORCH_VER=$(python3 -c "import torch; print(torch.__version__.split('+')[0])")
    pass_step "PyTorch $TORCH_VER"
} || {
    fail_step "PyTorch 安装失败"
    exit 1
}

# ========================================
# Step 4: 安装 PyTorch Geometric
# ========================================
echo ""
echo "[4/5] 安装 PyTorch Geometric..."

TORCH_MAJOR=$(echo $TORCH_VER | cut -d. -f1)
TORCH_MINOR=$(echo $TORCH_VER | cut -d. -f2)

PYG_OK=true

# 尝试预编译 wheel
pip install torch_scatter torch_sparse torch_cluster \
    -f "https://data.pyg.org/whl/torch-${TORCH_MAJOR}.${TORCH_MINOR}.0+cpu.html" -q 2>/dev/null || PYG_OK=false

if [ "$PYG_OK" = false ]; then
    warn_step "预编译 wheel 不可用 (Python $PY_VER + Torch $TORCH_VER)，尝试源码安装..."
    # 尝试不依赖预编译 wheel 的方式
    pip install torch-scatter torch-sparse torch-cluster -q 2>/dev/null && PYG_OK=true || PYG_OK=false
fi

if [ "$PYG_OK" = true ]; then
    pip install torch-geometric -q 2>/dev/null && PYG_OK=true || PYG_OK=false
fi

if [ "$PYG_OK" = true ]; then
    PYG_VER=$(python3 -c "import torch_geometric; print(torch_geometric.__version__)" 2>/dev/null)
    pass_step "PyTorch Geometric $PYG_VER"
else
    warn_step "PyTorch Geometric 安装失败（图模型功能暂不可用）"
    echo "          GCN/GAT/GraphSAGE 模型需要 PyG，但 train.py 等工具模块仍可正常使用"
    echo "          你可以稍后尝试: pip install torch-geometric --no-build-isolation"
fi

# ========================================
# Step 5: 安装其他依赖
# ========================================
echo ""
echo "[5/5] 安装其他依赖..."

# 逐个安装核心依赖（不因单个失败而中断）
for pkg in numpy pandas scipy scikit-learn matplotlib seaborn \
           umap-learn networkx jupyter ipykernel tqdm flask pytest; do
    pip install "$pkg" -q 2>/dev/null && pass_step "$pkg" || fail_step "$pkg"
done

# ========================================
# 验证
# ========================================
echo ""
echo "======================================"
echo "  安装验证"
echo "======================================"
echo ""

python3 << PYEOF
import sys
sys.path.insert(0, '.')

checks = []

# Core packages
for mod, name in [
    ("numpy", "NumPy"),
    ("pandas", "Pandas"),
    ("sklearn", "Scikit-learn"),
    ("matplotlib", "Matplotlib"),
    ("seaborn", "Seaborn"),
    ("networkx", "NetworkX"),
    ("flask", "Flask"),
]:
    try:
        __import__(mod)
        checks.append((True, f"✅ {name}"))
    except ImportError:
        checks.append((False, f"❌ {name}"))

# PyTorch
try:
    import torch
    checks.append((True, f"✅ PyTorch {torch.__version__}"))
except ImportError:
    checks.append((False, "❌ PyTorch"))

# PyG (optional)
try:
    import torch_geometric
    checks.append((True, f"✅ PyTorch Geometric {torch_geometric.__version__}"))
except ImportError:
    checks.append((False, "⚠️  PyTorch Geometric 未安装（图模型功能不可用）"))

# Project modules (non-PyG)
try:
    from src.train import FocalLoss, EarlyStopping
    from src.utils import set_seed
    from src.evaluate import compute_metrics
    from src.data.generate_fraud_data import generate_fraud_dataset
    checks.append((True, "✅ GCN 项目模块"))
except Exception as e:
    checks.append((False, f"❌ 项目模块: {e}"))

for ok, msg in checks:
    print(f"  {msg}")

all_ok = all(ok for ok, _ in checks)

print()
if all_ok:
    print("🎉 所有模块就绪，可以开始学习了！")
else:
    print("⚠️  部分模块未安装，上面标 ❌ 的项需要手动处理")
    print("   核心的训练/评估/数据生成模块不受影响")

PYEOF

echo ""
echo "======================================"
echo "  安装完成 ($PASS 成功, $FAIL 失败)"
echo "======================================"
echo ""
echo "  激活环境:  source .venv/bin/activate"
echo "  快速开始:  jupyter notebook notebooks/01-GCN-Cora节点分类入门.ipynb"
echo ""
