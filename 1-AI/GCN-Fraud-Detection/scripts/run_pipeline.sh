#!/bin/bash
# ============================================
# GCN 风控项目 — 一键运行 Pipeline
# ============================================

set -e

echo "======================================"
echo "  GCN 风控模型 — 一键运行 Pipeline"
echo "======================================"
echo ""

# 激活虚拟环境
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# Step 1: 生成模拟数据
echo "[1/3] 生成风控模拟数据..."
python -m src.data.generate_fraud_data
echo ""

# Step 2: 训练异构图 GNN
echo "[2/3] 训练异构图 GNN 模型..."
python -c "
import sys, torch, numpy as np
from src.data.generate_fraud_data import generate_fraud_dataset
from src.data.graph_builder import build_heterogeneous_graph, train_val_test_split_by_nodes
from src.models.hetero_gnn import HeteroGNN
from src.train import train_model
from src.utils import set_seed

set_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 数据准备
print('  生成数据...')
dataset = generate_fraud_dataset(seed=42)
n_users = dataset['user_features'].shape[0]
train_mask, val_mask, test_mask = train_val_test_split_by_nodes(n_users, seed=42)
data = build_heterogeneous_graph(
    node_features_dict={
        'user': dataset['user_features'],
        'device': dataset['device_features'],
        'ip': dataset['ip_features'],
        'phone': dataset['phone_features'],
    },
    edge_dict={
        ('user', 'uses', 'device'): dataset['user_device_edges'],
        ('user', 'from', 'ip'): dataset['user_ip_edges'],
        ('user', 'has', 'phone'): dataset['user_phone_edges'],
    },
    labels_dict={'user': dataset['labels']},
    train_mask_dict={'user': train_mask},
    val_mask_dict={'user': val_mask},
    test_mask_dict={'user': test_mask},
)

# 模型训练
print('  训练模型...')
model = HeteroGNN(
    node_types=['user', 'device', 'ip', 'phone'],
    edge_types=[('user', 'uses', 'device'), ('user', 'from', 'ip'), ('user', 'has', 'phone')],
    target_node_type='user',
    in_channels_dict={'user': 8, 'device': 6, 'ip': 5, 'phone': 4},
    hidden_channels=64, out_channels=2, num_layers=2,
)
result = train_model(model, data, device, epochs=100, use_focal_loss=True, is_hetero=True)
print(f'  最佳 Val F1: {result[\"best_score\"]:.4f}')

# 保存模型
import os, pickle
os.makedirs('saved_models', exist_ok=True)
torch.save(model.state_dict(), 'saved_models/gcn_fraud.pt')
print('  模型已保存至 saved_models/gcn_fraud.pt')
"
echo ""

# Step 3: 启动推理服务
echo "[3/3] 模型推理服务已就绪"
echo "  运行以下命令启动服务:"
echo "    python deploy/model_server.py"
echo "  或"
echo "    gunicorn -w 4 -b 0.0.0.0:8000 deploy.model_server:app"
echo ""

echo "======================================"
echo "  Pipeline 完成！"
echo "======================================"
