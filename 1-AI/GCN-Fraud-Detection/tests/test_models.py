"""
模型单元测试

测试各个 GNN 模型的基本功能：
  1. 模型初始化和参数数量
  2. 前向传播输出形状
  3. 嵌入提取
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import pytest
from torch_geometric.data import Data, HeteroData

from src.models.gcn import GCN
from src.models.gat import GAT
from src.models.sage import GraphSAGE
from src.models.hetero_gnn import HeteroGNN
from src.data.graph_builder import build_homogeneous_graph, build_heterogeneous_graph
from src.train import FocalLoss, compute_class_weights, EarlyStopping
from src.utils import set_seed


# ========================================
# Fixtures
# ========================================

@pytest.fixture(autouse=True)
def seed():
    set_seed(42)


@pytest.fixture
def homogeneous_data():
    """创建一个小型同构图用于测试"""
    n_nodes, n_features, n_classes = 50, 16, 3
    x = torch.randn(n_nodes, n_features)
    edge_index = torch.randint(0, n_nodes, (2, 100))
    y = torch.randint(0, n_classes, (n_nodes,))
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    train_mask[:30] = True
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask[30:] = True
    return Data(x=x, edge_index=edge_index, y=y,
                train_mask=train_mask, test_mask=test_mask)


@pytest.fixture
def heterogeneous_data():
    """创建一个小型异构图用于测试"""
    data = HeteroData()
    data['user'].x = torch.randn(20, 8)
    data['device'].x = torch.randn(15, 6)
    data['ip'].x = torch.randn(10, 5)

    data['user', 'uses', 'device'].edge_index = torch.randint(0, [20, 15], (2, 30))
    data['user', 'from', 'ip'].edge_index = torch.randint(0, [20, 10], (2, 25))

    data['user'].y = torch.randint(0, 2, (20,))
    data['user'].train_mask = torch.zeros(20, dtype=torch.bool)
    data['user'].train_mask[:12] = True
    data['user'].test_mask = torch.zeros(20, dtype=torch.bool)
    data['user'].test_mask[12:] = True
    return data


# ========================================
# GCN 测试
# ========================================

def test_gcn_init():
    """测试 GCN 模型初始化"""
    model = GCN(in_channels=16, hidden_channels=32, out_channels=3, num_layers=2)
    assert isinstance(model, torch.nn.Module)
    assert len(model.convs) == 2


def test_gcn_forward(homogeneous_data):
    """测试 GCN 前向传播"""
    model = GCN(in_channels=16, hidden_channels=32, out_channels=3, num_layers=2)
    out = model(homogeneous_data.x, homogeneous_data.edge_index)
    assert out.shape == (50, 3)
    # log_softmax 输出应该是负值
    assert (out < 0).all() or (out <= 0).all()


def test_gcn_embeddings(homogeneous_data):
    """测试 GCN 嵌入提取"""
    model = GCN(in_channels=16, hidden_channels=32, out_channels=3, num_layers=2)
    emb = model.get_embeddings(homogeneous_data.x, homogeneous_data.edge_index)
    assert emb.shape == (50, 32)


def test_gcn_invalid_layers():
    """测试 GCN 层数验证"""
    with pytest.raises(ValueError):
        GCN(in_channels=16, hidden_channels=32, out_channels=3, num_layers=1)


# ========================================
# GAT 测试
# ========================================

def test_gat_forward(homogeneous_data):
    """测试 GAT 前向传播"""
    model = GAT(in_channels=16, hidden_channels=8, out_channels=3, heads=4)
    out = model(homogeneous_data.x, homogeneous_data.edge_index)
    assert out.shape == (50, 3)


def test_gat_attention_weights(homogeneous_data):
    """测试 GAT 注意力权重提取"""
    model = GAT(in_channels=16, hidden_channels=8, out_channels=3, heads=4)
    attn = model.get_attention_weights(homogeneous_data.x, homogeneous_data.edge_index)
    assert attn is not None
    assert attn.ndim == 1  # (E,) — 每条边一个权重


# ========================================
# GraphSAGE 测试
# ========================================

def test_sage_forward(homogeneous_data):
    """测试 GraphSAGE 前向传播"""
    for aggr in ['mean', 'max', 'sum']:
        model = GraphSAGE(in_channels=16, hidden_channels=32, out_channels=3, aggr=aggr)
        out = model(homogeneous_data.x, homogeneous_data.edge_index)
        assert out.shape == (50, 3), f"aggr={aggr} 失败"


def test_sage_invalid_aggr():
    """测试 GraphSAGE 无效聚合函数检查"""
    with pytest.raises(ValueError):
        GraphSAGE(in_channels=16, hidden_channels=32, out_channels=3, aggr='invalid')


# ========================================
# HeteroGNN 测试
# ========================================

def test_hetero_gnn_forward(heterogeneous_data):
    """测试 HeteroGNN 前向传播"""
    model = HeteroGNN(
        node_types=['user', 'device', 'ip'],
        edge_types=[('user', 'uses', 'device'), ('user', 'from', 'ip')],
        target_node_type='user',
        in_channels_dict={'user': 8, 'device': 6, 'ip': 5},
        hidden_channels=32,
        out_channels=2,
        num_layers=2,
    )
    out = model(heterogeneous_data.x_dict, heterogeneous_data.edge_index_dict)
    assert out.shape == (20, 2)  # 20 个 user 节点, 2 分类


def test_hetero_gnn_embeddings(heterogeneous_data):
    """测试 HeteroGNN 嵌入提取"""
    model = HeteroGNN(
        node_types=['user', 'device', 'ip'],
        edge_types=[('user', 'uses', 'device'), ('user', 'from', 'ip')],
        target_node_type='user',
        in_channels_dict={'user': 8, 'device': 6, 'ip': 5},
        hidden_channels=32,
        out_channels=2,
        num_layers=2,
    )
    embs = model.get_embeddings(
        heterogeneous_data.x_dict,
        heterogeneous_data.edge_index_dict,
    )
    assert 'user' in embs
    assert 'device' in embs
    assert embs['user'].shape == (20, 32)


# ========================================
# Loss 函数测试
# ========================================

def test_focal_loss():
    """测试 Focal Loss"""
    criterion = FocalLoss(alpha=0.75, gamma=2.0)
    inputs = torch.randn(10, 2)
    targets = torch.randint(0, 2, (10,))
    loss = criterion(inputs, targets)
    assert loss.item() > 0
    assert isinstance(loss.item(), float)


def test_class_weights():
    """测试类别权重计算"""
    labels = np.array([0, 0, 0, 0, 0, 1])  # 5负 1正
    weights = compute_class_weights(labels, num_classes=2)
    # 正类应该比负类权重大
    assert weights[1] > weights[0]


# ========================================
# EarlyStopping 测试
# ========================================

def test_early_stopping_max():
    """测试早停机制 (max mode)"""
    model = torch.nn.Linear(10, 2)
    stopper = EarlyStopping(patience=3, mode='max')

    scores = [0.5, 0.6, 0.6, 0.6, 0.6]  # 第 2 次后不再提升
    should_stop = False
    for epoch, score in enumerate(scores):
        should_stop = stopper.step(score, model, epoch)

    assert should_stop
    assert stopper.best_score == 0.6


# ========================================
# 图构建测试
# ========================================

def test_build_homogeneous_graph():
    """测试同构图构建"""
    features = np.random.randn(100, 16)
    edges = np.random.randint(0, 100, (200, 2))
    labels = np.random.randint(0, 2, 100)
    data = build_homogeneous_graph(features, edges, labels)
    assert data.x.shape == (100, 16)
    assert data.edge_index.shape[1] == 200
    assert data.y.shape == (100,)


def test_build_heterogeneous_graph():
    """测试异构图构建"""
    node_features = {
        'user': np.random.randn(50, 16),
        'device': np.random.randn(30, 8),
    }
    edges = {
        ('user', 'uses', 'device'): np.random.randint(0, [50, 30], (100, 2)),
    }
    labels = {'user': np.random.randint(0, 2, 50)}
    data = build_heterogeneous_graph(node_features, edges, labels)
    assert 'user' in data.node_types
    assert ('user', 'uses', 'device') in data.edge_types
