"""
数据集加载与预处理模块

提供：
  1. 标准图数据集的加载（Cora, CiteSeer, PubMed）
  2. 风控模拟数据集的加载
"""

import torch
import os
import pickle
from pathlib import Path
from typing import Tuple, Optional

# PyTorch Geometric 是可选依赖（仅在使用图数据集加载时必需）
try:
    from torch_geometric.datasets import Planetoid
    from torch_geometric.transforms import NormalizeFeatures
    from torch_geometric.data import Data, HeteroData
    _PYG_AVAILABLE = True
except ImportError:
    _PYG_AVAILABLE = False
    Planetoid = None
    NormalizeFeatures = None
    Data = None
    HeteroData = None


def load_cora(
    data_root: str = "data",
    normalize: bool = True,
) -> Tuple["Data", "Planetoid"]:
    """
    加载 Cora 引文网络数据集。
    ...
    """
    if not _PYG_AVAILABLE:
        raise ImportError(
            "需要安装 PyTorch Geometric 来加载 Cora 数据集。\n"
            "请运行: pip install torch_geometric"
        )
    transform = NormalizeFeatures() if normalize else None
    dataset = Planetoid(
        root=os.path.join(data_root, "Planetoid"),
        name="Cora",
        transform=transform,
    )
    data = dataset[0]
    return data, dataset


def load_citeseer(data_root: str = "data") -> Tuple[Data, Planetoid]:
    """
    加载 CiteSeer 引文网络数据集。
    """
    dataset = Planetoid(
        root=os.path.join(data_root, "Planetoid"),
        name="CiteSeer",
        transform=NormalizeFeatures(),
    )
    return dataset[0], dataset


def load_pubmed(data_root: str = "data") -> Tuple[Data, Planetoid]:
    """
    加载 PubMed 引文网络数据集。
    """
    dataset = Planetoid(
        root=os.path.join(data_root, "Planetoid"),
        name="PubMed",
        transform=NormalizeFeatures(),
    )
    return dataset[0], dataset


def load_fraud_dataset(
    data_path: str = "data/fraud_dataset.pkl",
) -> Optional[HeteroData]:
    """
    加载风控模拟数据集（异构图格式）。

    Parameters
    ----------
    data_path : str
        序列化数据文件路径

    Returns
    -------
    HeteroData or None
        异构图数据对象，如果文件不存在则返回 None
    """
    path = Path(data_path)
    if not path.exists():
        print(f"数据集文件不存在: {data_path}")
        print("请先运行 src/data/generate_fraud_data.py 生成数据")
        return None

    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def print_dataset_info(data: Data, dataset: Optional[Planetoid] = None) -> None:
    """
    打印数据集的基本信息。

    Parameters
    ----------
    data : Data
        PyG Data 对象
    dataset : Planetoid, optional
        原始数据集（用于获取 num_features, num_classes）
    """
    print("=" * 50)
    print("数据集信息")
    print("=" * 50)
    print(f"  节点数:      {data.num_nodes:,}")
    print(f"  边数:        {data.num_edges:,}")

    if dataset is not None:
        print(f"  特征维度:    {dataset.num_features}")
        print(f"  类别数:      {dataset.num_classes}")

    if hasattr(data, "train_mask") and data.train_mask is not None:
        print(f"  训练节点数:  {data.train_mask.sum().item():,}")
    if hasattr(data, "val_mask") and data.val_mask is not None:
        print(f"  验证节点数:  {data.val_mask.sum().item():,}")
    if hasattr(data, "test_mask") and data.test_mask is not None:
        print(f"  测试节点数:  {data.test_mask.sum().item():,}")
    print("=" * 50)


def get_device() -> torch.device:
    """
    获取可用的计算设备。

    Returns
    -------
    torch.device
        'cuda' 或 'cpu'（MPS 仅在 macOS 上的特定操作可用）
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("cpu")  # PyG 的一些操作不完全支持 MPS
    else:
        return torch.device("cpu")
