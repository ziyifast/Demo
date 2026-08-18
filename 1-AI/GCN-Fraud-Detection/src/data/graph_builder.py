"""
图数据构建工具

在风控场景中，我们需要从原始数据（如 CSV、数据库查询结果）构建图结构。
本模块提供了构建同构图和异构图的工具函数。

常见的风控图构建模式：
  同构图：所有节点为同一类型（如账号），边为某种关系（如转账）
  异构图：包含多种节点类型（用户、设备、IP...）和多种边类型（登录、支付...）
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional
from sklearn.preprocessing import LabelEncoder, StandardScaler

try:
    from torch_geometric.data import Data, HeteroData
    _PYG_AVAILABLE = True
except ImportError:
    _PYG_AVAILABLE = False
    Data = object  # 占位类型
    HeteroData = object


def build_homogeneous_graph(
    node_features: np.ndarray,
    edge_list: np.ndarray,
    labels: Optional[np.ndarray] = None,
    train_mask: Optional[np.ndarray] = None,
    val_mask: Optional[np.ndarray] = None,
    test_mask: Optional[np.ndarray] = None,
    normalize_features: bool = True,
) -> "Data":
    """
    从 NumPy 数组构建同构图（PyG Data 对象）.
    ...
    """
    if not _PYG_AVAILABLE:
        raise ImportError("需要安装 PyTorch Geometric: pip install torch_geometric")
    if normalize_features:
        scaler = StandardScaler()
        node_features = scaler.fit_transform(node_features)

    x = torch.tensor(node_features, dtype=torch.float)
    edge_index = torch.tensor(edge_list.T, dtype=torch.long)

    data = Data(x=x, edge_index=edge_index)

    if labels is not None:
        data.y = torch.tensor(labels, dtype=torch.long)

    if train_mask is not None:
        data.train_mask = torch.tensor(train_mask, dtype=torch.bool)
    if val_mask is not None:
        data.val_mask = torch.tensor(val_mask, dtype=torch.bool)
    if test_mask is not None:
        data.test_mask = torch.tensor(test_mask, dtype=torch.bool)

    return data


def build_heterogeneous_graph(
    node_features_dict: Dict[str, np.ndarray],
    edge_dict: Dict[Tuple[str, str, str], np.ndarray],
    labels_dict: Optional[Dict[str, np.ndarray]] = None,
    train_mask_dict: Optional[Dict[str, np.ndarray]] = None,
    val_mask_dict: Optional[Dict[str, np.ndarray]] = None,
    test_mask_dict: Optional[Dict[str, np.ndarray]] = None,
    normalize_features: bool = True,
) -> HeteroData:
    """
    构建异构图（PyG HeteroData 对象）。

    Parameters
    ----------
    node_features_dict : Dict[str, np.ndarray]
        每种节点类型的特征矩阵，key 为节点类型名，value 为特征数组
    edge_dict : Dict[Tuple[str, str, str], np.ndarray]
        边，key 为 (src_type, rel_type, dst_type) 元组，
        value 为 (E, 2) 的边数组
    labels_dict : Dict[str, np.ndarray], optional
        每种节点类型的标签
    train_mask_dict / val_mask_dict / test_mask_dict : Dict, optional
        训练/验证/测试集掩码
    normalize_features : bool
        是否对特征做标准化

    Returns
    -------
    HeteroData
        PyG 异构图数据对象

    Examples
    --------
    >>> # 构建一个简单的用户-设备-IP 异构图
    >>> node_features = {
    ...     'user': np.random.randn(100, 32),
    ...     'device': np.random.randn(50, 16),
    ...     'ip': np.random.randn(30, 8),
    ... }
    >>> edges = {
    ...     ('user', 'uses', 'device'): np.random.randint(0, [100, 50], (200, 2)),
    ...     ('user', 'from', 'ip'): np.random.randint(0, [100, 30], (150, 2)),
    ... }
    >>> labels = {'user': np.random.randint(0, 2, 100)}
    >>> data = build_heterogeneous_graph(node_features, edges, labels)
    """
    data = HeteroData()

    # 添加节点特征
    for ntype, features in node_features_dict.items():
        if normalize_features:
            scaler = StandardScaler()
            features = scaler.fit_transform(features)
        data[ntype].x = torch.tensor(features, dtype=torch.float)

    # 添加边
    for etype, edges in edge_dict.items():
        src_type, rel_type, dst_type = etype
        if edges.shape[1] != 2:
            raise ValueError(
                f"边数组必须为 (E, 2) 形状，但 {etype} 的形状为 {edges.shape}"
            )
        data[etype].edge_index = torch.tensor(edges.T, dtype=torch.long)

    # 添加标签
    if labels_dict is not None:
        for ntype, labels in labels_dict.items():
            data[ntype].y = torch.tensor(labels, dtype=torch.long)

    # 添加掩码
    if train_mask_dict is not None:
        for ntype, mask in train_mask_dict.items():
            data[ntype].train_mask = torch.tensor(mask, dtype=torch.bool)
    if val_mask_dict is not None:
        for ntype, mask in val_mask_dict.items():
            data[ntype].val_mask = torch.tensor(mask, dtype=torch.bool)
    if test_mask_dict is not None:
        for ntype, mask in test_mask_dict.items():
            data[ntype].test_mask = torch.tensor(mask, dtype=torch.bool)

    return data


def encode_categorical_features(
    df: pd.DataFrame,
    columns: List[str],
) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    """
    对 DataFrame 中的类别特征进行编码。

    Parameters
    ----------
    df : pd.DataFrame
        输入数据
    columns : List[str]
        需要编码的列名列表

    Returns
    -------
    df_encoded : pd.DataFrame
        编码后的数据
    encoders : Dict[str, LabelEncoder]
        每个列的编码器（可用于逆变换）

    Examples
    --------
    >>> df = pd.DataFrame({'os': ['ios', 'android', 'ios', 'android']})
    >>> df_encoded, encoders = encode_categorical_features(df, ['os'])
    """
    df_encoded = df.copy()
    encoders = {}
    for col in columns:
        encoders[col] = LabelEncoder()
        df_encoded[col] = encoders[col].fit_transform(df_encoded[col].astype(str))
    return df_encoded, encoders


def train_val_test_split_by_nodes(
    n_nodes: int,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    按节点数量随机划分训练/验证/测试集。

    Parameters
    ----------
    n_nodes : int
        节点总数
    train_ratio : float
        训练集比例
    val_ratio : float
        验证集比例
    seed : int
        随机种子

    Returns
    -------
    train_mask, val_mask, test_mask : np.ndarray
        布尔掩码数组
    """
    np.random.seed(seed)
    indices = np.random.permutation(n_nodes)
    train_end = int(n_nodes * train_ratio)
    val_end = train_end + int(n_nodes * val_ratio)

    train_mask = np.zeros(n_nodes, dtype=bool)
    val_mask = np.zeros(n_nodes, dtype=bool)
    test_mask = np.zeros(n_nodes, dtype=bool)

    train_mask[indices[:train_end]] = True
    val_mask[indices[train_end:val_end]] = True
    test_mask[indices[val_end:]] = True

    return train_mask, val_mask, test_mask
