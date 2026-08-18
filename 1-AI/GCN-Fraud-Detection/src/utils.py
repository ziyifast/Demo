"""
工具函数模块

提供常用的辅助功能：
  - 时间统计
  - 随机种子设置
  - 日志配置
  - 图结构分析工具
"""

import time
import random
import numpy as np
import torch
import logging
from typing import Optional, List, Tuple
from functools import wraps
from collections import defaultdict


def set_seed(seed: int = 42) -> None:
    """
    设置全局随机种子，确保实验可复现。

    Parameters
    ----------
    seed : int
        随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_logger(name: str = "GCN-Fraud", level: int = logging.INFO) -> logging.Logger:
    """
    创建一个格式化的 logger。

    Parameters
    ----------
    name : str
        Logger 名称
    level : int
        日志级别

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


class Timer:
    """上下文管理器，用于计时。"""

    def __init__(self, name: str = ""):
        self.name = name
        self.start = None
        self.elapsed = None

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self.start
        if self.name:
            print(f"[{self.name}] 耗时: {self.elapsed:.2f}s")


def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    """
    统计模型参数数量。

    Returns
    -------
    total_params : int
        总参数数
    trainable_params : int
        可训练参数数
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def analyze_graph_structure(
    edge_index: torch.Tensor,
    num_nodes: Optional[int] = None,
) -> dict:
    """
    分析图结构的基本统计信息。

    Parameters
    ----------
    edge_index : Tensor, shape (2, E)
        边索引
    num_nodes : int, optional
        节点数。如果不提供，从 edge_index 推断

    Returns
    -------
    dict
        图结构统计信息
    """
    if num_nodes is None:
        num_nodes = edge_index.max().item() + 1

    source, target = edge_index[0].numpy(), edge_index[1].numpy()

    # 度数分布
    degrees = np.bincount(source, minlength=num_nodes)

    stats = {
        "num_nodes": num_nodes,
        "num_edges": edge_index.shape[1],
        "avg_degree": degrees.mean(),
        "max_degree": degrees.max(),
        "min_degree": degrees.min(),
        "median_degree": np.median(degrees),
        "isolated_nodes": (degrees == 0).sum(),
    }

    # 连通分量（需要 networkx）
    try:
        import networkx as nx
        G = nx.Graph()
        G.add_edges_from(edge_index.T.numpy())
        G.add_nodes_from(range(num_nodes))

        conn_components = list(nx.connected_components(G))
        stats["num_connected_components"] = len(conn_components)
        stats["largest_component_size"] = max(len(c) for c in conn_components)
    except ImportError:
        pass

    return stats


def find_suspicious_subgraphs(
    edge_index: torch.Tensor,
    num_nodes: int,
    threshold_degree: int = 20,
) -> List[int]:
    """
    发现图中的可疑密集子图节点（基于高度数）。

    在风控场景中，正常用户的连接数通常是有限的。
    如果一个节点连接到异常多的其他节点（如一个设备被数百个用户使用），
    这个节点及其邻居就非常可疑。

    Parameters
    ----------
    edge_index : Tensor, shape (2, E)
        边索引
    num_nodes : int
        节点总数
    threshold_degree : int
        度数的异常判定阈值

    Returns
    -------
    List[int]
        高度数节点列表
    """
    source = edge_index[0].numpy()
    degrees = np.bincount(source, minlength=num_nodes)
    suspicious = np.where(degrees >= threshold_degree)[0]
    return suspicious.tolist()
