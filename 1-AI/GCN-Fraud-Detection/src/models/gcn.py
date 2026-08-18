"""
图卷积神经网络 (Graph Convolutional Network, GCN)

基于 Kipf & Welling (2017) "Semi-Supervised Classification with Graph
Convolutional Networks" 论文实现。

核心思想：每一层 GCN 做两件事
  1. 聚合：将每个节点及其邻居节点的特征汇总
  2. 变换：对汇总后的特征做线性变换 + 非线性激活

计算公式：
  H^(l+1) = σ(D̃^(-1/2) * Ã * D̃^(-1/2) * H^(l) * W^(l))

其中：
  - Ã = A + I  (带自环的邻接矩阵)
  - D̃ 是 Ã 的度矩阵
  - W^(l) 是可训练的权重矩阵
  - σ 是激活函数（如 ReLU）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN(nn.Module):
    """
    多层图卷积神经网络，用于节点分类任务。

    Parameters
    ----------
    in_channels : int
        输入特征维度
    hidden_channels : int
        隐藏层特征维度
    out_channels : int
        输出类别数
    num_layers : int, default=2
        GCN 层数（包含输入层和输出层）
    dropout : float, default=0.5
        Dropout 概率

    Examples
    --------
    >>> from src.models.gcn import GCN
    >>> model = GCN(in_channels=1433, hidden_channels=16, out_channels=7, num_layers=2)
    >>> out = model(x, edge_index)  # x: (N, 1433), edge_index: (2, E)
    >>> # out: (N, 7) — 每个节点的类别 log 概率
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        dropout: float = 0.5,
    ):
        super(GCN, self).__init__()

        if num_layers < 2:
            raise ValueError("num_layers 至少为 2（输入层 + 输出层）")

        self.num_layers = num_layers
        self.dropout = dropout

        # 构建 GCN 层列表
        self.convs = nn.ModuleList()

        # 第一层：输入 → 隐藏
        self.convs.append(GCNConv(in_channels, hidden_channels))

        # 中间层：隐藏 → 隐藏
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))

        # 最后一层：隐藏 → 输出
        self.convs.append(GCNConv(hidden_channels, out_channels))

        # Batch Normalization
        self.bns = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        self._reset_parameters()

    def _reset_parameters(self):
        """初始化模型参数"""
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        Parameters
        ----------
        x : torch.Tensor, shape (N, in_channels)
            节点特征矩阵
        edge_index : torch.Tensor, shape (2, E)
            边索引（COO 格式）

        Returns
        -------
        torch.Tensor, shape (N, out_channels)
            节点分类 log 概率
        """
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # 最后一层（不使用 BN 和 ReLU，直接输出 log_softmax）
        x = self.convs[-1](x, edge_index)
        return F.log_softmax(x, dim=1)

    def get_embeddings(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        获取倒数第二层的节点嵌入表示（用于可视化和聚类）。

        Parameters
        ----------
        x : torch.Tensor, shape (N, in_channels)
            节点特征矩阵
        edge_index : torch.Tensor, shape (2, E)
            边索引

        Returns
        -------
        torch.Tensor, shape (N, hidden_channels)
            节点嵌入向量
        """
        with torch.no_grad():
            for i, conv in enumerate(self.convs[:-1]):
                x = conv(x, edge_index)
                x = self.bns[i](x)
                x = F.relu(x)
            return x
