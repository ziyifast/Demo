"""
GraphSAGE (Graph SAmple and aggreGatE)

基于 Hamilton et al. (2017) "Inductive Representation Learning on Large Graphs"
论文实现。

GraphSAGE 与 GCN 的关键区别：
  1. GCN 需要全图训练（transductive），GraphSAGE 支持 mini-batch 训练（inductive）
  2. GraphSAGE 支持多种聚合函数：mean, max, LSTM, sum
  3. GraphSAGE 对未见过的节点也能生成嵌入

这使其特别适合风控场景中：
  - 图上不断有新用户/设备/订单加入（inductive 能力）
  - 超大规模图需要 mini-batch 训练
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class GraphSAGE(nn.Module):
    """
    多层 GraphSAGE 模型，支持归纳式节点分类。

    Parameters
    ----------
    in_channels : int
        输入特征维度
    hidden_channels : int
        隐藏层特征维度
    out_channels : int
        输出类别数
    num_layers : int, default=2
        层数
    aggr : str, default='mean'
        聚合函数类型，支持 'mean', 'max', 'sum', 'lstm'
    dropout : float, default=0.5
        Dropout 概率

    Examples
    --------
    >>> from src.models.sage import GraphSAGE
    >>> model = GraphSAGE(in_channels=64, hidden_channels=256, out_channels=2, aggr='mean')
    >>> out = model(x, edge_index)
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        aggr: str = "mean",
        dropout: float = 0.5,
    ):
        super(GraphSAGE, self).__init__()

        if num_layers < 2:
            raise ValueError("num_layers 至少为 2")

        valid_aggrs = ["mean", "max", "sum", "lstm"]
        if aggr not in valid_aggrs:
            raise ValueError(f"aggr 必须是 {valid_aggrs} 之一，但得到 '{aggr}'")

        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()

        # 第一层：输入 → 隐藏
        self.convs.append(SAGEConv(in_channels, hidden_channels, aggr=aggr))

        # 中间层：隐藏 → 隐藏
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr=aggr))

        # 最后一层：隐藏 → 输出
        self.convs.append(SAGEConv(hidden_channels, out_channels, aggr=aggr))

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
            边索引

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

        x = self.convs[-1](x, edge_index)
        return F.log_softmax(x, dim=1)

    def get_embeddings(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        获取节点的嵌入表示（倒数第二层输出）。
        """
        with torch.no_grad():
            for i, conv in enumerate(self.convs[:-1]):
                x = conv(x, edge_index)
                x = self.bns[i](x)
                x = F.relu(x)
            return x
