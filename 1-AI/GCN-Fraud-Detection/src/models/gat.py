"""
图注意力网络 (Graph Attention Network, GAT)

基于 Velickovic et al. (2018) "Graph Attention Networks" 论文实现。

与 GCN 的核心区别：
  - GCN 的邻居聚合权重是固定的（由度矩阵决定）
  - GAT 使用注意力机制动态学习邻居的重要性权重

这使得 GAT 能够：
  1. 对不同的邻居赋予不同的重要性
  2. 无需预先知道图结构即可计算注意力权重
  3. 通过多头注意力提高模型稳定性

在风控场景中，GAT 的注意力权重可以用来解释：
  "为什么某个用户被判定为高风险？因为他的哪些关联节点权重更高？"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GAT(nn.Module):
    """
    多层图注意力网络，用于节点分类任务。

    Parameters
    ----------
    in_channels : int
        输入特征维度
    hidden_channels : int
        隐藏层特征维度（每个注意力头的维度）
    out_channels : int
        输出类别数
    num_layers : int, default=2
        GAT 层数
    heads : int, default=8
        多头注意力的头数
    dropout : float, default=0.6
        Dropout 概率
    attention_dropout : float, default=0.2
        注意力权重的 Dropout 概率

    Examples
    --------
    >>> from src.models.gat import GAT
    >>> model = GAT(in_channels=1433, hidden_channels=8, out_channels=7, heads=8)
    >>> out = model(x, edge_index)
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        heads: int = 8,
        dropout: float = 0.6,
        attention_dropout: float = 0.2,
    ):
        super(GAT, self).__init__()

        if num_layers < 2:
            raise ValueError("num_layers 至少为 2")

        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()

        # 第一层：输入 → 隐藏（多头）
        self.convs.append(
            GATConv(
                in_channels,
                hidden_channels,
                heads=heads,
                dropout=attention_dropout,
                concat=True,  # 拼接多头输出
            )
        )

        # 中间层：隐藏*heads → 隐藏（多头）
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(
                    hidden_channels * heads,
                    hidden_channels,
                    heads=heads,
                    dropout=attention_dropout,
                    concat=True,
                )
            )

        # 最后一层：隐藏*heads → 输出（单头，取平均）
        self.convs.append(
            GATConv(
                hidden_channels * heads,
                out_channels,
                heads=1,
                dropout=attention_dropout,
                concat=False,
            )
        )

        self._reset_parameters()

    def _reset_parameters(self):
        """初始化模型参数"""
        for conv in self.convs:
            conv.reset_parameters()

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
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.convs[-1](x, edge_index)
        return F.log_softmax(x, dim=1)

    def get_attention_weights(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        获取第一层的注意力权重（用于模型解释性分析）。

        Returns
        -------
        torch.Tensor, shape (E, heads)
            每条边在每个注意力头下的权重
        """
        with torch.no_grad():
            # 第一层 GATConv 返回注意力权重
            x, attn_weights = self.convs[0](x, edge_index, return_attention_weights=True)
            return attn_weights
