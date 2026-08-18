"""
异构图神经网络 (Heterogeneous Graph Neural Network)

在真实的风控场景中，图通常包含多种类型的节点和边。例如：
  - 节点类型: 用户(user)、设备(device)、IP地址(ip)、手机号(phone)、邮箱(email)
  - 边类型: (user, uses, device), (user, from, ip), (user, has, phone)

异构图 GNN 为每种边类型学习独立的参数，能够更好地捕获不同类型关系之间的差异。

本模块实现了两种主流方法：
  1. HeteroSAGE: 使用 HeteroConv + SAGEConv 的异构图模型
  2. 支持灵活的节点类型和边类型配置
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv, GCNConv, GATConv
from typing import Dict, List, Optional


class HeteroGNN(nn.Module):
    """
    异构图神经网络，用于风控场景中的多实体团伙检测。

    通过 HeteroConv 为每种边类型学习独立的卷积算子，
    最终对目标节点类型（如 user）进行分类。

    Parameters
    ----------
    node_types : List[str]
        所有节点类型列表，如 ['user', 'device', 'ip', 'phone']
    edge_types : List[tuple]
        所有边类型列表，如 [('user', 'uses', 'device'), ('user', 'from', 'ip')]
    target_node_type : str
        目标节点类型（需要分类的节点），如 'user'
    in_channels_dict : Dict[str, int]
        每种节点类型的输入特征维度
    hidden_channels : int, default=64
        隐藏层维度
    out_channels : int, default=2
        输出类别数（风控场景通常为 2：正常/欺诈）
    num_layers : int, default=2
        层数
    conv_type : str, default='sage'
        卷积类型，支持 'sage', 'gcn', 'gat'
    dropout : float, default=0.3
        Dropout 概率
    add_reverse_edges : bool, default=True
        是否自动为每条边添加反向边。在风控场景中，如果边定义为
        ('user', 'uses', 'device')，user 是 source，device 是 destination。
        为了能让 user 也聚合到 device 的信息，需要添加反向边
        ('device', 'rev_uses', 'user')。推荐保持默认 True。

    Examples
    --------
    >>> from src.models.hetero_gnn import HeteroGNN
    >>> node_types = ['user', 'device', 'ip']
    >>> edge_types = [('user', 'uses', 'device'), ('user', 'from', 'ip')]
    >>> in_channels = {'user': 64, 'device': 32, 'ip': 16}
    >>> model = HeteroGNN(node_types, edge_types, 'user', in_channels,
    ...                   hidden_channels=64, out_channels=2, num_layers=2)
    >>> # x_dict = {'user': user_x, 'device': device_x, 'ip': ip_x}
    >>> # edge_index_dict = {('user','uses','device'): edge_idx, ...}
    >>> out = model(x_dict, edge_index_dict)  # 返回 user 节点的分类结果
    """

    def __init__(
        self,
        node_types: List[str],
        edge_types: List[tuple],
        target_node_type: str,
        in_channels_dict: Dict[str, int],
        hidden_channels: int = 64,
        out_channels: int = 2,
        num_layers: int = 2,
        conv_type: str = "sage",
        dropout: float = 0.3,
        add_reverse_edges: bool = True,
    ):
        super(HeteroGNN, self).__init__()

        if num_layers < 1:
            raise ValueError("num_layers 至少为 1")

        if target_node_type not in node_types:
            raise ValueError(
                f"target_node_type '{target_node_type}' 必须在 node_types 中"
            )

        valid_convs = ["sage", "gcn", "gat"]
        if conv_type not in valid_convs:
            raise ValueError(f"conv_type 必须是 {valid_convs} 之一")

        self.node_types = node_types
        self.edge_types = edge_types
        self.target_node_type = target_node_type
        self.num_layers = num_layers
        self.dropout = dropout
        self.add_reverse_edges = add_reverse_edges

        # 构建完整的边类型列表（包括反向边）
        self.all_edge_types = list(edge_types)
        if add_reverse_edges:
            for src, rel, dst in edge_types:
                rev_etype = (dst, f"rev_{rel}", src)
                if rev_etype not in self.all_edge_types:
                    self.all_edge_types.append(rev_etype)

        # 选择卷积类型
        ConvClass = {"sage": SAGEConv, "gcn": GCNConv, "gat": GATConv}[conv_type]

        # 特征投影层：将不同节点类型的特征统一映射到 hidden_channels
        self.proj = nn.ModuleDict()
        for ntype in node_types:
            self.proj[ntype] = nn.Linear(in_channels_dict[ntype], hidden_channels)

        # 异构图卷积层
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv_dict = {}
            for etype in self.all_edge_types:
                # etype: (src_type, rel_type, dst_type)
                conv_dict[etype] = ConvClass(
                    (-1, -1),  # 自动推断输入维度
                    hidden_channels,
                )
            self.convs.append(HeteroConv(conv_dict, aggr="sum"))

        # Batch Normalization
        self.bns = nn.ModuleList()
        for _ in range(num_layers):
            bn_dict = nn.ModuleDict()
            for ntype in node_types:
                bn_dict[ntype] = nn.BatchNorm1d(hidden_channels)
            self.bns.append(bn_dict)

        # 分类层（仅对目标节点类型）
        self.classifier = nn.Linear(hidden_channels, out_channels)

        self._reset_parameters()

    def _reset_parameters(self):
        """初始化模型参数"""
        for proj in self.proj.values():
            proj.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for bn_dict in self.bns:
            for bn in bn_dict.values():
                bn.reset_parameters()
        self.classifier.reset_parameters()

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[tuple, torch.Tensor],
    ) -> torch.Tensor:
        """
        前向传播。

        Parameters
        ----------
        x_dict : Dict[str, Tensor]
            每种节点类型的特征，key 为节点类型，value 为 (N_t, in_channels_t) 的特征矩阵
        edge_index_dict : Dict[tuple, Tensor]
            每种边类型的边索引，key 为 (src, rel, dst) 元组，value 为 (2, E) 的边索引

        Returns
        -------
        torch.Tensor, shape (N_target, out_channels)
            目标节点类型的分类 log 概率
        """
        # Step 0: 自动添加反向边（如果未提供）
        if self.add_reverse_edges:
            full_ei_dict = dict(edge_index_dict)
            for etype, ei in edge_index_dict.items():
                src, rel, dst = etype
                rev_etype = (dst, f"rev_{rel}", src)
                if rev_etype not in full_ei_dict:
                    # 反向边：交换 source 和 target
                    rev_ei = torch.stack([ei[1], ei[0]], dim=0)
                    full_ei_dict[rev_etype] = rev_ei
            edge_index_dict = full_ei_dict

        # Step 1: 特征投影到统一维度
        x_dict = {ntype: self.proj[ntype](x) for ntype, x in x_dict.items()}

        # Step 2: 多轮异构图卷积
        for i in range(self.num_layers):
            # HeteroConv 前向传播（只更新作为 destination 的节点）
            x_dict_new = self.convs[i](x_dict, edge_index_dict)

            # 更新节点表示：destination 类型用新值，其他类型保留旧值
            prev_x_dict = x_dict  # 上一轮的完整表示
            x_dict = {}
            for ntype in self.node_types:
                x_new = x_dict_new.get(ntype)
                if x_new is not None:
                    x = x_new
                else:
                    x = prev_x_dict[ntype]  # 保留原表示
                x = self.bns[i][ntype](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                x_dict[ntype] = x

        # Step 3: 分类目标节点类型
        target_x = x_dict[self.target_node_type]
        return self.classifier(target_x)

    def get_embeddings(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[tuple, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        获取所有节点类型的嵌入表示。

        Returns
        -------
        Dict[str, Tensor]
            每种节点类型的嵌入向量
        """
        with torch.no_grad():
            # 添加反向边
            if self.add_reverse_edges:
                full_ei_dict = dict(edge_index_dict)
                for etype, ei in edge_index_dict.items():
                    src, rel, dst = etype
                    rev_etype = (dst, f"rev_{rel}", src)
                    if rev_etype not in full_ei_dict:
                        rev_ei = torch.stack([ei[1], ei[0]], dim=0)
                        full_ei_dict[rev_etype] = rev_ei
                edge_index_dict = full_ei_dict

            x_dict = {ntype: self.proj[ntype](x) for ntype, x in x_dict.items()}
            for i in range(self.num_layers):
                x_dict_new = self.convs[i](x_dict, edge_index_dict)
                prev_x_dict = x_dict
                x_dict = {}
                for ntype in self.node_types:
                    x_new = x_dict_new.get(ntype)
                    if x_new is not None:
                        x = x_new
                    else:
                        x = prev_x_dict[ntype]
                    x = self.bns[i][ntype](x)
                    if i < self.num_layers - 1:
                        x = F.relu(x)
                    x_dict[ntype] = x
            return x_dict
