# 二、图卷积神经网络（GCN）

## 2.1 什么是 GCN

图卷积神经网络（Graph Convolutional Network，**GCN**）是目前最主流、应用最广泛的图神经网络模型之一，由 Thomas N. Kipf 和 Max Welling 于 2017 年在论文《Semi-Supervised Classification with Graph Convolutional Networks》中提出。

GCN 本质上就是一个**特征提取器**，只不过它的对象是图数据。CNN 处理的是图像这样的网格结构数据，而 GCN 处理的是图这样的非规则结构数据。GCN 精妙地设计了一种从图数据中提取特征的方法，可以用于：

- **节点分类（node classification）**：判断每个节点属于哪个类别
- **图分类（graph classification）**：判断整张图属于哪个类别
- **边预测（link prediction）**：判断两个节点之间是否应该存在边

## 2.2 核心原理

### 一句话概括

**每一层 GCN = 邻居信息聚合 + 特征变换**

具体来说，GCN 的每一层做两件事：

1. **聚合（Aggregate）**：将每个节点及其所有邻居节点的特征汇总起来（取平均或求和）
2. **变换（Transform）**：对汇总后的特征做一次线性变换（乘以权重矩阵）和非线性激活

通过堆叠多层 GCN，每个节点最终能够聚合到多跳邻居的信息。比如：
- **1 层 GCN**：每个节点感知到其直接邻居（一度邻居）的信息
- **2 层 GCN**：每个节点感知到其二度邻居（朋友的朋友）的信息
- **3 层 GCN**：每个节点感知到其三度邻居的信息

### 数学公式

GCN 的层间传播规则可以表示为：

```
H^(l+1) = σ(D̃^(-1/2) Ã D̃^(-1/2) H^(l) W^(l))
```

其中：
- **H^(l)**：第 l 层的节点表示矩阵（第 0 层即输入特征 X）
- **Ã = A + I**：添加了自环的邻接矩阵（确保节点也能聚合自己的信息）
- **D̃**：Ã 的度矩阵（D̃_ii = Σ_j Ã_ij）
- **W^(l)**：第 l 层的可训练权重矩阵
- **σ**：非线性激活函数（通常为 ReLU）
- **D̃^(-1/2) Ã D̃^(-1/2)**：对称归一化的邻接矩阵

### 直观理解

可以把 GCN 每一层的计算理解为一个**三步过程**：

```
Step 1 (聚集):  对每个节点，将其邻居的特征取平均
Step 2 (变换):  将平均值通过一个线性层（乘以 W）
Step 3 (激活):  通过非线性函数（ReLU）增加表达能力
```

## 2.3 GCN 的关键性质

### 1. 局部参数共享

GCN 的卷积算子是**处处共享的**——同一个权重矩阵 W 适用于图中所有节点。这使得模型参数不会随图的大小而增长。

### 2. 感受野正比于层数

- 第 1 层：每个节点包含直接邻居的信息
- 第 2 层：包含邻居的邻居的信息
- 第 3 层：感受野进一步扩大

层数越多，感受野越广，参与运算的信息也越充分。但层数过多会导致**过度平滑（over-smoothing）**——所有节点的表示趋于相同。在风控场景中，通常使用 **2-3 层** GCN。

### 3. 半监督学习

GCN 的一个重要特性是**只需要少量有标签的样本**就能训练出精度高的模型。在金融风控领域，我们往往只知道少量样本的真实标签（如发生过违约的用户），GCN 的这种半监督特性使其非常适合此类场景。

### 4. 端到端学习

GCN 能够同时学习图的结构信息和节点的特征信息，不需要手工设计特征或定义规则。

## 2.4 代码实现解析

以下是 PyTorch Geometric 中 GCN 层的核心实现逻辑（简化版）：

```python
class GCNConv(nn.Module):
    def forward(self, x, edge_index):
        # x: 节点特征矩阵 (N, in_channels)
        # edge_index: 边索引 (2, E)

        # 1. 计算归一化的邻接矩阵
        #    等价于 D̃^(-1/2) * Ã * D̃^(-1/2)
        edge_weight = self.compute_norm(edge_index, num_nodes)

        # 2. 特征线性变换
        x = self.lin(x)  # x * W

        # 3. 邻居聚合（消息传递）
        out = self.propagate(edge_index, x=x, norm=edge_weight)

        # 4. 偏置项
        if self.bias is not None:
            out += self.bias

        return out
```

### 完整的 2 层 GCN 实现

```python
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        # 第一层：聚合 + 变换 + 激活 + Dropout
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        # 第二层：聚合 + 变换 → 输出
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
```

## 2.5 GNN 与 GCN 的关系

需要明确的是：**GNN 是图神经网络的总称，而 GCN 是 GNN 的一种具体实现**。

可以这样理解：

- **GNN** 是"交通工具"这个大类
- **GCN** 是其中的"汽车"（最常用、最经典）

| | GNN（概念） | GCN（实现） |
|---|---|---|
| 提出时间 | 2005 年概念，2009 年理论基础 | 2017 年 |
| 范围 | 所有处理图数据的神经网络 | GNN 的一种谱域实现 |
| 关系 | 父类/总称 | 子类/具体实现 |

除了 GCN 之外，还有：
- **GraphSAGE**：支持 mini-batch 训练的归纳式 GNN
- **GAT**：使用注意力机制学习邻居权重的 GNN
- **GIN**：理论上表达能力最强的 GNN

## 2.6 GCN 的局限性

了解 GCN 的局限有助于在风控场景中选择合适的模型：

1. **全图训练**：GCN 需要在全图上进行训练，对于超大规模图（亿级节点）效率较低（可使用 GraphSAGE 替代）
2. **过度平滑**：层数过多时所有节点表示趋于相同（通常 2-3 层效果最好）
3. **同质性假设**：GCN 假设相连的节点倾向属于同一类别，不适用于异配图
4. **不支持异构关系**：标准 GCN 处理的是同构图（可使用异构图 GNN 解决）

> **下一步**：了解 GCN 在风控中的实际应用场景，请阅读 [03-黑产风控场景与图建模方法论](./03-黑产风控场景与图建模方法论.md)
