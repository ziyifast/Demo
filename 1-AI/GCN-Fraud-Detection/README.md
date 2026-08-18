# 🕸️ GCN 图神经网络风控实战

> **面向新人的完整 GNN（图神经网络）学习项目 —— 从零理解图、GCN、异构图到构建欺诈团伙检测系统。**

---

## 📖 目录

- [项目概述](#项目概述)
- [核心概念速览](#核心概念速览)
- [快速开始](#快速开始)
- [分步教程](#分步教程)
  - [第一步：理解图与 GNN](#第一步理解图与-gnn)
  - [第二步：GCN 节点分类入门](#第二步gcn-节点分类入门)
  - [第三步：风控场景的图建模](#第三步风控场景的图建模)
  - [第四步：欺诈团伙检测实战](#第四步欺诈团伙检测实战)
  - [第五步：模型评估与可解释性](#第五步模型评估与可解释性)
- [项目架构](#项目架构)
- [进阶优化](#进阶优化)
- [常见问题](#常见问题)
- [学习路径建议](#学习路径建议)

---

## 项目概述

### 🤔 这个项目解决什么问题？

传统的风控方法（规则引擎、孤立森林、单点特征模型）只能分析**单个个体**的行为是否异常，而无法捕捉个体之间的**关联关系**。但黑灰产恰恰是**团伙化**的：

```
正常用户:                欺诈团伙:

  用户A                   用户1 ---- 共享设备 ---- 用户2
   |                      |                          |
   设备1                   |        共享 IP          |
   |                      |                          |
   IP_A                  用户3 ---- 共享手机号 ---- 用户4

单看每个用户行为 → 看起来正常      但他们之间的关联 → 暴露了团伙特征！
```

> 🎯 **一句话理解：GNN = 给风控系统装上了关系雷达。单个账号看似清白，但账号之间的关联网络会出卖整个团伙。**

### 🎯 你将学到什么？

| 能力 | 说明 | 对应教程 |
|------|------|---------|
| **理解 GNN** | 图、消息传递、GCN 公式的含义 | 第一步（阅读文档） |
| **手写 GCN** | 用 PyG 在 Cora 数据集上训练 GCN 分类器 | 第二步（Jupyter Notebook） |
| **风控建模** | 将业务数据抽象成异构图 | 第三步（Jupyter Notebook） |
| **团伙检测** | 端到端的欺诈团伙识别系统 | 第四、五步（Jupyter Notebook） |

### 📊 行业真实效果

| 场景 | 公司/方法 | 效果 |
|------|----------|------|
| 刷单识别 | MAML-GCN | 准确率 89.47%，F1 90.90% |
| 垃圾注册 | 阿里巴巴 GNN | 每日额外识别 10-15% 垃圾账号 |
| 反垃圾评论 | 闲鱼 GAS | 覆盖率提升 16% |
| 酒店风控 | 携程图计算 | 月止损 100 万+ |
| 金融欺诈 | GNN + 强化学习 | 召回率提升 19.7%，误报率降低 33% |

---

## 核心概念速览

### 1. 图（Graph）

**图由节点和边组成，用于表示谁和谁有关系。**

| 图的元素 | 风控中的含义 | 示例 |
|---------|------------|------|
| **节点 (Node)** | 业务实体 | 用户账号、设备指纹、IP 地址、手机号 |
| **边 (Edge)** | 实体间关系 | 登录关系、支付关系、共用关系 |
| **度 (Degree)** | 一个节点关联的节点数 | 一个设备被 200 个用户使用 = 度=200（很可疑！） |

> 💡 **在风控中，共用关系是最重要的欺诈信号——正常用户不会几百人共用同一个设备。**

### 2. GNN（图神经网络）

**GNN 的核心思想：消息传递——每个节点通过观察邻居来更新自己的认知。**

```
第 0 层：每个节点只知道自己的特征
   用户A: [注册3天, 活跃度0.9, 设备更换5次] ← 有点可疑但不确定

第 1 层：A 发现邻居（设备和 IP）也被大量其他用户共用
   用户A: 我的设备和 IP 都很可疑！

第 2 层：A 发现共用设备的其他用户也被标记为可疑
   用户A: 我和一群可疑用户混在一起 → 我也是欺诈！
```

### 3. GCN（图卷积网络）

**GCN 是 GNN 中最经典、应用最广的实现，2017 年由 Kipf & Welling 提出。**

GCN 每一层做两件事：**邻居聚合 + 特征变换**。

| 对比维度 | CNN（图像） | GCN（图） |
|---------|-----------|----------|
| 处理对象 | 像素网格（规则） | 节点和边（不规则） |
| 卷积窗口 | 固定的 3x3 矩形 | 由邻居关系动态决定 |
| 参数共享 | 空间位置共享 | 所有节点共享 |

> 💡 **GNN 是总称（交通工具），GCN 是具体实现（汽车）。还有 GraphSAGE、GAT 等变体。**

### 4. 异构图（Heterogeneous Graph）

**真实风控图中包含多种节点和多种边，这就是异构图。**

| 图类型 | 节点 | 边 | 风控适用性 |
|--------|------|-----|-----------|
| 同构图 | 只有一种类型 | 只有一种关系 | 简单场景 |
| 异构图 | user/device/ip/phone | uses/from/has 等多种 | **风控标准方案** |

本项目同时支持同构图 GCN 和异构图 GNN（`src/models/hetero_gnn.py`）。

---

## 快速开始

### 环境要求

- Python 3.9+
- pip 包管理器
- 8GB+ 内存（推荐）

### 1️⃣ 进入项目目录

```bash
cd 1-AI/GCN-Fraud-Detection
```

### 2️⃣ 安装依赖

**方式一：使用安装脚本（推荐）**

```bash
bash scripts/setup_env.sh
source .venv/bin/activate
```

**方式二：手动安装（如果脚本报错）**

```bash
# 1. 创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate

# 2. 安装 PyTorch（必须先装！PyG 依赖它）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 3. 安装 PyTorch Geometric
pip install torch-scatter torch-sparse torch-cluster torch-geometric

# 4. 安装其他依赖
pip install -r requirements.txt
```

### 3️⃣ 验证安装

```bash
python3 -c "
import torch
from src.train import train_model, FocalLoss
from src.utils import set_seed
print(f'PyTorch: {torch.__version__}')
print('All modules loaded successfully!')
"
```

### 4️⃣ 按顺序学习

```bash
# 第一步：阅读理论文档
open docs/01-GNN概述与基础.md

# 第二步：运行 Cora GCN 入门 Notebook
jupyter notebook notebooks/01-GCN-Cora节点分类入门.ipynb

# 第三步：理解风控图建模
jupyter notebook notebooks/02-风控异构图构建.ipynb

# 第四步：欺诈团伙检测实战
jupyter notebook notebooks/03-欺诈团伙检测实战.ipynb

# 第五步：模型评估与解释
jupyter notebook notebooks/04-模型解释性与效果评估.ipynb
```

---

## 分步教程

### 第一步：理解图与 GNN

**阅读文档（按顺序）：**

1. `docs/01-GNN概述与基础.md` —— 图、GNN 概念、为什么适合风控
2. `docs/02-GCN原理详解.md` —— GCN 数学原理、代码解析
3. `docs/03-黑产风控场景与图建模方法论.md` —— 行业案例、建模方法论

**你会理解：**

- 图的基本概念：节点、边、度、邻接矩阵、同构/异构图
- GNN 的消息传递机制：发送 → 接收 → 变换
- GCN 的核心公式 `H = σ(D̃⁻½ Ã D̃⁻½ H W)` 每部分的含义
- 为什么传统方法（只看单点特征）不如 GNN（能看关联网络）

> 🎯 **核心收获：图结构本身包含了大量信息——一个设备被多少用户使用、用户之间有多少共享资源，这些都是识别黑产团伙的关键信号。**

---

### 第二步：GCN 节点分类入门

**运行命令：**

```bash
jupyter notebook notebooks/01-GCN-Cora节点分类入门.ipynb
```

**这个 Notebook 做了什么：**

1. **加载 Cora 数据集** — 2708 篇论文、10556 条引用边、1433 维特征、7 个类别
2. **定义 2 层 GCN** — `GCNConv(1433→16) → ReLU → Dropout → GCNConv(16→7)`
3. **训练** — 只用 140 个有标签样本（仅 5%）！这就是半监督学习
4. **对比随机森林** — GCN 用图结构信息 + 特征，随机森林只用特征
5. **超参数实验** — 改层数、改隐藏维度，观察效果变化

**关键代码：**

```python
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)   # 邻居聚合 + 特征变换
        x = F.relu(x)                    # 非线性激活
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)   # 再来一次
        return F.log_softmax(x, dim=1)
```

> 🎯 **核心收获：GCN 的代码非常简洁。它只需要 5% 的节点有标签，就能学到有效的分类器——因为图结构本身提供了免费的信息通道。**

---

### 第三步：风控场景的图建模

**运行命令：**

```bash
# 先生成模拟数据
python3 -c "from src.data.generate_fraud_data import generate_and_save; generate_and_save()"

# 然后运行 Notebook
jupyter notebook notebooks/02-风控异构图构建.ipynb
```

**你会看到：**

1. **特征分布对比** — 正常用户 vs 欺诈用户在注册天数、活跃度、交易金额等特征上的差异
2. **图结构分析** — 欺诈团伙成员的度数（共用设备/IP 数）是正常用户的 5-10 倍
3. **团伙子图可视化** — 用 NetworkX 画出一个团伙的关联网络

> 🎯 **核心收获：欺诈团伙在图上的结构特征非常明显——他们通过共享设备和 IP 形成了密集的连接模式。GNN 正是通过捕获这种结构差异来识别欺诈。**

---

### 第四步：欺诈团伙检测实战

**运行命令：**

```bash
jupyter notebook notebooks/03-欺诈团伙检测实战.ipynb
```

**这个 Notebook 覆盖完整的端到端流程：**

1. **异构图构建** — 4 种节点（user/device/ip/phone）+ 3 种边（uses/from/has）
2. **定义异构图 GNN** — 用 HeteroConv + SAGEConv，每种边类型独立学习
3. **类别不平衡处理** — 对比 Focal Loss vs 加权 CrossEntropyLoss
4. **模型训练** — 200 epochs + 早停 + 学习率衰减
5. **团伙发现评估** — 评估模型能检出多少个欺诈团伙

**为什么 Focal Loss 在风控中很重要？**

```
正常样本占 95%，且特征明显 → 容易分 → Focal Loss 自动降低其权重
欺诈样本占 5%，且特征隐蔽 → 难分   → Focal Loss 保持其权重

结果：模型不再被海量正常样本淹没，更关注难以分类的欺诈样本。
```

> 🎯 **核心收获：风控场景的核心挑战不是模型结构，而是类别极不平衡。Focal Loss 和加权损失是必须掌握的技巧。Precision@K 比 Accuracy 更能反映业务价值。**

---

### 第五步：模型评估与可解释性

**运行命令：**

```bash
jupyter notebook notebooks/04-模型解释性与效果评估.ipynb
```

**关键内容：**

| 内容 | 说明 |
|------|------|
| 多阈值对比 | 不同阈值下的 Precision/Recall 变化曲线 |
| PR vs ROC | 为什么在风控中 PR-AUC 比 ROC-AUC 更重要 |
| UMAP 可视化 | 节点嵌入在二维空间的分布（团伙聚类效果） |
| 错误分析 | 漏网用户（FN）和误拦用户（FP）有什么特征差异 |

**风控中该用什么评估指标？**

| 指标 | 适合场景 | 不适合场景 |
|------|---------|-----------|
| Accuracy | 类别平衡的分类任务 | 欺诈占比 5% → 全预测正常也能 95% |
| ROC-AUC | 类别相对平衡 | 极不平衡时会高估模型效果 |
| **PR-AUC** | **风控等正类极少的场景** | 类别平衡时和 ROC-AUC 差不多 |
| **Precision@K** | **只干预 Top-K 的业务场景** | 需要对全部样本做决策 |

> 🎯 **核心收获：在风控中不要用 Accuracy！一个把所有用户都预测为正常的模型也有 95% 准确率。应该关注 PR-AUC、Precision@K 和团伙检出覆盖率。**

---

## 项目架构

```
GCN-Fraud-Detection/
├── README.md                          # 📖 本文档
├── requirements.txt                   # 📦 Python 依赖
│
├── docs/                              # 📚 理论知识文档（先看这个！）
│   ├── 01-GNN概述与基础.md
│   ├── 02-GCN原理详解.md
│   ├── 03-黑产风控场景与图建模方法论.md
│   ├── 04-学习路径与资源推荐.md
│   └── 05-模型部署与上线指南.md
│
├── notebooks/                         # 🔬 交互式 Jupyter 教程
│   ├── 01-GCN-Cora节点分类入门.ipynb
│   ├── 02-风控异构图构建.ipynb
│   ├── 03-欺诈团伙检测实战.ipynb
│   └── 04-模型解释性与效果评估.ipynb
│
├── src/                               # 💻 生产级源代码
│   ├── models/                        #    GNN 模型（GCN/GAT/SAGE/HeteroGNN）
│   ├── data/                          #    数据处理与图构建
│   ├── train.py                       #    训练模块（Focal Loss / 早停）
│   ├── evaluate.py                    #    评估指标与可视化
│   └── utils.py                       #    工具函数
│
├── deploy/                            # 🚀 模型部署
│   ├── model_server.py                #    Flask API 推理服务
│   ├── config.py                      #    部署配置
│   └── Dockerfile                     #    Docker 部署
│
├── scripts/                           # 🔧 工具脚本
│   ├── setup_env.sh                   #    一键环境安装
│   └── run_pipeline.sh                #    一键运行完整流水线
│
└── tests/                             # 🧪 测试
    └── test_models.py
```

### 模型选择指南

| 模型 | 文件 | 适用场景 | 优点 |
|------|------|---------|------|
| **GCN** | `src/models/gcn.py` | 同构图、入门学习 | 简单高效、半监督 |
| **GAT** | `src/models/gat.py` | 需要可解释性 | 自动学习边权重 |
| **GraphSAGE** | `src/models/sage.py` | 大规模动态图 | 支持 mini-batch |
| **HeteroGNN** | `src/models/hetero_gnn.py` | 多实体多关系 | **风控标准方案** |

---

## 进阶优化

当你跑通基础流程后，可以探索以下方向：

### 1. 模型层面

| 优化方向 | 方法 | 效果 |
|---------|------|------|
| **时序图** | 引入时间维度，构建时序异构图 | 捕获行为模式的时间演化 |
| **图结构学习** | 不只用预定义边，让模型学习隐藏关系 | 发现未知的关联模式 |
| **对比学习** | 自监督预训练节点嵌入 | 减少对标签的依赖 |
| **集成学习** | GCN + GAT + GraphSAGE 融合 | 多角度互补 |

### 2. 特征层面

| 优化方向 | 方法 |
|---------|------|
| 统计特征 | 节点的度中心性、PageRank、聚类系数 |
| 行为序列 | 用 LSTM/Transformer 编码用户行为序列 |
| 图嵌入 | Node2Vec / DeepWalk 预训练嵌入作为额外特征 |

### 3. 业务层面

| 优化方向 | 方法 | 效果 |
|---------|------|------|
| 多阈值策略 | 不同场景用不同阈值 | 平衡拦截率和误拦率 |
| 级联模型 | 规则初筛 → GNN 精判 → 人工审核 | 降低计算成本 |
| 主动学习 | 把不确定的样本送人工标注 | 持续扩充训练集 |

---

## 常见问题

### Q1: PyTorch Geometric 安装失败怎么办？

**解决方法：先装 PyTorch，再根据 PyTorch 版本装匹配的 PyG wheel。**

```bash
# 1. 确认 PyTorch 版本
python3 -c "import torch; print(torch.__version__)"  # 例如: 2.5.0

# 2. 根据你的 PyTorch 版本安装对应的 PyG
#    查看可用 wheel: https://data.pyg.org/whl/
pip install torch-scatter torch-sparse torch-cluster \
    -f https://data.pyg.org/whl/torch-2.5.0+cpu.html
pip install torch-geometric

# 3. 如果 wheel 版本不匹配，尝试从源码安装
pip install torch-scatter torch-sparse torch-cluster torch-geometric --no-build-isolation
```

### Q2: 我没有 GPU 怎么办？

本项目所有代码都支持 CPU 运行。Cora 数据集（2708 个节点）在 CPU 上几分钟就能训练完。模拟风控数据（5000 个节点）也在几分钟内完成。

### Q3: 为什么 GCN 只用 5% 的标签就能达到 80%+ 准确率？

这就是**半监督学习**的威力。图的边就是免费的信息通道——140 个标签通过 10556 条边传播到整个图。关键信息通过引用关系在图结构中流动，即使大部分节点没有标签，模型也能学到有效的分类。

### Q4: 用自己的数据集该怎么构建图？

```python
from src.data.graph_builder import build_heterogeneous_graph

# 准备数据
data = build_heterogeneous_graph(
    node_features_dict={
        'user': user_features,      # (N1, F1) numpy array
        'device': device_features,  # (N2, F2) numpy array
    },
    edge_dict={
        ('user', 'uses', 'device'): edges,  # (E, 2) numpy array
    },
    labels_dict={'user': labels},  # 0=正常, 1=欺诈
)
```

### Q5: 欺诈占比只有 1%，该怎么处理？

三招组合：
1. **Focal Loss** — 让模型更关注难分类的欺诈样本（`src/train.py` 已实现）
2. **PR-AUC 评估** — 不要看 Accuracy（会骗你）
3. **Precision@K** — 只看最高风险的 Top-K 个用户

---

## 学习路径建议

### 🌱 新人学习路径（3-5 天）

**Day 1 - 理解核心概念**
- 阅读 `docs/01-GNN概述与基础.md` 和 `docs/02-GCN原理详解.md`（共约 1 小时）
- 用自己的话总结：图是什么、GNN 是什么、为什么适合风控

**Day 2 - 跑通第一个 GCN**
- 运行 `notebooks/01-GCN-Cora节点分类入门.ipynb`
- 修改隐藏层维度（16→32→64），观察效果变化
- 修改层数（2→3→4），观察过平滑现象
- 理解每一行代码在做什么

**Day 3 - 理解风控场景**
- 阅读 `docs/03-黑产风控场景与图建模方法论.md`
- 运行 `notebooks/02-风控异构图构建.ipynb`
- 观察团伙在图结构上的模式
- 思考：如果让你设计一个风控图，需要哪些实体和关系？

**Day 4 - 实战欺诈检测**
- 运行 `notebooks/03-欺诈团伙检测实战.ipynb`
- 对比 Focal Loss vs 加权 CrossEntropy 的效果
- 观察不同阈值下的 Precision/Recall 变化
- 评估模型对团伙的检出能力

**Day 5 - 深入理解**
- 运行 `notebooks/04-模型解释性与效果评估.ipynb`
- 做模型错误分析（什么样的人会被漏掉？）
- 阅读 `src/models/` 源码，理解各模型的实现
- 尝试用自己的理解把 GCN 讲给别人听

### 🌿 进阶提升路径

1. **模型对比** — 在相同数据上对比 GCN / GAT / GraphSAGE / HeteroGNN
2. **特征工程深入** — 添加图结构特征（度中心性、PageRank、聚类系数）
3. **大规模图处理** — 用 GraphSAGE 的 mini-batch 训练处理更大规模数据
4. **部署实战** — 按 `docs/05-模型部署与上线指南.md` 用 Docker 部署模型服务
5. **论文精读** — 读 GCN/GraphSAGE/GAT 原论文，理解设计动机

---

## 🔗 参考资源

- [PyTorch Geometric 官方文档](https://pytorch-geometric.readthedocs.io/)
- [GCN 原论文 (Kipf & Welling, 2017)](https://arxiv.org/abs/1609.02907)
- [GraphSAGE 原论文 (Hamilton et al., 2017)](https://arxiv.org/abs/1706.02216)
- [GAT 原论文 (Veličković et al., 2018)](https://arxiv.org/abs/1710.10903)
- [图欺诈检测论文合集 (GitHub)](https://github.com/safe-graph/graph-fraud-detection-papers)
- [Stanford CS224W: Machine Learning with Graphs](https://web.stanford.edu/class/cs224w/)
- [NebulaGraph 图数据库](https://www.nebula-graph.io/)

---

## 📄 License

MIT License - 自由使用、修改和分发。

---

> 💡 **学习建议：先跑通 Cora GCN 的代码建立信心，再逐步深入理解原理。风控图建模最重要的是正确的图 Schema 设计——想清楚什么实体和关系最重要，比调模型参数更有价值。**
>
> 遇到问题？先看文档中的常见问题部分，再看源码注释（每个函数都有详细说明），最后欢迎提 Issue。
