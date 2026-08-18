# data/ — 数据目录

此目录用于存放数据集和模型文件。

## 目录结构

```
data/
├── Planetoid/          # Cora/CiteSeer/PubMed 数据集（自动下载）
├── fraud_dataset.pkl   # 风控模拟数据（运行 generate_fraud_data.py 生成）
└── raw/                # 原始数据（可选）
```

## 生成数据

运行以下命令生成风控模拟数据集：

```bash
python src/data/generate_fraud_data.py
```
