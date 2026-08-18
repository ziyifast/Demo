"""
部署配置文件

包含模型路径、特征配置、阈值等部署参数。
"""

import os
from pathlib import Path

# ========================================
# 模型配置
# ========================================
MODEL_PATH = os.environ.get("MODEL_PATH", "saved_models/gcn_fraud.pt")
MODEL_TYPE = os.environ.get("MODEL_TYPE", "hetero_gnn")  # 'gcn' | 'hetero_gnn'

# 模型超参数（与训练时保持一致）
MODEL_PARAMS = {
    "node_types": ["user", "device", "ip", "phone"],
    "edge_types": [
        ("user", "uses", "device"),
        ("user", "from", "ip"),
        ("user", "has", "phone"),
    ],
    "target_node_type": "user",
    "in_channels_dict": {
        "user": 8,
        "device": 6,
        "ip": 5,
        "phone": 4,
    },
    "hidden_channels": 64,
    "out_channels": 2,
    "num_layers": 2,
    "conv_type": "sage",
    "dropout": 0.3,
}

# ========================================
# 推理配置
# ========================================
FRAUD_THRESHOLD = float(os.environ.get("FRAUD_THRESHOLD", "0.5"))

# 风险等级划分
RISK_LEVELS = {
    "low": (0.0, 0.3),     # 低风险 — 放行
    "medium": (0.3, 0.7),  # 中风险 — 需要额外验证
    "high": (0.7, 1.0),    # 高风险 — 直接拦截
}

# ========================================
# 服务配置
# ========================================
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8000"))
WORKERS = int(os.environ.get("WORKERS", "4"))
TIMEOUT = int(os.environ.get("TIMEOUT", "30"))

# ========================================
# 特征配置
# ========================================
# 用户特征名称（与 generate_fraud_data.py 保持一致）
USER_FEATURE_NAMES = [
    "注册天数",
    "活跃度评分",
    "设备更换次数",
    "平均交易金额",
    "7天登录次数",
    "账号等级",
    "是否实名",
    "好友关注数",
]

# 设备特征名称
DEVICE_FEATURE_NAMES = [
    "关联用户数",
    "是否模拟器",
    "设备类型",
    "使用时长",
    "设备健康度",
    "安装APP数",
]

# IP 特征名称
IP_FEATURE_NAMES = [
    "关联用户数",
    "IP类型",
    "风险评分",
    "日均流量",
    "黑名单关联度",
]

# 手机号特征名称
PHONE_FEATURE_NAMES = [
    "关联用户数",
    "是否虚拟号",
    "注册时间",
    "归属地风险",
]

# ========================================
# 日志配置
# ========================================
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
LOG_FORMAT = "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s"
