"""
风控模拟数据生成器

生成包含黑产团伙特征的模拟数据集，用于 GNN 欺诈检测模型训练。

数据模拟策略：
  1. 正常用户：独立的设备、IP、手机号，行为模式分散
  2. 欺诈用户（团伙）：共享设备/IP/手机号，形成密集的子图结构
  3. 欺诈用户（个体）：使用异常行为模式（如频繁更换设备）

生成的数据包含：
  - 用户节点特征：注册天数、活跃度、设备更换频率、交易金额等
  - 设备节点特征：使用用户数、是否模拟器、设备类型等
  - IP 节点特征：连接用户数、IP 类型（代理/机房/家庭）等
  - 边：用户-设备（登录关系）、用户-IP（网络连接关系）、用户-手机号
"""

import numpy as np
import pandas as pd
import pickle
import os
from pathlib import Path
from typing import Dict, Tuple, List, Optional
from sklearn.preprocessing import StandardScaler


def generate_fraud_dataset(
    n_users: int = 5000,
    n_devices: int = 3000,
    n_ips: int = 2000,
    n_phones: int = 4000,
    fraud_ratio: float = 0.05,
    gang_size_range: Tuple[int, int] = (5, 30),
    n_gangs: int = 8,
    seed: int = 42,
) -> Dict:
    """
    生成风控模拟数据集。

    Parameters
    ----------
    n_users : int
        用户总数
    n_devices : int
        设备总数
    n_ips : int
        IP 总数
    n_phones : int
        手机号总数
    fraud_ratio : float
        欺诈用户占比
    gang_size_range : Tuple[int, int]
        每个团伙的人数范围 (min, max)
    n_gangs : int
        欺诈团伙数量
    seed : int
        随机种子

    Returns
    -------
    Dict 包含:
        - 'user_features': np.ndarray (n_users, feature_dim)
        - 'device_features': np.ndarray (n_devices, feature_dim)
        - 'ip_features': np.ndarray (n_ips, feature_dim)
        - 'phone_features': np.ndarray (n_phones, feature_dim)
        - 'user_device_edges': np.ndarray (E1, 2)
        - 'user_ip_edges': np.ndarray (E2, 2)
        - 'user_phone_edges': np.ndarray (E3, 2)
        - 'labels': np.ndarray (n_users,)  — 0:正常, 1:欺诈
        - 'gang_ids': np.ndarray (n_users,) — -1:非团伙, 0..n_gangs-1:团伙编号
        - 'feature_names': Dict[str, List[str]]
    """
    rng = np.random.default_rng(seed)

    # ========================================
    # 1. 确定欺诈用户和正常用户
    # ========================================
    n_fraud = int(n_users * fraud_ratio)

    # 分配各团伙人数
    gang_sizes = rng.integers(gang_size_range[0], gang_size_range[1] + 1, n_gangs)
    gang_sizes = (gang_sizes * n_fraud / gang_sizes.sum()).astype(int)
    gang_sizes[-1] += n_fraud - gang_sizes.sum()  # 补齐差异

    # 创建标签和团伙编号
    labels = np.zeros(n_users, dtype=int)
    gang_ids = np.full(n_users, -1, dtype=int)

    fraud_indices = rng.choice(n_users, size=n_fraud, replace=False)
    labels[fraud_indices] = 1

    gang_start = 0
    for gang_id, gs in enumerate(gang_sizes):
        gang_members = fraud_indices[gang_start : gang_start + gs]
        gang_ids[gang_members] = gang_id
        gang_start += gs

    # ========================================
    # 2. 生成节点特征
    # ========================================

    # --- 用户特征 ---
    # 正常用户的行为特征
    normal_reg_days = rng.exponential(100, n_users - n_fraud)  # 注册天数（偏长）
    normal_active = rng.beta(2, 5, n_users - n_fraud)  # 活跃度（偏低）
    normal_device_changes = rng.poisson(1, n_users - n_fraud)  # 设备更换（少）
    normal_txn_amount = rng.lognormal(4, 1, n_users - n_fraud)  # 交易金额

    # 欺诈用户的行为特征（明显不同）
    fraud_reg_days = rng.exponential(10, n_fraud)  # 注册天数（偏短，新注册号）
    fraud_active = rng.beta(8, 2, n_fraud)  # 活跃度（偏高，批量操作）
    fraud_device_changes = rng.poisson(5, n_fraud)  # 设备更换（频繁）
    fraud_txn_amount = rng.lognormal(3, 0.5, n_fraud)  # 交易金额（相对集中）

    user_features = np.zeros((n_users, 8))
    # 特征0: 注册天数
    user_features[labels == 0, 0] = normal_reg_days
    user_features[labels == 1, 0] = fraud_reg_days
    # 特征1: 活跃度评分 (0-1)
    user_features[labels == 0, 1] = normal_active
    user_features[labels == 1, 1] = fraud_active
    # 特征2: 近30天设备更换次数
    user_features[labels == 0, 2] = normal_device_changes
    user_features[labels == 1, 2] = fraud_device_changes
    # 特征3: 平均交易金额
    user_features[labels == 0, 3] = normal_txn_amount
    user_features[labels == 1, 3] = fraud_txn_amount
    # 特征4: 近7天登录次数
    user_features[labels == 0, 4] = rng.poisson(2, n_users - n_fraud)
    user_features[labels == 1, 4] = rng.poisson(15, n_fraud)
    # 特征5: 账号等级（0-5，正常用户偏高）
    user_features[labels == 0, 5] = rng.integers(1, 6, n_users - n_fraud)
    user_features[labels == 1, 5] = rng.integers(0, 2, n_fraud)
    # 特征6: 是否绑定实名（0/1）
    user_features[labels == 0, 6] = rng.binomial(1, 0.85, n_users - n_fraud)
    user_features[labels == 1, 6] = rng.binomial(1, 0.15, n_fraud)
    # 特征7: 好友/关注数
    user_features[labels == 0, 7] = rng.poisson(30, n_users - n_fraud)
    user_features[labels == 1, 7] = rng.poisson(2, n_fraud)

    # 噪声 + 非负处理
    user_features += rng.normal(0, 0.1, user_features.shape)
    user_features = np.maximum(user_features, 0)

    # --- 设备特征 ---
    device_features = np.zeros((n_devices, 6))
    device_features[:, 0] = rng.exponential(1, n_devices)  # 设备关联用户数预期
    device_features[:, 1] = rng.binomial(1, 0.05, n_devices)  # 是否为模拟器
    device_features[:, 2] = rng.choice([0, 1, 2], n_devices, p=[0.7, 0.2, 0.1])  # 设备类型
    device_features[:, 3] = rng.lognormal(3, 1, n_devices)  # 设备使用时长(天)
    device_features[:, 4] = rng.beta(2, 2, n_devices)  # 设备健康度
    device_features[:, 5] = rng.poisson(3, n_devices)  # 安装APP数

    # --- IP 特征 ---
    ip_features = np.zeros((n_ips, 5))
    ip_features[:, 0] = rng.exponential(0.5, n_ips)  # 关联用户数
    ip_features[:, 1] = rng.choice([0, 1, 2], n_ips, p=[0.6, 0.25, 0.15])  # IP类型(家庭/企业/机房)
    ip_features[:, 2] = rng.uniform(0, 1, n_ips)  # IP 风险评分
    ip_features[:, 3] = rng.poisson(50, n_ips)  # 日均流量
    ip_features[:, 4] = rng.beta(1, 5, n_ips)  # 历史黑名单关联度

    # --- 手机号特征 ---
    phone_features = np.zeros((n_phones, 4))
    phone_features[:, 0] = rng.exponential(0.5, n_phones)  # 关联用户数
    phone_features[:, 1] = rng.choice([0, 1], n_phones, p=[0.9, 0.1])  # 是否虚拟号
    phone_features[:, 2] = rng.lognormal(6, 1, n_phones)  # 注册时间(天)
    phone_features[:, 3] = rng.beta(1, 3, n_phones)  # 归属地风险评分

    # ========================================
    # 3. 生成边（关系）
    # ========================================

    user_device_edges = []
    user_ip_edges = []
    user_phone_edges = []

    # --- 正常用户关系：每个用户独立使用1-2个设备和IP ---
    for user_id in range(n_users):
        if labels[user_id] == 0:
            # 正常用户：1-2 个设备，1-3 个 IP，1-2 个手机号
            n_dev = rng.integers(1, 3)
            n_ip = rng.integers(1, 4)
            n_ph = rng.integers(1, 3)

            devices = rng.choice(n_devices, size=n_dev, replace=False)
            ips = rng.choice(n_ips, size=n_ip, replace=False)
            phones = rng.choice(n_phones, size=n_ph, replace=False)

            for d in devices:
                user_device_edges.append([user_id, d])
            for ip in ips:
                user_ip_edges.append([user_id, ip])
            for ph in phones:
                user_phone_edges.append([user_id, ph])

    # --- 欺诈团伙关系：团伙内共享设备和IP（这是关键模式） ---
    for gang_id in range(n_gangs):
        gang_members = np.where(gang_ids == gang_id)[0]

        # 团伙共享少量设备/IP/手机号（形成密集子图）
        shared_devices = rng.choice(n_devices, size=max(1, len(gang_members) // 8), replace=False)
        shared_ips = rng.choice(n_ips, size=max(1, len(gang_members) // 5), replace=False)
        shared_phones = rng.choice(n_phones, size=max(1, len(gang_members) // 6), replace=False)

        for user_id in gang_members:
            for d in shared_devices:
                user_device_edges.append([user_id, d])
            for ip in shared_ips:
                user_ip_edges.append([user_id, ip])
            for ph in shared_phones:
                user_phone_edges.append([user_id, ph])

            # 个别用户还可能有额外的设备
            extra_dev = rng.choice(n_devices, size=rng.integers(0, 2), replace=False)
            for d in extra_dev:
                user_device_edges.append([user_id, d])

    # ========================================
    # 4. 后处理：特征标准化 + 去重
    # ========================================

    user_device_edges = np.unique(np.array(user_device_edges), axis=0)
    user_ip_edges = np.unique(np.array(user_ip_edges), axis=0)
    user_phone_edges = np.unique(np.array(user_phone_edges), axis=0)

    # 更新设备/IP 特征中的实际关联用户数
    for i, edges in enumerate([user_device_edges, user_ip_edges, user_phone_edges]):
        if i == 0:
            unique, counts = np.unique(edges[:, 1], return_counts=True)
            device_features[unique, 0] = counts
        elif i == 1:
            unique, counts = np.unique(edges[:, 1], return_counts=True)
            ip_features[unique, 0] = counts
        else:
            unique, counts = np.unique(edges[:, 1], return_counts=True)
            phone_features[unique, 0] = counts

    # 特征名称
    feature_names = {
        "user": [
            "注册天数", "活跃度评分", "设备更换次数", "平均交易金额",
            "7天登录次数", "账号等级", "是否实名", "好友关注数",
        ],
        "device": [
            "关联用户数", "是否模拟器", "设备类型", "使用时长",
            "设备健康度", "安装APP数",
        ],
        "ip": [
            "关联用户数", "IP类型", "风险评分", "日均流量", "黑名单关联度",
        ],
        "phone": [
            "关联用户数", "是否虚拟号", "注册时间", "归属地风险",
        ],
    }

    return {
        "user_features": user_features,
        "device_features": device_features,
        "ip_features": ip_features,
        "phone_features": phone_features,
        "user_device_edges": user_device_edges,
        "user_ip_edges": user_ip_edges,
        "user_phone_edges": user_phone_edges,
        "labels": labels,
        "gang_ids": gang_ids,
        "feature_names": feature_names,
    }


def generate_and_save(
    output_path: str = "data/fraud_dataset.pkl",
    **kwargs,
) -> Dict:
    """
    生成模拟数据集并保存到磁盘。

    Parameters
    ----------
    output_path : str
        输出文件路径
    **kwargs
        传递给 generate_fraud_dataset 的参数

    Returns
    -------
    Dict
        生成的数据集
    """
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else "data", exist_ok=True)

    dataset = generate_fraud_dataset(**kwargs)

    with open(output_path, "wb") as f:
        pickle.dump(dataset, f)

    print(f"数据集已保存至: {output_path}")
    print_dataset_summary(dataset)

    return dataset


def print_dataset_summary(dataset: Dict) -> None:
    """打印模拟数据集的摘要信息。"""
    n_users = dataset["user_features"].shape[0]
    n_fraud = dataset["labels"].sum()
    n_gangs = len(np.unique(dataset["gang_ids"][dataset["gang_ids"] >= 0]))

    print("=" * 55)
    print("风控模拟数据集摘要")
    print("=" * 55)
    print(f"  用户总数:          {n_users:,}")
    print(f"  欺诈用户:          {n_fraud:,} ({n_fraud/n_users*100:.2f}%)")
    print(f"  欺诈团伙数:        {n_gangs}")
    print(f"  设备数:            {dataset['device_features'].shape[0]:,}")
    print(f"  IP 数:             {dataset['ip_features'].shape[0]:,}")
    print(f"  手机号数:          {dataset['phone_features'].shape[0]:,}")
    print(f"  用户-设备边:       {len(dataset['user_device_edges']):,}")
    print(f"  用户-IP 边:        {len(dataset['user_ip_edges']):,}")
    print(f"  用户-手机号边:     {len(dataset['user_phone_edges']):,}")

    # 团伙统计
    for gang_id in range(n_gangs):
        members = np.where(dataset["gang_ids"] == gang_id)[0]
        print(f"  团伙 {gang_id}: {len(members)} 个成员")
    print("=" * 55)


# ========================================
# 命令行入口
# ========================================
if __name__ == "__main__":
    generate_and_save(
        output_path="data/fraud_dataset.pkl",
        n_users=5000,
        n_devices=3000,
        n_ips=2000,
        n_phones=4000,
        fraud_ratio=0.05,
        n_gangs=8,
        seed=42,
    )
