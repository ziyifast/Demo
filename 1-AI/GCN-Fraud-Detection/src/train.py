"""
通用 GNN 训练模块

提供标准化的训练流程：
  1. 同构图节点分类训练
  2. 异构图节点分类训练（处理类别不平衡）
  3. 早停（early stopping）机制

在风控场景中，由于正负样本极不均衡（欺诈用户占比通常 < 5%），
训练时需要特别关注：
  - 使用加权损失函数（weighted CrossEntropy）
  - 使用 Focal Loss 缓解类别不平衡
  - 重点关注 Precision/Recall 而非仅看 Accuracy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, Callable
from copy import deepcopy
from collections import defaultdict


class EarlyStopping:
    """
    早停机制：当验证集指标在 patience 个 epoch 内没有提升时停止训练。

    Parameters
    ----------
    patience : int
        容忍多少个 epoch 不提升
    min_delta : float
        最小的提升阈值
    mode : str
        'max' 表示指标越大越好（如 accuracy），'min' 表示越小越好（如 loss）
    restore_best : bool
        是否在停止时恢复最佳模型参数
    """

    def __init__(
        self,
        patience: int = 50,
        min_delta: float = 1e-4,
        mode: str = "max",
        restore_best: bool = True,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best = restore_best

        self.best_score = None
        self.best_epoch = 0
        self.counter = 0
        self.best_state_dict = None
        self.should_stop = False

    def step(self, score: float, model: nn.Module, epoch: int) -> bool:
        """
        执行一步检查。

        Returns
        -------
        bool
            是否应该停止训练
        """
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            self.best_state_dict = deepcopy(model.state_dict())
            return False

        if self.mode == "max":
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            self.best_state_dict = deepcopy(model.state_dict())
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.should_stop = True
            if self.restore_best and self.best_state_dict is not None:
                model.load_state_dict(self.best_state_dict)

        return self.should_stop


class FocalLoss(nn.Module):
    """
    Focal Loss，用于缓解类别不平衡问题。

    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    其中：
      - α_t: 类别权重
      - γ: 聚焦参数，降低易分类样本的 loss 权重
      - p_t: 模型对正确类别的预测概率

    在风控场景中，正常样本（负样本）占绝大多数且容易分类，
    Focal Loss 可以降低这些样本的 loss 贡献，让模型更关注难以分类的欺诈样本。

    Parameters
    ----------
    alpha : float, optional
        正类（欺诈）的权重。1 - alpha 为负类权重
    gamma : float, default=2.0
        聚焦参数
    reduction : str, default='mean'
        'mean' | 'sum' | 'none'
    """

    def __init__(self, alpha: Optional[float] = None, gamma: float = 2.0, reduction: str = "mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        inputs : Tensor, shape (N, C)
            模型输出的 logits（不是 softmax 后的概率）
        targets : Tensor, shape (N,)
            真实标签
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        p_t = torch.exp(-ce_loss)  # 正确类别的预测概率

        # Focal Loss
        focal_loss = (1 - p_t) ** self.gamma * ce_loss

        if self.alpha is not None:
            alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
            focal_loss = alpha_t * focal_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


def compute_class_weights(labels: np.ndarray, num_classes: int = 2) -> torch.Tensor:
    """
    基于标签分布计算类别权重（用于加权 CrossEntropyLoss）。

    weight_c = N / (num_classes * N_c)

    Parameters
    ----------
    labels : np.ndarray
        标签数组
    num_classes : int
        类别总数

    Returns
    -------
    torch.Tensor, shape (num_classes,)
        每个类别的权重
    """
    class_counts = np.bincount(labels, minlength=num_classes)
    total = len(labels)
    weights = total / (num_classes * class_counts + 1e-8)
    return torch.tensor(weights, dtype=torch.float)


def train_epoch(
    model: nn.Module,
    data,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    is_hetero: bool = False,
    target_node_type: str = "user",
) -> float:
    """
    训练一个 epoch（同构图或异构图）。

    Parameters
    ----------
    model : nn.Module
        GNN 模型
    data : Data or HeteroData
        图数据
    optimizer : Optimizer
        优化器
    criterion : nn.Module
        损失函数
    device : torch.device
        计算设备
    is_hetero : bool
        是否为异构图
    target_node_type : str
        目标节点类型（仅异构图使用）

    Returns
    -------
    float
        训练 loss
    """
    model.train()
    optimizer.zero_grad()

    if is_hetero:
        # 异构图 forward
        out = model(data.x_dict, data.edge_index_dict)

        # 获取目标节点类型的标签和掩码
        y = data[target_node_type].y
        train_mask = data[target_node_type].train_mask
    else:
        # 同构图 forward
        out = model(data.x, data.edge_index)
        y = data.y
        train_mask = data.train_mask

    loss = criterion(out[train_mask], y[train_mask])
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def evaluate(
    model: nn.Module,
    data,
    device: torch.device,
    is_hetero: bool = False,
    target_node_type: str = "user",
) -> Dict[str, float]:
    """
    评估模型在训练集、验证集、测试集上的表现。

    Returns
    -------
    Dict[str, float]
        包含 loss, acc, precision, recall, f1 等指标
    """
    model.eval()

    if is_hetero:
        out = model(data.x_dict, data.edge_index_dict)
        y = data[target_node_type].y
        train_mask = data[target_node_type].train_mask
        val_mask = data[target_node_type].val_mask
        test_mask = data[target_node_type].test_mask
    else:
        out = model(data.x, data.edge_index)
        y = data.y
        train_mask = data.train_mask
        val_mask = data.val_mask
        test_mask = data.test_mask

    pred = out.argmax(dim=1)

    results = {}

    for split_name, mask in [("train", train_mask), ("val", val_mask), ("test", test_mask)]:
        if mask is None or not mask.any():
            continue

        y_true = y[mask].cpu().numpy()
        y_pred = pred[mask].cpu().numpy()

        acc = (y_true == y_pred).mean()
        results[f"{split_name}_acc"] = acc

        # 对于二分类，计算 precision, recall, f1
        if out.shape[1] == 2:
            tp = ((y_true == 1) & (y_pred == 1)).sum()
            fp = ((y_true == 0) & (y_pred == 1)).sum()
            fn = ((y_true == 1) & (y_pred == 0)).sum()

            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)

            results[f"{split_name}_precision"] = precision
            results[f"{split_name}_recall"] = recall
            results[f"{split_name}_f1"] = f1

    return results


def train_model(
    model: nn.Module,
    data,
    device: torch.device,
    epochs: int = 200,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    patience: int = 50,
    use_focal_loss: bool = False,
    focal_alpha: Optional[float] = None,
    focal_gamma: float = 2.0,
    class_weight: bool = True,
    is_hetero: bool = False,
    target_node_type: str = "user",
    verbose: bool = True,
    print_interval: int = 20,
) -> Dict:
    """
    完整的训练流程。

    Parameters
    ----------
    model : nn.Module
        GNN 模型
    data : Data or HeteroData
        图数据
    device : torch.device
        计算设备
    epochs : int
        最大训练轮数
    lr : float
        学习率
    weight_decay : float
        L2 正则化系数
    patience : int
        早停耐心值
    use_focal_loss : bool
        是否使用 Focal Loss
    focal_alpha : float, optional
        Focal Loss 的正类权重
    focal_gamma : float
        Focal Loss 的聚焦参数
    class_weight : bool
        是否在 CrossEntropyLoss 中使用类别权重
    is_hetero : bool
        是否为异构图
    target_node_type : str
        目标节点类型（仅异构图使用）
    verbose : bool
        是否打印训练日志
    print_interval : int
        每隔多少个 epoch 打印一次

    Returns
    -------
    Dict
        训练历史记录
    """
    model = model.to(device)

    # 将数据移到设备（注意异构图的特殊处理）
    if is_hetero:
        data = data.to(device)
        labels_np = data[target_node_type].y.cpu().numpy()
    else:
        data = data.to(device)
        labels_np = data.y.cpu().numpy()

    # 损失函数
    if use_focal_loss:
        criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
    else:
        if class_weight:
            weights = compute_class_weights(labels_np).to(device)
            criterion = nn.CrossEntropyLoss(weight=weights)
        else:
            criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=20, min_lr=1e-5
    )

    early_stopper = EarlyStopping(patience=patience, mode="max", restore_best=True)

    history = defaultdict(list)

    for epoch in range(1, epochs + 1):
        loss = train_epoch(
            model, data, optimizer, criterion, device,
            is_hetero=is_hetero, target_node_type=target_node_type,
        )
        history["train_loss"].append(loss)

        eval_results = evaluate(
            model, data, device,
            is_hetero=is_hetero, target_node_type=target_node_type,
        )
        for k, v in eval_results.items():
            history[k].append(v)

        # 更新学习率调度器
        val_key = "val_acc"
        if val_key not in eval_results:
            # 如果没有验证集，使用训练集指标
            val_key = next(k for k in eval_results if "acc" in k)
        scheduler.step(eval_results[val_key])

        # 早停检查
        if early_stopper.step(eval_results[val_key], model, epoch):
            if verbose:
                print(f"早停于 epoch {epoch}，最佳 {val_key}: {early_stopper.best_score:.4f}")
            break

        if verbose and epoch % print_interval == 0:
            log_str = f"Epoch {epoch:3d}, Loss: {loss:.4f}"
            for k, v in eval_results.items():
                if "acc" in k or "f1" in k:
                    log_str += f", {k}: {v:.4f}"
            print(log_str)

    # 最终评估
    final_results = evaluate(
        model, data, device,
        is_hetero=is_hetero, target_node_type=target_node_type,
    )

    if verbose:
        print("\n" + "=" * 50)
        print("训练完成，最终结果:")
        print("=" * 50)
        for k, v in final_results.items():
            print(f"  {k}: {v:.4f}")
        print(f"  最佳 epoch: {early_stopper.best_epoch}")
        print(f"  最佳 val 分数: {early_stopper.best_score:.4f}")

    return {
        "model": model,
        "history": dict(history),
        "final_results": final_results,
        "best_epoch": early_stopper.best_epoch,
        "best_score": early_stopper.best_score,
    }
