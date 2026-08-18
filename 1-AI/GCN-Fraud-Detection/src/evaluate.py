"""
模型评估与可视化模块

提供风控场景下的模型评估方法：
  1. 标准分类指标（ROC-AUC, PR-AUC, F1, Confusion Matrix）
  2. 节点嵌入可视化（t-SNE / UMAP）
  3. 欺诈团伙发现与评估
  4. 业务指标（Precision@K, Recall@K, 拦截率等）
"""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
)
from collections import defaultdict
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: Optional[np.ndarray] = None,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    计算分类模型的全套评估指标。

    Parameters
    ----------
    y_true : np.ndarray
        真实标签（0/1）
    y_pred : np.ndarray
        预测标签（0/1）
    y_score : np.ndarray, optional
        预测概率（欺诈类的概率），用于计算 AUC
    threshold : float
        分类阈值

    Returns
    -------
    Dict[str, float]
        各项评估指标
    """
    metrics = {}

    # 基本指标
    metrics["accuracy"] = (y_true == y_pred).mean()
    metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)
    metrics["recall"] = recall_score(y_true, y_pred, zero_division=0)
    metrics["f1"] = f1_score(y_true, y_pred, zero_division=0)

    # AUC
    if y_score is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_score)
            metrics["pr_auc"] = average_precision_score(y_true, y_score)
        except ValueError:
            metrics["roc_auc"] = float("nan")
            metrics["pr_auc"] = float("nan")

    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    metrics["tp"] = int(tp)
    metrics["fp"] = int(fp)
    metrics["fn"] = int(fn)
    metrics["tn"] = int(tn)
    metrics["false_positive_rate"] = fp / (fp + tn + 1e-8)
    metrics["false_negative_rate"] = fn / (fn + tp + 1e-8)

    return metrics


def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    """
    计算 Precision@K —— 在风控场景中，我们通常只对最高风险的 K 个用户采取措施。

    Parameters
    ----------
    y_true : np.ndarray
        真实标签
    y_score : np.ndarray
        预测分数（概率）
    k : int
        选取的 top-K 数量

    Returns
    -------
    float
        Precision@K
    """
    top_k_indices = np.argsort(y_score)[-k:]
    y_true_topk = y_true[top_k_indices]
    return y_true_topk.mean()


def find_optimal_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric: str = "f1",
) -> Tuple[float, float]:
    """
    寻找最优分类阈值（基于 PR 曲线）。

    Parameters
    ----------
    y_true : np.ndarray
        真实标签
    y_score : np.ndarray
        预测概率
    metric : str
        优化目标，'f1' | 'precision' | 'recall'

    Returns
    -------
    threshold : float
        最优阈值
    best_score : float
        最优分数
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)

    if metric == "f1":
        scores = 2 * precision * recall / (precision + recall + 1e-8)
    elif metric == "precision":
        scores = precision
    elif metric == "recall":
        scores = recall
    else:
        raise ValueError(f"不支持的 metric: {metric}")

    best_idx = np.argmax(scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    best_score = scores[best_idx]

    return best_threshold, best_score


def evaluate_on_thresholds(
    y_true: np.ndarray,
    y_score: np.ndarray,
    thresholds: List[float] = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
) -> pd.DataFrame:
    """
    在不同阈值下评估模型，用于选择业务最优策略。

    Returns
    -------
    pd.DataFrame
        每个阈值对应的各项指标
    """
    rows = []
    for t in thresholds:
        y_pred = (y_score >= t).astype(int)
        metrics = compute_metrics(y_true, y_pred, y_score)
        metrics["threshold"] = t
        rows.append(metrics)
    return pd.DataFrame(rows)


def evaluate_gang_detection(
    gang_ids: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.7,
) -> Dict:
    """
    评估团伙检测效果。

    思路：在风控中，光发现单个欺诈用户还不够，能够"连根拔起"整个团伙
    才是图方法的价值所在。这个函数评估：
      - 多少个团伙被准确识别（团伙中超过一半成员被标记）
      - 团伙发现覆盖率

    Parameters
    ----------
    gang_ids : np.ndarray
        每个用户的团伙编号（-1 表示非团伙成员）
    y_score : np.ndarray
        模型预测的欺诈概率
    threshold : float
        判定为欺诈的阈值

    Returns
    -------
    Dict
        团伙检测指标
    """
    predicted_fraud = y_score >= threshold

    results = {}
    unique_gangs = np.unique(gang_ids[gang_ids >= 0])

    fully_detected = 0
    partially_detected = 0
    gang_coverage_rates = []

    for gang_id in unique_gangs:
        gang_mask = gang_ids == gang_id
        gang_size = gang_mask.sum()
        detected = (predicted_fraud & gang_mask).sum()
        coverage = detected / gang_size

        gang_coverage_rates.append(coverage)

        if coverage > 0.8:
            fully_detected += 1
        elif coverage > 0.3:
            partially_detected += 1

    n_gangs = len(unique_gangs)

    results["total_gangs"] = n_gangs
    results["fully_detected_gangs"] = fully_detected
    results["partially_detected_gangs"] = partially_detected
    results["missed_gangs"] = n_gangs - fully_detected - partially_detected
    results["avg_gang_coverage"] = np.mean(gang_coverage_rates) if gang_coverage_rates else 0

    return results


def print_evaluation_summary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: Optional[np.ndarray] = None,
) -> None:
    """
    打印完整的模型评估报告。
    """
    metrics = compute_metrics(y_true, y_pred, y_score)

    print("=" * 55)
    print("模型评估报告")
    print("=" * 55)
    print(f"  Accuracy:          {metrics['accuracy']:.4f}")
    print(f"  Precision:         {metrics['precision']:.4f}")
    print(f"  Recall:            {metrics['recall']:.4f}")
    print(f"  F1 Score:          {metrics['f1']:.4f}")
    if y_score is not None:
        print(f"  ROC-AUC:           {metrics.get('roc_auc', np.nan):.4f}")
        print(f"  PR-AUC:            {metrics.get('pr_auc', np.nan):.4f}")
    print(f"  False Positive Rate: {metrics['false_positive_rate']:.4f}")
    print(f"  False Negative Rate: {metrics['false_negative_rate']:.4f}")
    print("-" * 55)
    print(f"  混淆矩阵:")
    print(f"    TP: {metrics['tp']:<6} FP: {metrics['fp']}")
    print(f"    FN: {metrics['fn']:<6} TN: {metrics['tn']}")
    print("=" * 55)
    print()
    print(classification_report(y_true, y_pred, target_names=["正常", "欺诈"]))


def visualize_embeddings(
    embeddings: np.ndarray,
    labels: np.ndarray,
    method: str = "umap",
    random_state: int = 42,
):
    """
    对节点嵌入进行降维可视化。

    Parameters
    ----------
    embeddings : np.ndarray, shape (N, D)
        节点嵌入向量
    labels : np.ndarray, shape (N,)
        节点标签
    method : str
        'umap' | 'tsne'
    random_state : int
        随机种子

    Returns
    -------
    fig : matplotlib Figure
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    if method == "umap":
        import umap
        reducer = umap.UMAP(random_state=random_state)
        coords = reducer.fit_transform(embeddings)
    elif method == "tsne":
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, random_state=random_state)
        coords = reducer.fit_transform(embeddings)
    else:
        raise ValueError(f"不支持的方法: {method}，请选择 'umap' 或 'tsne'")

    fig, ax = plt.subplots(figsize=(12, 8))
    scatter = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=labels,
        cmap="coolwarm",
        alpha=0.6,
        s=10,
    )
    ax.set_title(f"节点嵌入可视化 ({method.upper()})", fontsize=14)
    legend = ax.legend(*scatter.legend_elements(), title="标签")
    ax.add_artist(legend)
    fig.colorbar(scatter, ax=ax, label="类别")

    return fig
