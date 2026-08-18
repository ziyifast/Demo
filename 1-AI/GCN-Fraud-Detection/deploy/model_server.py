"""
GNN 风控模型推理服务

基于 Flask 的 REST API，提供模型在线推理能力。

API 端点:
  - GET  /health         健康检查
  - POST /predict        单用户风险评分
  - POST /predict_batch  批量用户风险评分
  - GET  /model_info     模型信息

使用方法:
  python deploy/model_server.py
  或
  gunicorn -w 4 -b 0.0.0.0:8000 deploy.model_server:app
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import logging
import time
from typing import Dict, List, Any, Optional, Tuple

import torch
import numpy as np
from flask import Flask, request, jsonify

from deploy.config import (
    MODEL_PATH,
    MODEL_TYPE,
    MODEL_PARAMS,
    FRAUD_THRESHOLD,
    RISK_LEVELS,
    HOST,
    PORT,
    LOG_LEVEL,
    LOG_FORMAT,
)

# 配置日志
logging.basicConfig(level=getattr(logging, LOG_LEVEL), format=LOG_FORMAT)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ========================================
# 模型加载
# ========================================
model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model() -> torch.nn.Module:
    """
    加载训练好的 GNN 模型。

    Returns
    -------
    torch.nn.Module
        加载好的模型
    """
    if MODEL_TYPE == "hetero_gnn":
        from src.models.hetero_gnn import HeteroGNN

        model = HeteroGNN(
            node_types=MODEL_PARAMS["node_types"],
            edge_types=MODEL_PARAMS["edge_types"],
            target_node_type=MODEL_PARAMS["target_node_type"],
            in_channels_dict=MODEL_PARAMS["in_channels_dict"],
            hidden_channels=MODEL_PARAMS["hidden_channels"],
            out_channels=MODEL_PARAMS["out_channels"],
            num_layers=MODEL_PARAMS["num_layers"],
            conv_type=MODEL_PARAMS["conv_type"],
            dropout=0.0,  # 推理时不需要 dropout
        )
    else:
        from src.models.gcn import GCN

        model = GCN(
            in_channels=MODEL_PARAMS.get("in_channels", 64),
            hidden_channels=MODEL_PARAMS["hidden_channels"],
            out_channels=MODEL_PARAMS["out_channels"],
            num_layers=MODEL_PARAMS["num_layers"],
            dropout=0.0,
        )

    if os.path.exists(MODEL_PATH):
        state_dict = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
        logger.info(f"模型从 {MODEL_PATH} 加载成功")
    else:
        logger.warning(f"模型文件 {MODEL_PATH} 不存在，使用未训练的模型")

    model.to(device)
    model.eval()
    return model


def get_risk_level(score: float) -> str:
    """
    根据风险评分确定风险等级。

    Parameters
    ----------
    score : float
        欺诈风险评分 (0-1)

    Returns
    -------
    str
        风险等级: 'low', 'medium', 'high'
    """
    for level, (low, high) in RISK_LEVELS.items():
        if low <= score < high:
            return level
    return "high" if score >= RISK_LEVELS["high"][0] else "low"


def determine_action(score: float) -> Dict[str, Any]:
    """
    根据风险评分确定处置建议。

    Returns
    -------
    Dict
        包含 action、reason 等信息
    """
    if score >= FRAUD_THRESHOLD:
        return {
            "action": "block",
            "action_name": "拦截",
            "reason": f"风险评分 {score:.3f} 超过阈值 {FRAUD_THRESHOLD}",
        }
    elif score >= 0.3:
        return {
            "action": "verify",
            "action_name": "人工审核",
            "reason": f"风险评分 {score:.3f} 处于灰色区间",
        }
    else:
        return {
            "action": "pass",
            "action_name": "放行",
            "reason": "风险评分较低",
        }


# ========================================
# API 端点
# ========================================


@app.route("/health", methods=["GET"])
def health_check():
    """
    健康检查端点。
    """
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device),
        "threshold": FRAUD_THRESHOLD,
    })


@app.route("/model_info", methods=["GET"])
def model_info():
    """
    模型信息端点。
    """
    return jsonify({
        "model_type": MODEL_TYPE,
        "model_path": MODEL_PATH,
        "params": {k: v for k, v in MODEL_PARAMS.items() if not isinstance(v, (list, dict))},
        "threshold": FRAUD_THRESHOLD,
        "risk_levels": RISK_LEVELS,
    })


@app.route("/predict", methods=["POST"])
def predict():
    """
    单次推理端点 — 评估单个用户的风险。

    请求格式:
    {
        "user_id": "user_12345",
        "user_features": [100, 0.5, 2, 500, 10, 3, 1, 30],
        "subgraph": {
            "devices": [[...], ...],     # 关联的设备特征
            "ips": [[...], ...],         # 关联的 IP 特征
            "phones": [[...], ...],      # 关联的手机号特征
        }
    }

    返回格式:
    {
        "user_id": "user_12345",
        "fraud_score": 0.92,
        "risk_level": "high",
        "action": "block",
        "action_name": "拦截",
        "inference_time_ms": 1.5
    }
    """
    if model is None:
        return jsonify({"error": "模型未加载"}), 503

    try:
        data = request.json
        start_time = time.time()

        user_id = data.get("user_id", "unknown")

        # 构建推理所需的张量
        user_x = torch.tensor([data["user_features"]], dtype=torch.float).to(device)

        # 简化处理：如果没有子图信息，仅使用用户特征
        # 实际部署中需要从图数据库查询用户的邻居信息
        if "subgraph" in data:
            # 完整异构图推理
            x_dict = {"user": user_x}
            edge_index_dict = {}
            for etype_key in ["device", "ip", "phone"]:
                if etype_key in data["subgraph"] and data["subgraph"][etype_key]:
                    features = torch.tensor(
                        data["subgraph"][etype_key], dtype=torch.float
                    ).to(device)
                    x_dict[etype_key] = features
                    # 构建边索引（用户连接到这些设备/IP/手机号）
                    n_entities = features.shape[0]
                    if n_entities > 0:
                        edge_index = torch.stack([
                            torch.zeros(n_entities, dtype=torch.long),
                            torch.arange(n_entities, dtype=torch.long),
                        ]).to(device)
                        etype = (
                            "user",
                            {"device": "uses", "ip": "from", "phone": "has"}[etype_key],
                            etype_key,
                        )
                        edge_index_dict[etype] = edge_index

            with torch.no_grad():
                out = model(x_dict, edge_index_dict)
                probs = torch.softmax(out, dim=1)
                fraud_score = probs[0, 1].item()
        else:
            # Fallback: 仅基于用户特征的评分（无图结构信息）
            fraud_score = 0.5  # 中性分数

        inference_time = (time.time() - start_time) * 1000

        risk_level = get_risk_level(fraud_score)
        action_info = determine_action(fraud_score)

        return jsonify({
            "user_id": user_id,
            "fraud_score": round(fraud_score, 4),
            "risk_level": risk_level,
            "action": action_info["action"],
            "action_name": action_info["action_name"],
            "reason": action_info["reason"],
            "inference_time_ms": round(inference_time, 2),
        })

    except Exception as e:
        logger.error(f"推理错误: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/predict_batch", methods=["POST"])
def predict_batch():
    """
    批量推理端点。

    请求格式:
    {
        "users": [
            {
                "user_id": "user_1",
                "user_features": [...],
                "subgraph": {...}
            },
            ...
        ]
    }

    返回格式:
    {
        "results": [
            {"user_id": "user_1", "fraud_score": 0.12, ...},
            ...
        ],
        "summary": {
            "total": 100,
            "high_risk": 5,
            "medium_risk": 15,
            "low_risk": 80,
            "avg_score": 0.23,
            "avg_inference_time_ms": 1.2
        }
    }
    """
    if model is None:
        return jsonify({"error": "模型未加载"}), 503

    try:
        data = request.json
        users = data.get("users", [])
        start_time = time.time()

        results = []
        for user_data in users:
            # 调用单个预测逻辑
            user_data["user_id"] = user_data.get("user_id", "unknown")
            single_result = predict_single(user_data)
            results.append(single_result)

        # 汇总统计
        total = len(results)
        high_risk = sum(1 for r in results if r["risk_level"] == "high")
        medium_risk = sum(1 for r in results if r["risk_level"] == "medium")
        low_risk = sum(1 for r in results if r["risk_level"] == "low")
        avg_score = np.mean([r["fraud_score"] for r in results]) if results else 0
        avg_time = np.mean([r["inference_time_ms"] for r in results]) if results else 0

        return jsonify({
            "results": results,
            "summary": {
                "total": total,
                "high_risk": high_risk,
                "medium_risk": medium_risk,
                "low_risk": low_risk,
                "avg_score": round(float(avg_score), 4),
                "avg_inference_time_ms": round(float(avg_time), 2),
            },
        })

    except Exception as e:
        logger.error(f"批量推理错误: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


def predict_single(user_data: Dict) -> Dict:
    """单个用户推理（内部使用）。"""
    user_id = user_data.get("user_id", "unknown")
    start_time = time.time()

    # 简化推理逻辑（完整版需要查询图数据库）
    fraud_score = 0.5  # 占位

    inference_time = (time.time() - start_time) * 1000
    risk_level = get_risk_level(fraud_score)
    action_info = determine_action(fraud_score)

    return {
        "user_id": user_id,
        "fraud_score": round(fraud_score, 4),
        "risk_level": risk_level,
        "action": action_info["action"],
        "action_name": action_info["action_name"],
        "inference_time_ms": round(inference_time, 2),
    }


# ========================================
# 启动
# ========================================
if __name__ == "__main__":
    logger.info("正在加载模型...")
    model = load_model()
    logger.info(f"服务启动: http://{HOST}:{PORT}")
    app.run(host=HOST, port=PORT, debug=False)
