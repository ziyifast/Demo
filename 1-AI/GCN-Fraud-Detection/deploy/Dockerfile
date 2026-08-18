# GCN 风控模型 — Docker 部署文件
#
# 构建: docker build -t gcn-fraud-detector -f deploy/Dockerfile .
# 运行: docker run -p 8000:8000 gcn-fraud-detector

FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

LABEL maintainer="GCN-Fraud-Team"
LABEL description="GCN-based Fraud Detection Model Server"

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 安装 Python 依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir \
    torch-scatter torch-sparse torch-cluster \
    -f https://data.pyg.org/whl/torch-2.0.0+cu117.html && \
    pip install --no-cache-dir torch-geometric

# 复制代码
COPY src/ ./src/
COPY deploy/ ./deploy/
COPY saved_models/ ./saved_models/

# 健康检查
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

# 使用 gunicorn 作为生产服务器
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:8000", \
     "--timeout", "30", "--keep-alive", "5", \
     "--access-logfile", "-", "--error-logfile", "-", \
     "deploy.model_server:app"]
