"""
======================================================================
  第三步：构建知识库 —— 从文档到向量数据库
======================================================================
本脚本演示完整的"文档 → 分块 → 向量化 → 存入向量数据库"流程。

前置条件：
  1. 确保 docs/ 目录下有文档（已提供示例文件 product_manual.txt）

运行方式：
  # 使用本地 BGE 模型（免费，无需 API，推荐）
  python scripts/03_build_knowledge_base.py

  # 使用远程 Embedding API（需在 .env 中配置）
  python scripts/03_build_knowledge_base.py --embedding remote
======================================================================
"""

import os
import sys
import argparse

# 将项目根目录加入 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.knowledge_base import KnowledgeBaseBuilder


def main():
    parser = argparse.ArgumentParser(description="构建 RAG 知识库")
    parser.add_argument(
        "--embedding",
        choices=["bge", "remote"],
        default="bge",
        help="Embedding 模型来源 (默认: bge 本地免费模型)",
    )
    parser.add_argument(
        "--docs-dir",
        default="./docs",
        help="文档目录路径",
    )
    parser.add_argument(
        "--persist-dir",
        default="./chroma_db",
        help="向量数据库存储路径",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="文档块大小（字符数，默认 500）",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=50,
        help="文档块重叠大小（字符数，默认 50）",
    )

    args = parser.parse_args()

    # 构建知识库
    builder = KnowledgeBaseBuilder(embedding_provider=args.embedding)
    vector_store = builder.build_from_directory(
        docs_dir=args.docs_dir,
        persist_dir=args.persist_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    print("=" * 60)
    print("  🎉 知识库构建完成！")
    print("=" * 60)
    print()
    print("  接下来运行:")
    print("    python scripts/04_rag_qa.py          # RAG 问答演示")
    print("    python scripts/05_agent_demo.py      # Agent 智能体演示")
    print()


if __name__ == "__main__":
    main()
