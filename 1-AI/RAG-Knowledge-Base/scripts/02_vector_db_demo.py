"""
======================================================================
  第二步：向量数据库实战 —— ChromaDB 基本操作
======================================================================
本脚本演示：
  1. ChromaDB 的创建和基本 CRUD 操作
  2. 向量的添加、查询、删除
  3. 语义搜索（相似度检索）
  4. 持久化存储

运行方式：
  python scripts/02_vector_db_demo.py
======================================================================
"""

import os
import chromadb
from sentence_transformers import SentenceTransformer


def print_separator(title: str):
    """打印格式化的章节标题"""
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)
    print()


def main():
    # ================================================================
    print_separator("1. 初始化 ChromaDB 客户端")
    # ================================================================

    # 创建持久化存储目录
    persist_dir = "./chroma_demo_db"

    # ChromaDB 客户端：类似于 MySQL 的"连接"
    client = chromadb.PersistentClient(path=persist_dir)

    print(f"  ✅ ChromaDB 客户端已创建")
    print(f"  📁 数据存储路径: {os.path.abspath(persist_dir)}")
    print()
    print("  💡 ChromaDB 提供两种模式：")
    print("     - PersistentClient: 数据持久化到磁盘（推荐）")
    print("     - Client(): 仅内存模式，程序结束数据消失")

    # ================================================================
    print_separator("2. 创建/获取 Collection（集合）")

    # Collection 类似于关系数据库中的"表"
    # get_or_create: 如果已存在则获取，否则新建
    collection_name = "demo_knowledge_base"

    # 先删除旧的（如果存在），保证每次运行是干净的演示
    try:
        client.delete_collection(name=collection_name)
        print(f"  🗑️  已删除旧 Collection: {collection_name}")
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name,
        metadata={"description": "演示用知识库", "version": "1.0"},
    )

    print(f"  ✅ Collection 已创建: {collection_name}")
    print(f"  📋 元数据: {collection.metadata}")
    print()
    print("  💡 Collection 类比：")
    print("     - ChromaDB 客户端    ≈ MySQL 连接")
    print("     - Collection         ≈ MySQL 中的一张表")
    print("     - 每个文档           ≈ 表中的一行")
    print("     - 向量（embedding）   ≈ 该行的向量索引列")

    # ================================================================
    print_separator("3. 添加文档到 Collection")
    # ================================================================

    # 准备 Embedding 模型
    model = SentenceTransformer("BAAI/bge-small-zh-v1.5")

    # 准备要添加的文档
    documents = [
        "NeoCloud 云服务器提供 3 年标准保修服务",
        "平台支持 7 天内无条件全额退款",
        "NeoAI Max 推理服务使用 A100 GPU，每小时 28 元",
        "所有数据传输采用 TLS 1.3 加密",
        "新用户注册即赠送 200 元体验金",
        "技术支持热线：400-888-NEO-CLOUD，7×24 小时服务",
    ]

    # 生成文档 ID（实际项目中可用文档 hash 或 UUID）
    doc_ids = [f"doc_{i+1}" for i in range(len(documents))]

    # 生成向量（Embedding）
    print("  正在生成向量...")
    embeddings = model.encode(documents).tolist()

    # 添加文档：文档内容 + 向量 + ID + 元数据
    collection.add(
        documents=documents,          # 原始文本
        embeddings=embeddings,        # 向量（OpenAI/Cortex 会自动生成）
        ids=doc_ids,                  # 唯一 ID
        metadatas=[                   # 额外元数据
            {"category": "保修", "source": "product_manual"},
            {"category": "退款", "source": "product_manual"},
            {"category": "定价", "source": "product_manual"},
            {"category": "安全", "source": "product_manual"},
            {"category": "优惠", "source": "product_manual"},
            {"category": "支持", "source": "product_manual"},
        ],
    )

    print(f"  ✅ 已添加 {len(documents)} 个文档")
    print(f"  向量维度: {len(embeddings[0])}")
    print()
    for i, doc in enumerate(documents):
        print(f"    [{doc_ids[i]}] {doc}")

    # ================================================================
    print_separator("4. 语义搜索（最核心的操作）")
    # ================================================================

    # 搜索 1：精确语义搜索
    query1 = "产品保修多久"
    query_emb1 = model.encode(query1).tolist()

    results1 = collection.query(
        query_embeddings=[query_emb1],
        n_results=3,  # 返回 Top-3 最相关的结果
    )

    print(f"  🔍 查询: \"{query1}\"")
    print()
    for i in range(len(results1["ids"][0])):
        doc_id = results1["ids"][0][i]
        doc_text = results1["documents"][0][i]
        distance = results1["distances"][0][i]
        similarity = 1 - distance  # Chroma 默认用余弦距离
        print(f"    #{i+1} [相似度: {similarity:.4f}] {doc_id}: {doc_text}")

    print()
    print("  💡 虽然查询词是「保修多久」，但结果精确命中了「3年保修」——")
    print("     这就是向量数据库的核心能力：按语义相似度检索，而非字面匹配。")
    print()

    # 搜索 2：另一个查询
    query2 = "怎么联系客服"
    query_emb2 = model.encode(query2).tolist()

    results2 = collection.query(
        query_embeddings=[query_emb2],
        n_results=2,
    )

    print(f"  🔍 查询: \"{query2}\"")
    print()
    for i in range(len(results2["ids"][0])):
        doc_text = results2["documents"][0][i]
        similarity = 1 - results2["distances"][0][i]
        print(f"    #{i+1} [相似度: {similarity:.4f}] {doc_text}")

    print()
    print("  🎯 「怎么联系客服」→ 命中了「技术支持热线」！")
    print("     查询词和文档内容字面完全不同，但语义高度相关。")

    # ================================================================
    print_separator("5. 带过滤条件的查询")
    # ================================================================

    # ChromaDB 支持元数据过滤
    query3 = "平台有什么优惠政策"
    query_emb3 = model.encode(query3).tolist()

    # 只在「优惠」和「退款」类别中搜索
    results3 = collection.query(
        query_embeddings=[query_emb3],
        n_results=3,
        where={"category": {"$in": ["优惠", "退款"]}},  # 元数据过滤
    )

    print(f"  🔍 查询: \"{query3}\" (仅搜索「优惠」和「退款」类别)")
    print()
    for i in range(len(results3["ids"][0])):
        doc_text = results3["documents"][0][i]
        similarity = 1 - results3["distances"][0][i]
        print(f"    #{i+1} [相似度: {similarity:.4f}] {doc_text}")

    print()
    print("  💡 元数据过滤让我们可以在语义搜索的基础上加上精确条件，")
    print("     比如只搜索某个类别的文档、某个时间范围、某个数据源等。")

    # ================================================================
    print_separator("6. 文档管理：更新与删除")
    # ================================================================

    # 更新文档
    print("  📝 更新文档 doc_1...")
    updated_text = "NeoCloud 云服务器已升级为 5 年超长保修服务"
    collection.update(
        ids=["doc_1"],
        documents=[updated_text],
        embeddings=[model.encode(updated_text).tolist()],
    )

    # 验证更新
    verify = collection.get(ids=["doc_1"])
    print(f"  更新后: {verify['documents'][0]}")
    print()

    # 删除文档
    print("  🗑️  删除文档 doc_4...")
    collection.delete(ids=["doc_4"])

    # 查看剩余文档
    remaining = collection.get()
    print(f"  剩余文档数: {len(remaining['ids'])}")
    print(f"  剩余文档 IDs: {remaining['ids']}")

    # ================================================================
    print_separator("7. 查看 Collection 信息")
    # ================================================================

    count = collection.count()
    print(f"  📊 Collection 名称: {collection_name}")
    print(f"  📊 文档总数: {count}")
    print(f"  📊 元数据: {collection.metadata}")

    print()
    print("=" * 60)
    print("  🎉 第二部分完成！你已经掌握了 ChromaDB 的基本操作。")
    print("  接下来：用真实文档构建知识库 —— 见 03_build_knowledge_base.py")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
