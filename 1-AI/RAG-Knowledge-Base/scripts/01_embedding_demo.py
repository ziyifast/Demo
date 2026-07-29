"""
======================================================================
  第一步：理解 Embedding —— 让计算机"理解"语义
======================================================================
本脚本演示 Embedding 的核心概念：
  1. 文字如何变成数字向量
  2. 语义相近的文字，向量距离也更近
  3. 通过余弦相似度来衡量语义相似性

运行方式：
  python scripts/01_embedding_demo.py
======================================================================
"""

import numpy as np
from sentence_transformers import SentenceTransformer


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    计算两个向量之间的余弦相似度。
    取值范围：[-1, 1]，越接近 1 表示越相似。

    公式：cos(θ) = (A·B) / (|A| × |B|)
    """
    dot_product = np.dot(vec1, vec2)
    norm_a = np.linalg.norm(vec1)
    norm_b = np.linalg.norm(vec2)
    return dot_product / (norm_a * norm_b)


def main():
    print("=" * 65)
    print("  第一步：加载 Embedding 模型")
    print("=" * 65)
    print()

    # 使用 BGE 中文小模型（无需联网，首次运行会自动下载）
    # 它是目前中文语义理解方面表现最好的开源模型之一
    model_name = "BAAI/bge-small-zh-v1.5"
    print(f"  正在加载模型: {model_name}")
    print("  (首次运行需要下载模型文件，约 100MB，请耐心等待...)")
    print()

    model = SentenceTransformer(model_name)
    embedding_dim = model.get_sentence_embedding_dimension()
    print(f"  ✅ 模型加载成功！向量维度: {embedding_dim}")
    print()
    print("  💡 向量维度 = {0} 意味着每个句子被表示为 {0} 个数字。".format(embedding_dim))
    print("     维度越高 → 能捕捉的语义信息越丰富，但计算成本也越大。")
    print()

    # ================================================================
    print("=" * 65)
    print("  第二步：观察 Embedding 向量的样子")
    print("=" * 65)
    print()

    # 将一句话转成向量
    sentence = "苹果是一种很好吃的水果"
    vector = model.encode(sentence)

    print(f"  原句: \"{sentence}\"")
    print()
    print(f"  向量（前10维）: {vector[:10].tolist()}")
    print(f"  向量（第10-20维）: {vector[10:20].tolist()}")
    print(f"  ... (共{len(vector)}维)")
    print()
    print("  💡 每个数字代表了这句话在某个语义维度上的特征值。")
    print("     比如：某个维度可能代表「食物相关程度」，另一个可能代表「植物相关程度」")
    print()

    # ================================================================
    print("=" * 65)
    print("  第三步：语义相近 = 向量距离近")
    print("=" * 65)
    print()

    # 准备一组测试句子
    test_sentences = [
        "苹果是一种很好吃的水果",
        "香蕉也是一种很好吃的水果",
        "我喜欢吃新鲜的水果",
        "苹果公司发布了一款新手机",
        "iPhone是苹果公司最著名的产品",
        "今天天气真不错",
        "我喜欢在晴天出门散步",
    ]

    print("  正在将以下句子转为向量...")
    for i, s in enumerate(test_sentences):
        print(f"    [{i+1}] \"{s}\"")
    print()

    # 批量转成向量
    embeddings = model.encode(test_sentences)

    # 以第一句话为基准，计算它与其他所有句子的相似度
    base_idx = 0
    base_sentence = test_sentences[base_idx]

    print(f"  以「{base_sentence}」为基准，计算其他句子的相似度：")
    print()
    print(f"  {'序号':<6}{'相似度':<10}{'句子'}")
    print(f"  {'-'*60}")

    similarities = []
    for i, (s, emb) in enumerate(zip(test_sentences, embeddings)):
        sim = cosine_similarity(embeddings[base_idx], emb)
        similarities.append((i, sim, s))

    # 按相似度从高到低排序
    similarities.sort(key=lambda x: x[1], reverse=True)

    for rank, (i, sim, s) in enumerate(similarities):
        bar = "█" * int(sim * 30)
        print(f"  {rank+1:<6}{sim:.4f}   {s}  {bar}")

    print()
    print("  🔍 观察发现：")
    print("     - 「苹果水果」「香蕉水果」「喜欢水果」→ 相似度高（都是水果话题）")
    print("     - 「苹果公司」「iPhone」 → 相似度较低（虽然都含'苹果'，但意思是科技公司）")
    print("     - 「天气」「散步」 → 相似度最低（完全不相关的话题）")
    print()
    print("  💡 这就是 Embedding 的核心威力：它不是匹配字面，而是理解语义！")
    print()

    # ================================================================
    print("=" * 65)
    print("  第四步：语义搜索演示（最简单的「检索」）")
    print("=" * 65)
    print()

    # 模拟一个文档库
    documents = [
        "NeoCloud 平台的云服务器保修期为 3 年",
        "我们的产品支持 7 天内无条件退款",
        "NeoAI 平台的 A100 GPU 推理服务每小时 28 元",
        "公司年会在 12 月 25 日举行，地点在三楼宴会厅",
        "数据加密使用 AES-256 算法，传输采用 TLS 1.3",
        "新用户注册即送 200 元体验金",
    ]

    doc_embeddings = model.encode(documents)

    # 用户查询
    query = "产品保修多久"
    query_embedding = model.encode(query)

    print(f"  查询: \"{query}\"")
    print()
    print(f"  检索结果（按语义相关性排序）:")
    print(f"  {'排名':<6}{'相似度':<10}{'文档内容'}")
    print(f"  {'-'*60}")

    doc_similarities = []
    for i, (doc, emb) in enumerate(zip(documents, doc_embeddings)):
        sim = cosine_similarity(query_embedding, emb)
        doc_similarities.append((i, sim, doc))

    doc_similarities.sort(key=lambda x: x[1], reverse=True)

    for rank, (i, sim, doc) in enumerate(doc_similarities):
        bar = "█" * int(sim * 30)
        print(f"  {rank+1:<6}{sim:.4f}   {doc}  {bar}")

    print()
    print("  🎯 最相关的文档精确命中了「保修期 3 年」！")
    print("     虽然查询词是「保修多久」，文档写的是「保修期 3 年」——")
    print("     字面不同，但语义相同。这就是 Embedding 检索的优势。")
    print()

    print("=" * 65)
    print("  🎉 第一部分完成！你已经理解了 Embedding 的核心原理。")
    print("  接下来，让我们把向量存入「向量数据库」——见 02_vector_db_demo.py")
    print("=" * 65)


if __name__ == "__main__":
    main()
