"""
======================================================================
  知识库构建模块
======================================================================
负责文档的加载、分块、向量化和存储。

支持多种文档格式：
  - .txt 纯文本
  - .pdf PDF 文档
  - .md  Markdown 文档
  - .docx Word 文档

分块策略说明：
  - chunk_size: 每个文档块的大小（字数/字符数）
    太小 → 上下文不完整，检索准确率下降
    太大 → 噪声信息多，检索精度下降
    建议：500-1000

  - chunk_overlap: 相邻块之间的重叠字数
    防止关键信息被切断在块边界
    建议：chunk_size 的 10%-20%
======================================================================
"""

import os
from typing import List, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    UnstructuredMarkdownLoader,
    Docx2txtLoader,
)
from langchain_core.documents import Document


class DocumentLoader:
    """
    统一的文档加载器，根据文件扩展名自动选择合适的 Loader。
    """

    # 支持的文件类型及对应的 Loader
    LOADER_MAP = {
        ".txt": TextLoader,
        ".pdf": PyPDFLoader,
        ".md": UnstructuredMarkdownLoader,
        ".docx": Docx2txtLoader,
    }

    @classmethod
    def load_file(cls, file_path: str) -> List[Document]:
        """
        加载单个文档文件。

        Args:
            file_path: 文件路径

        Returns:
            加载后的 Document 对象列表（通常是一个文件 = 一个 Document）
        """
        ext = os.path.splitext(file_path)[1].lower()

        if ext not in cls.LOADER_MAP:
            raise ValueError(
                f"不支持的文件格式: {ext}\n"
                f"支持的格式: {list(cls.LOADER_MAP.keys())}"
            )

        loader_class = cls.LOADER_MAP[ext]

        # TXT 文件需要指定 UTF-8 编码
        if ext == ".txt":
            loader = loader_class(file_path, encoding="utf-8")
        else:
            loader = loader_class(file_path)

        documents = loader.load()
        print(f"  📄 已加载: {os.path.basename(file_path)} ({len(documents)} 个文档)")
        return documents

    @classmethod
    def load_directory(cls, directory: str) -> List[Document]:
        """
        加载目录下所有支持的文档文件。

        Args:
            directory: 目录路径

        Returns:
            所有加载后的 Document 对象列表
        """
        all_documents = []

        if not os.path.exists(directory):
            raise FileNotFoundError(f"目录不存在: {directory}")

        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            ext = os.path.splitext(filename)[1].lower()

            if ext in cls.LOADER_MAP:
                try:
                    docs = cls.load_file(file_path)
                    all_documents.extend(docs)
                except Exception as e:
                    print(f"  ⚠️  加载文件失败 {filename}: {e}")

        print(f"\n  ✅ 共加载 {len(all_documents)} 个文档")
        return all_documents


class TextChunker:
    """
    文档分块器 —— RAG 系统中最关键的环节之一。

    RecursiveCharacterTextSplitter 的工作原理：
    1. 先尝试用「段落分隔符」（\\n\\n）分割
    2. 如果块仍然太大，尝试用「换行符」（\\n）分割
    3. 如果还不够小，用「句号」（。）分割
    4. 最后才用「字符级别」切割
    这种递归方式能最大程度保持语义完整性。
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separators: Optional[List[str]] = None,
    ):
        """
        Args:
            chunk_size: 每个块的最大字符数
            chunk_overlap: 相邻块的重叠字符数
            separators: 分隔符优先级列表（None 则使用中文优化的默认值）
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # 中文文档优化的分隔符
        if separators is None:
            separators = [
                "\n\n",    # 段落分隔（最高优先级）
                "\n",      # 换行
                "。",      # 中文句号
                "；",      # 中文分号
                "，",      # 中文逗号
                ".",       # 英文句号
                " ",       # 空格
                "",        # 字符级分割（最低优先级）
            ]

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            length_function=len,  # 用字符数计算长度
        )

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        将文档列表分割成小块。

        Args:
            documents: 原始文档列表

        Returns:
            分块后的文档列表
        """
        chunks = self.splitter.split_documents(documents)
        print(f"  ✂️  文档分块完成: {len(documents)} 个文档 → {len(chunks)} 个块")
        print(f"      (chunk_size={self.chunk_size}, overlap={self.chunk_overlap})")
        return chunks

    def split_text(self, text: str) -> List[str]:
        """
        将纯文本分割成小块。

        Args:
            text: 原始文本

        Returns:
            分块后的文本列表
        """
        return self.splitter.split_text(text)


class KnowledgeBaseBuilder:
    """
    知识库构建器 —— 编排文档加载、分块、向量化的完整流程。

    使用示例:
        builder = KnowledgeBaseBuilder(embedding_provider="openai")
        builder.build_from_directory("./docs")
    """

    def __init__(self, embedding_provider: str = "bge"):
        """
        Args:
            embedding_provider: Embedding 模型来源
                - "bge":    使用本地 BAAI/bge-small-zh-v1.5（免费，无需 API）
                - "remote": 使用远程 Embedding API（需配置 .env）
        """
        self.embedding_provider = embedding_provider
        self.embeddings = self._create_embeddings()
        self.vector_store = None

    def _create_embeddings(self):
        """创建 Embedding 模型实例"""
        if self.embedding_provider == "bge":
            from langchain_huggingface import HuggingFaceEmbeddings

            print("  🤖 使用本地 BGE 中文 Embedding 模型（免费、无需联网）")
            return HuggingFaceEmbeddings(
                model_name="BAAI/bge-small-zh-v1.5",
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
        else:
            from langchain_openai import OpenAIEmbeddings
            from app.config import get_embedding_config

            print("  🤖 使用远程 Embedding API")
            return OpenAIEmbeddings(**get_embedding_config())

    def build_from_directory(
        self,
        docs_dir: str = "./docs",
        persist_dir: str = "./chroma_db",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ):
        """
        从文档目录构建向量知识库。

        Args:
            docs_dir: 文档目录路径
            persist_dir: 向量数据库持久化目录
            chunk_size: 每个块的最大字符数
            chunk_overlap: 相邻块的重叠字符数
        """
        from langchain_chroma import Chroma

        print("=" * 60)
        print("  📚 开始构建知识库")
        print("=" * 60)
        print()

        # 步骤 1: 加载文档
        print("  [1/3] 加载文档...")
        loader = DocumentLoader()
        documents = loader.load_directory(docs_dir)

        if not documents:
            raise ValueError(
                f"在 {docs_dir} 中没有找到可加载的文档。\n"
                f"请放入 .txt/.pdf/.md/.docx 格式的文件。"
            )

        # 步骤 2: 分块
        print(f"\n  [2/3] 文档分块...")
        chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = chunker.split_documents(documents)

        # 打印分块示例
        if chunks:
            print(f"\n  📝 第一个块预览:")
            preview = chunks[0].page_content[:200]
            print(f"  \"{preview}...\"")
            print(f"  块大小: {len(chunks[0].page_content)} 字符")

        # 步骤 3: 向量化并存储
        print(f"\n  [3/3] 向量化并存入 ChromaDB...")
        print(f"  正在为 {len(chunks)} 个块生成向量，请耐心等待...")

        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=persist_dir,
            collection_name="knowledge_base",
        )

        print(f"\n  ✅ 知识库构建完成！")
        print(f"     📁 存储路径: {os.path.abspath(persist_dir)}")
        print(f"     📊 文档块总数: {len(chunks)}")
        print(f"     🔢 向量维度: {len(self.embeddings.embed_query('test'))}")
        print()

        return self.vector_store


# ================================================================
# 便捷函数
# ================================================================

def build_knowledge_base(
    docs_dir: str = "./docs",
    persist_dir: str = "./chroma_db",
    embedding_provider: str = "bge",
    chunk_size: int = 500,
    chunk_overlap: int = 50,
):
    """
    一键构建知识库的便捷函数。
    如果你只想快速开始，直接调用这个函数即可。

    Args:
        docs_dir: 文档目录
        persist_dir: 向量库存储目录
        embedding_provider: "bge"（本地免费）或 "remote"（使用 API）
        chunk_size: 块大小
        chunk_overlap: 块重叠大小
    """
    builder = KnowledgeBaseBuilder(embedding_provider=embedding_provider)
    return builder.build_from_directory(
        docs_dir=docs_dir,
        persist_dir=persist_dir,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
