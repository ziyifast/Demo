"""
======================================================================
  RAG 引擎模块 —— 检索增强生成的核心
======================================================================
实现完整的 RAG 流程：
  1. 接收用户问题
  2. 将问题向量化
  3. 在向量数据库中检索 Top-K 最相关文档
  4. 将检索结果注入 Prompt
  5. 调用 LLM 生成带引用的回答

chain_type 说明：
  - "stuff":  将所有检索文档直接塞进 Prompt（简单直接，文档少时推荐）
  - "map_reduce": 先对每篇文档分别提问，再汇总答案（文档多时用）
  - "refine":  逐篇文档迭代优化答案（质量最高，但最慢）
  - "map_rerank": 对每篇文档打分，取最高分的答案
======================================================================
"""

from typing import List, Dict, Optional, Tuple

from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document


# ================================================================
# 自定义 Prompt 模板
# ================================================================

# 中文 RAG Prompt：引导 LLM 基于检索内容回答，避免幻觉
RAG_PROMPT_TEMPLATE = """你是一个专业的企业知识库助手。请严格基于以下提供的参考资料回答问题。

【回答规则】
1. 如果参考资料中有明确答案，直接引用相关内容回答
2. 如果参考资料中只有部分相关信息，说明已知部分并指出缺失
3. 如果参考资料中完全没有相关信息，诚实地说"根据现有资料无法回答"，不要编造
4. 回答时注明引用了哪条资料（如"根据资料[1]..."）
5. 保持回答简洁、专业、有帮助

【参考资料】
{context}

【用户问题】
{question}

【回答】"""

RAG_PROMPT = PromptTemplate(
    template=RAG_PROMPT_TEMPLATE,
    input_variables=["context", "question"],
)


class RAGEngine:
    """
    RAG 引擎 —— 检索增强生成的核心类。

    所有 LLM 配置（API Key、Base URL、模型名）统一从 .env 文件读取。
    也可通过构造参数显式传入。

    使用示例:
        # 使用 .env 配置（推荐）
        engine = RAGEngine()

        # 显式指定参数
        engine = RAGEngine(
            persist_dir="./chroma_db",
            embedding_provider="bge",
            llm_model="deepseek-chat",
            llm_api_key="sk-xxx",
            llm_api_base="https://api.deepseek.com/v1",
        )
        answer, sources = engine.query("产品保修期是多久？")
    """

    def __init__(
        self,
        persist_dir: str = "./chroma_db",
        embedding_provider: str = "bge",
        llm_model: str | None = None,        # None=从 .env 读取
        llm_api_key: str | None = None,       # None=从 .env 读取
        llm_api_base: str | None = None,      # None=从 .env 读取
        llm_temperature: float = 0.0,
        search_k: int = 3,
        chain_type: str = "stuff",
        request_timeout: int = 60,
    ):
        """
        Args:
            persist_dir:     ChromaDB 持久化目录
            embedding_provider: "bge"（本地免费）或 "remote"（远程 API）
            llm_model:       LLM 模型名。None=从 LLM_MODEL_NAME 环境变量读取
            llm_api_key:     API 密钥。None=从 LLM_API_KEY 环境变量读取
            llm_api_base:    API 地址。None=从 LLM_API_BASE 环境变量读取
            llm_temperature: 温度 (0=确定, 1=创造)
            search_k:        检索文档数
            chain_type:      LangChain 文档链类型
            request_timeout: API 请求超时秒数
        """
        import os
        self.persist_dir = persist_dir
        self.search_k = search_k

        # ---- Embedding ----
        if embedding_provider == "bge":
            self.embeddings = HuggingFaceEmbeddings(
                model_name="BAAI/bge-small-zh-v1.5",
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
        else:
            from app.config import get_embedding_config
            self.embeddings = OpenAIEmbeddings(**get_embedding_config())

        # ---- 加载向量数据库 ----
        self._load_vector_store()

        # ---- LLM ----
        # 环境变量由 get_llm_config 统一解析，这里只负责把显式参数传递进去
        # 优先级：显式参数 > .env / 环境变量
        from app.config import get_llm_config
        self.llm = ChatOpenAI(
            **get_llm_config(
                model=llm_model,
                temperature=llm_temperature,
                api_base=llm_api_base,
                api_key=llm_api_key,
                request_timeout=request_timeout,
            )
        )

        # ---- 检索器 ----
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": search_k},
        )

        # ---- RAG 链 ----
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type=chain_type,
            retriever=self.retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": RAG_PROMPT},
        )

    def _load_vector_store(self):
        """加载向量数据库"""
        import os

        if os.path.exists(self.persist_dir) and os.listdir(self.persist_dir):
            self.vector_store = Chroma(
                persist_directory=self.persist_dir,
                embedding_function=self.embeddings,
                collection_name="knowledge_base",
            )
            doc_count = self.vector_store._collection.count()
            print(f"  📂 已加载向量数据库: {doc_count} 个文档块")
        else:
            raise FileNotFoundError(
                f"向量数据库不存在: {self.persist_dir}\n"
                f"请先运行 python scripts/03_build_knowledge_base.py 构建知识库"
            )

    # ------------------------------------------------------------------
    #  查询接口
    # ------------------------------------------------------------------

    def query(self, question: str) -> Tuple[str, List[Document]]:
        """
        执行 RAG 查询。

        Returns:
            (回答文本, 引用的源文档列表)
        """
        try:
            result = self.qa_chain.invoke({"query": question})
            return result["result"], result["source_documents"]
        except Exception as e:
            raise self._translate_error(e)

    def query_with_scores(self, question: str) -> Dict:
        """
        执行 RAG 查询，附带检索的相似度分数。
        """
        docs_with_scores = self.vector_store.similarity_search_with_score(
            question, k=self.search_k,
        )
        answer, sources = self.query(question)
        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "retrieved_docs": [
                {
                    "content": doc.page_content[:200] + "...",
                    "score": float(score),
                    "metadata": doc.metadata,
                }
                for doc, score in docs_with_scores
            ],
        }

    def inspect_retrieval(self, question: str) -> None:
        """
        调试用：只检索不生成，查看检索效果。
        """
        docs_with_scores = self.vector_store.similarity_search_with_score(
            question, k=self.search_k,
        )
        print(f"\n  🔍 问题: \"{question}\"")
        print(f"  {'='*50}")
        for i, (doc, score) in enumerate(docs_with_scores):
            similarity = 1 / (1 + score)
            print(f"\n  📄 检索结果 #{i+1} - 相似度: {similarity:.4f}")
            print(f"  {'-'*40}")
            content = doc.page_content
            if len(content) > 300:
                content = content[:300] + "..."
            print(f"  {content}")
            print(f"  元数据: {doc.metadata}")
        print()

    # ------------------------------------------------------------------
    #  错误翻译 —— 把原始 HTTP 错误翻译成可操作的中文提示
    # ------------------------------------------------------------------

    @staticmethod
    def _translate_error(e: Exception) -> RuntimeError:
        error_msg = str(e)
        msg_lower = error_msg.lower()

        if "401" in error_msg or "unauthorized" in msg_lower or "authentication" in msg_lower:
            return RuntimeError(
                "🔐 LLM API 认证失败 (401) —— API Key 无效或已过期。\n"
                "  → 请检查 .env 文件中 LLM_API_KEY 是否正确\n"
                "  → 确认该 Key 在对应平台是否已激活\n"
                f"  原始错误: {error_msg}"
            )
        if "timeout" in msg_lower or "timed out" in msg_lower:
            return RuntimeError(
                "⏱️  LLM API 请求超时 —— 网络不通或服务端无响应。\n"
                "  → 检查 LLM_API_BASE 地址是否能从当前网络访问\n"
                "  → 尝试在 .env 中增大 LLM_REQUEST_TIMEOUT（如 120）\n"
                f"  原始错误: {error_msg}"
            )
        if "404" in error_msg or "not found" in msg_lower:
            return RuntimeError(
                "❌ LLM API 接口未找到 (404)。\n"
                "  请依次检查 .env 中的三项配置：\n"
                f"    1. LLM_API_BASE:  地址是否正确？末尾通常为 /v1\n"
                f"    2. LLM_MODEL_NAME: 模型名是否在该平台可用？\n"
                "    3. 你的 LLM_API_KEY 是否有权访问该模型？\n"
                "\n"
                "  各服务商示例 ——\n"
                "    OpenAI:   LLM_API_BASE=https://api.openai.com/v1              LLM_MODEL_NAME=gpt-4o\n"
                "    DeepSeek: LLM_API_BASE=https://api.deepseek.com/v1            LLM_MODEL_NAME=deepseek-chat\n"
                "    阿里百炼:  LLM_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1  LLM_MODEL_NAME=qwen-plus\n"
                "    智谱GLM:  LLM_API_BASE=https://open.bigmodel.cn/api/paas/v4   LLM_MODEL_NAME=glm-4-plus\n"
                "\n"
                f"  原始错误: {error_msg}"
            )
        if "connection" in msg_lower or "connect" in msg_lower:
            return RuntimeError(
                "🌐 无法连接 LLM API 服务器。\n"
                "  → 检查 LLM_API_BASE 地址是否正确\n"
                "  → 浏览器或 curl 访问该地址，确认网络可达\n"
                f"  原始错误: {error_msg}"
            )
        if "429" in error_msg or "rate" in msg_lower:
            return RuntimeError(
                "🚦 LLM API 请求频率过高 (429)，请稍后重试。\n"
                f"  原始错误: {error_msg}"
            )
        if "402" in error_msg or "insufficient" in msg_lower or "quota" in msg_lower:
            return RuntimeError(
                "💰 LLM API 额度不足 (402)。\n"
                "  → 请到对应平台充值或检查账户余额\n"
                f"  原始错误: {error_msg}"
            )
        return RuntimeError(
            f"❌ LLM API 调用失败。请检查 .env 中的 LLM_API_KEY / LLM_API_BASE / LLM_MODEL_NAME。\n"
            f"  原始错误: {error_msg}"
        )


def create_rag_engine(
    persist_dir: str = "./chroma_db",
    embedding_provider: str = "bge",
    llm_model: str | None = None,
    search_k: int = 3,
) -> RAGEngine:
    """创建 RAGEngine 实例的便捷函数。"""
    return RAGEngine(
        persist_dir=persist_dir,
        embedding_provider=embedding_provider,
        llm_model=llm_model,
        search_k=search_k,
    )
