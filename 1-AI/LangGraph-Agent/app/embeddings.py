from langchain_openai import OpenAIEmbeddings
from app.config import EMBEDDING_BASE_URL, EMBEDDING_MODEL

embeddings = OpenAIEmbeddings(
    model=EMBEDDING_MODEL,
    base_url=EMBEDDING_BASE_URL,
    api_key="local",
    check_embedding_ctx_length=False,   # ← 关键：让 Ollama 收到原始文本
)