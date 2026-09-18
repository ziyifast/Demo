import os
from dotenv import load_dotenv

load_dotenv() 

from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.tools import create_retriever_tool

# ========== 1. 初始化对话模型（LLM） ==========
m = init_chat_model(
    model="deepseek-v4-flash",
    model_provider="openai",
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
)

# ========== 2. 准备知识库文档 ==========
docs = [
    Document(page_content="公司年假政策：入职满1年享有5天年假，满3年享有10天，满5年享有15天。", metadata={"source": "员工手册"}),
    Document(page_content="公司报销政策：差旅费需在出差结束后7天内提交报销单，逾期不予受理。", metadata={"source": "财务制度"}),
    Document(page_content="公司考勤政策：上班时间为9:00-18:00，迟到超过30分钟扣半天工资。", metadata={"source": "员工手册"}),
]

# ========== 3. 文档分块 ==========
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20,
)
chunks = text_splitter.split_documents(docs)

# ========== 4. 初始化 Embedding 模型（与对话模型不同！） ==========
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

# ========== 5. 向量化并存储 ==========
vector_store = FAISS.from_documents(chunks, embeddings)

# ========== 6. 创建检索器 ==========
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# ========== 7. 创建检索工具 ==========
retriever_tool = create_retriever_tool(
    retriever,
    name="knowledge_base_search",
    description="搜索内部知识库以回答关于公司政策、产品手册、技术文档等专业问题。当用户询问需要特定专业知识的问题时，使用此工具。",
)

# ========== 8. 创建 Agent ==========
agent = create_agent(
    model=m,
    tools=[retriever_tool],
    system_prompt="你是一个智能助手。如果用户询问需要特定专业知识的问题，请使用知识库检索工具查找相关信息后再回答。",
)

# ========== 9. 调用 Agent ==========
response = agent.invoke(
    {"messages": [HumanMessage(content="我们公司的年假政策是怎样的？")]},
)

print(response["messages"][-1].content)