import sqlite3
import sys
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from dotenv import load_dotenv
import os

load_dotenv()

# 1. 初始化模型
m = init_chat_model(
    model="deepseek-v4-flash",
    model_provider="openai",
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY")
)

# 2. 创建 SQLite 连接和 Checkpointer
conn = sqlite3.connect("short_memory.db", check_same_thread=False)
checkpointer = SqliteSaver(conn)

# 3. 创建 Agent
agent = create_agent(
    model=m,
    system_prompt="你是一个智能助手。",
    checkpointer=checkpointer
)

# 4. 固定 thread_id
config = {"configurable": {"thread_id": "thread_1"}}

# 5. 根据命令行参数决定模式
if len(sys.argv) < 2:
    print("用法: python main.py save|load")
    sys.exit(1)

mode = sys.argv[1]

if mode == "save":
    # 第一次运行：告诉 AI 信息
    r = agent.invoke(
        {"messages": [HumanMessage(content="你好，我叫龙哥，我喜欢足球，司职左边锋")]},
        config
    )
    print("【保存模式】AI 回复：", r["messages"][-1].content)

elif mode == "load":
    # 第二次运行：询问 AI，验证是否还记得
    r = agent.invoke(
        {"messages": [HumanMessage(content="我在足球中踢什么位置？")]},
        config
    )
    print("【读取模式】AI 回复：", r["messages"][-1].content)

else:
    print("未知模式，请使用 save 或 load")