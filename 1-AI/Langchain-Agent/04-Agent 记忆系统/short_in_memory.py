from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
import os
from dotenv import load_dotenv

load_dotenv()

# 1. 初始化模型
m = init_chat_model(
    model="deepseek-v4-flash",
    model_provider="openai",
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY")
)

# 2. 创建短期记忆 Checkpointer
checkpointer = InMemorySaver()

# 3. 创建 Agent，注入短期记忆
agent = create_agent(
    model=m,
    system_prompt="你是一个智能助手。",
    checkpointer=checkpointer
)

# 4. 指定 thread_id，同一个 thread_id 视为同一次会话
config = {"configurable": {"thread_id": "thread_1"}}

# 5. 第一次调用：告诉 AI 信息
response1 = agent.invoke(
    {"messages": [HumanMessage(content="你好，我叫龙哥，我喜欢足球，司职左边锋")]},
    config
)
print("第一次调用：")
print(response1["messages"][-1].content)

# 6. 第二次调用：验证短期记忆是否生效
response2 = agent.invoke(
    {"messages": [HumanMessage(content="我在足球中踢什么位置？")]},
    config
)
print("\n第二次调用：")
print(response2["messages"][-1].content)