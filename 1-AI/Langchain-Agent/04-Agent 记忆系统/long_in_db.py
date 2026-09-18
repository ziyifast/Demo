import sqlite3
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain.tools import tool, ToolRuntime
from langgraph.store.sqlite import SqliteStore
from dotenv import load_dotenv
import os

load_dotenv()

m = init_chat_model(
    model="deepseek-v4-flash",
    model_provider="openai",
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY")
)

# 长期记忆 Store
conn = sqlite3.connect("long_memory.db", check_same_thread=False, isolation_level=None)
store = SqliteStore(conn)
store.setup()

@tool
def save_user_preference(key: str, value: str, runtime: ToolRuntime) -> str:
    """保存用户的长期偏好。

    Args:
        key: 偏好名称，如 "favorite_animal"
        value: 偏好值，如 "猫猫"
    """
    user_id = runtime.config.get("configurable", {}).get("user_id", "default")
    namespace = (user_id, "preferences")
    store.put(namespace, key, {"value": value})
    return f"已记住：{key} = {value}"

@tool
def get_user_preference(key: str, runtime: ToolRuntime) -> str:
    """读取用户之前保存的长期偏好。

    Args:
        key: 要查询的偏好名称，如 "favorite_animal"
    """
    user_id = runtime.config.get("configurable", {}).get("user_id", "default")
    namespace = (user_id, "preferences")
    item = store.get(namespace, key)
    if item:
        return f"{key} = {item.value['value']}"
    return f"没有找到关于 {key} 的记忆"

agent = create_agent(
    model=m,
    system_prompt=(
        "你是一个有记忆的智能助手。"
        "当用户告诉你偏好时，用 save_user_preference 保存。"
        "当用户问你还记得什么时，用 get_user_preference 查询。"
    ),
    tools=[save_user_preference, get_user_preference],
    store=store,
)

# 会话 1：保存偏好
config1 = {"configurable": {"thread_id": "session_001", "user_id": "user_longge"}}
r1 = agent.invoke(
    {"messages": [HumanMessage(content="你好，我叫龙哥，我最喜欢的动物是猫猫。请帮我记住。")]},
    config1,
)
print("会话1：", r1["messages"][-1].content)

# 会话 2：新 thread_id，验证长期记忆
config2 = {"configurable": {"thread_id": "session_002", "user_id": "user_longge"}}
r2 = agent.invoke(
    {"messages": [HumanMessage(content="你还记得我最喜欢什么动物吗？")]},
    config2,
)
print("会话2：", r2["messages"][-1].content)