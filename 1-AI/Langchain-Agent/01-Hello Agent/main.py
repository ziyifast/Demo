from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.messages import HumanMessage
import os
from dotenv import load_dotenv
load_dotenv()

m = init_chat_model(
    model="deepseek-v4-flash", # 模型名称，这里可以自定义，我们用的是deepseek
    model_provider="openai", # 如果是Langchain不支持的模型，需要指定模型提供者（虽然我们用的是deepseek，但是兼容openai，所以这里用openai，就是默认采用openai的API规范）
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY")
)

system_prompt = """
# 身份
- 你是一个科幻作家，根据用户的要求创建一个太空之都。

# 指令
- 请务必以JSON格式输出，不要加任何markdown样式。

# 示例：
user: 月球的首都是什么？
assistant:
{
    "name": "月华市（Lunaria）",
    "location": "位于月球正面赤道附近的静海基地遗址之上，依托巨大的穹顶与地下网络建成",
    "vibe": "冷冽、高效、革新",
    "economy": "氦-3能源开采、量子通信枢纽、尖端生物圈农业"
}
"""

agent = create_agent(
    model=m,
    system_prompt=system_prompt
)

response = agent.invoke(
    {"messages": [HumanMessage(content="金星的首都是什么?")]},
)

print(response['messages'][-1].content)