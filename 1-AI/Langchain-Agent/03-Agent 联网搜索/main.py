from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.messages import HumanMessage 
from typing import Literal
from pydantic import BaseModel, Field
from langchain.tools import tool
# 使用tavily作为web搜索工具
from langchain_tavily import TavilySearch
import os
from dotenv import load_dotenv

load_dotenv()

m = init_chat_model(
    model="deepseek-v4-flash", # 模型名称，这里可以自定义
    model_provider="openai", # 如果是Langchain不支持的模型，需要指定模型提供者（虽然我们用的是deepseek，但是deepseek兼容openai，所以这里用openai，就是默认采用openai的API规范）
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY")
)
# 例如一个查询天气的tool
class WeatherInput(BaseModel):
    """查询天气的输入参数."""
    location: str = Field(description="City name or coordinates")
    units: Literal["celsius", "fahrenheit"] = Field(
        default="celsius",
        description="Temperature unit preference"
    )
    include_forecast: bool = Field(
        default=False,
        description="Include 5-day forecast"
    )

# 使用tavily作为web搜索工具（TavilySearch会默认从环境变量中读取TAVILY_API_KEY密钥）
tavily = TavilySearch(
    max_results=5,
    topic="general"
)

@tool
def web_search(query: str):
    """Search the web for information"""
    return tavily.invoke(query)

# Agent回答内容引用的网页信息
class Reference(BaseModel):
    title: str = Field(description="The title of the web page cited in the answer")
    url: str = Field(description="The url of the web page cited in the answer")

# Agent的回答内容
class AnswerInfo (BaseModel):
    answer: str = Field(description="The final answer for user")
    reference: list[Reference] = Field(description="The web pages cited in the answer")

agent = create_agent(
    model=m,
    system_prompt="你是一个智能助手，你使用工具来解决用户问题。",
    tools=[web_search],
    response_format=AnswerInfo
)
# 调用agent
response = agent.invoke(
    {"messages": [HumanMessage(content="帮我看看26年中国男篮对战黎巴嫩的开赛时间")]},
)

# 获取输出
print(response)