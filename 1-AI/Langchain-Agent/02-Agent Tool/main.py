from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.messages import HumanMessage 
from typing import Literal
import os
from pydantic import BaseModel, Field
from langchain.tools import tool
from dotenv import load_dotenv

load_dotenv()

m = init_chat_model(
    model="deepseek-v4-flash", # 模型名称，这里可以自定义，我们用的是Deepseek
    model_provider="openai", # 如果是Langchain不支持的模型，需要指定模型提供者（虽然我们用的是deepseek，但是兼容openai，所以这里用openai，就是默认采用openai的API规范）
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

# 定义一个查询天气的tool
@tool(args_schema=WeatherInput)
def get_weather(location: str, units: str = "celsius", include_forecast: bool = False) -> str:
    """Get current weather and optional forecast."""
    temp = 22 if units == "celsius" else 72
    result = f"Current weather in {location}: {temp} degrees {units[0].upper()}"
    if include_forecast:
        result += "\nNext 5 days: Sunny"
    return result

agent = create_agent(
    model=m,
    system_prompt="你是一个智能助手，你使用工具来解决用户问题。",
    tools=[get_weather]
)
# 调用agent
response = agent.invoke(
    {"messages": [HumanMessage(content="帮我查询一下北京的天气")]},
)

# 获取输出
print(response)