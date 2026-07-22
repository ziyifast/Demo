from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from dotenv import load_dotenv
import os

# 加载 .env 文件
load_dotenv()

# 创建 LiteLlm 实例，并把 api_key 和 api_base 传给它
my_model = LiteLlm(
    model="tencent/deepseek-v4-flash",
    api_key=os.getenv("TENCENT_API_KEY"),          # 显式传递
    api_base="https://tokenhub.tencentmaas.com/v1" # 基础 URL，去掉 /chat/completions
)

# 定义 Agent，只传入 model
root_agent = LlmAgent(
    model=my_model,
    name="my_openai_agent",
    instruction="You are a helpful assistant powered by OpenAI.",
    # 不再写 api_key 和 api_base，因为已在 model 中配置
)