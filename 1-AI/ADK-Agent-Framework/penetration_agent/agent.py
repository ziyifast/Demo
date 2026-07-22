import asyncio
import sys
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from google.adk.agents import SequentialAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.genai.types import Content, Part
from google.adk.sessions import InMemorySessionService
from dotenv import load_dotenv

from sub_agent.planner_agent import create_planner_agent
from sub_agent.scanner_agent import create_scanner_agent
from sub_agent.reviewer_agent import create_reviewer_agent
from sub_agent.reporter_agent import create_reporter_agent

env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
load_dotenv(dotenv_path=env_path)

api_key = os.getenv("TENCENT_API_KEY")
if not api_key:
    logger.error("TENCENT_API_KEY not found in environment variables")
else:
    logger.info("TENCENT_API_KEY loaded successfully")

model = LiteLlm(
    model="tencent/deepseek-v4-flash",
    api_key=api_key,
    api_base="https://tokenhub.tencentmaas.com/v1"
)

logger.info("Model initialized successfully")

planner_agent = create_planner_agent(model)
scanner_agent = create_scanner_agent(model)
reviewer_agent = create_reviewer_agent(model)
reporter_agent = create_reporter_agent(model)

logger.info("All sub-agents created successfully")

root_agent = SequentialAgent(
    name="penetration_test_orchestrator",
    sub_agents=[
        planner_agent,
        scanner_agent,
        reviewer_agent,
        reporter_agent,
    ],
    description="渗透测试工作流编排器，按顺序执行任务规划、漏洞扫描、漏洞验证和报告输出"
)

logger.info("Agent orchestrator created successfully")

async def main():
    session_service = InMemorySessionService()
    runner = Runner(agent=root_agent, session_service=session_service, app_name="pen_test")
    
    print("=" * 60)
    print("          ADK 渗透测试 Agent")
    print("=" * 60)
    print()
    
    target = input("请输入测试目标URL（如 http://localhost/Less-1/）：").strip()
    
    while not target:
        print("错误：目标URL不能为空！")
        target = input("请输入测试目标URL：").strip()
    
    if not target.startswith(('http://', 'https://')):
        target = "http://" + target
    
    print(f"\n========== 渗透测试开始 ==========")
    print(f"测试目标: {target}")
    print()
    
    _ = await session_service.create_session(
        app_name="pen_test",
        user_id="test_user",
        session_id="test_session",
        state={}
    )
    
    user_message = Content(parts=[Part(text=f"请对 {target} 进行安全测试")])
    
    async for event in runner.run_async(
        user_id="test_user",
        session_id="test_session",
        new_message=user_message
    ):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text:
                    print(f"[{event.author}]:")
                    print(part.text)
                    print("-" * 80)
    
    print(f"\n========== 渗透测试结束 ==========")

if __name__ == "__main__":
    asyncio.run(main())