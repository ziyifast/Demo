import json
import os
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

# ========== 1. 创建 FastAPI 应用 ==========
app = FastAPI()

# 允许前端跨域访问（开发时用 *，生产环境改成具体域名）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== 2. 初始化 Agent（复用前面章节） ==========
m = init_chat_model(
    model="deepseek-v4-flash",
    model_provider="openai",
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
)

# 仅单进程有效；多 worker 部署需换成 PostgresSaver/RedisSaver 等持久化实现
memory = InMemorySaver()

agent = create_agent(
    model=m,
    system_prompt="你是一个智能助手。",
    checkpointer=memory,
)

# ========== 3. 定义请求体 ==========
class ChatRequest(BaseModel):
    query: str
    session_id: str = "default"  # 具体应该由前端传入，此处纯为了方便。仅演示用；生产建议必填，且由前端生成 UUID

# ========== 4. 普通接口 ==========
@app.post("/chat")
async def chat(req: ChatRequest):
    config = {"configurable": {"thread_id": req.session_id}}
    result = await agent.ainvoke(
        {"messages": [HumanMessage(content=req.query)]},
        config,
    )
    return {"answer": result["messages"][-1].content}

# ========== 5. 流式接口（SSE） ==========
@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    config = {"configurable": {"thread_id": req.session_id}}

    async def gen():
        async for chunk, _ in agent.astream(
            {"messages": [HumanMessage(content=req.query)]},
            stream_mode="messages",
            config=config,
        ):
            if chunk.content:
                yield f"data: {json.dumps({'content': chunk.content}, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"X-Accel-Buffering": "no"},  # 关键：防止 Nginx 缓冲
    )