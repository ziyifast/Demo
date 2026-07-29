"""
======================================================================
  Agent 智能体模块 —— 让 AI 不只是回答，还能"行动"
======================================================================
Agent 的核心能力：
  1. 理解用户意图
  2. 自主决策 —— 判断需要使用哪些工具、按什么顺序
  3. 调用工具 —— 查询知识库、查订单、发邮件等
  4. 综合结果 —— 将工具返回的信息整合成最终回答

Agent 与普通 RAG 的区别：
  - RAG:     问题 → 检索 → 生成答案（固定流程）
  - Agent:   问题 → 思考 → 选工具 → 执行 → 观察结果 → 再思考 → ... → 回答
             ↑_____________________________________________↓
                     这个"思考→执行→观察"循环就是 Agent 的智能所在

本模块使用自定义 JSON Agent 循环：
  LLM 输出 JSON → 解析 → 执行工具 → 回到 LLM → 直到 final_answer
  绕过了 LangChain ReAct 的文本解析脆弱性和 function calling 兼容性问题，
  对小模型（如 llama-3.1-8b）更友好。
======================================================================
"""

import json
import re
import textwrap
from typing import List, Optional, Callable, Dict, Any

from langchain_core.tools import Tool
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

from app.rag_engine import RAGEngine


# ================================================================
# JSON Agent 系统提示词
# ================================================================

def _build_system_prompt(tool_descriptions: str, tool_names: str) -> str:
    return textwrap.dedent(f"""\
    You are an intelligent enterprise assistant. You answer user questions by calling tools when needed.

    ## Available Tools
    {tool_descriptions}

    ## Response Format (CRITICAL — you MUST output valid JSON only)

    You are a step-by-step assistant. Each turn you output EXACTLY ONE JSON object.
    Do NOT simulate multiple turns. Do NOT write what the tool would return.
    Just make ONE decision per turn:

    If you need to call a tool, output:
    {{"action": "<ToolName>", "action_input": "<input value>"}}

    If you have enough information to answer, output:
    {{"final_answer": "<your complete answer to the user>"}}

    ## Action Input Rules
    - Pass ONLY the raw value. Do NOT include variable names, "key=", or extra quotes.
    - For OrderQuery: just "ORD-001" (not "order_id = ORD-001")
    - For KnowledgeBase: a natural language question or keywords

    ## Rules
    1. ALWAYS search KnowledgeBase before answering product/policy/warranty questions
    2. If tools return no relevant results, honestly tell the user — NEVER make up information
    3. Answer in the SAME language as the user's question
    4. Cite sources when using KnowledgeBase results (e.g. "According to the product manual...")
    5. When the user asks about multiple things (e.g. order + refund policy), call tools one at a time
    6. Output ONLY ONE JSON object per turn — the system will call tools and return results to you
    """)


# ================================================================
# RAGAgent 类
# ================================================================

class RAGAgent:
    """
    基于 RAG 的智能 Agent（自定义 JSON 循环）。

    特点：
    - 默认配备知识库查询工具
    - 支持扩展自定义工具（订单查询、邮件发送等）
    - 使用 JSON 格式进行 LLM ↔ 执行器通信，比 ReAct 文本解析可靠

    使用示例:
        agent = RAGAgent(persist_dir="./chroma_db", embedding_provider="bge")
        agent.add_tool(name="OrderQuery", func=query_order, description="...")
        response = agent.run("产品保修期是多久？")
        print(response['output'])
    """

    def __init__(
        self,
        persist_dir: str = "./chroma_db",
        embedding_provider: str = "bge",
        llm_model: str | None = None,
        llm_temperature: float = 0.0,
        search_k: int = 3,
        verbose: bool = True,
        request_timeout: int = 60,
        max_iterations: int = 6,
    ):
        self.verbose = verbose
        self.search_k = search_k
        self.max_iterations = max_iterations

        # 初始化 LLM
        from app.config import get_llm_config
        self.llm = ChatOpenAI(
            **get_llm_config(model=llm_model, temperature=llm_temperature, request_timeout=request_timeout)
        )

        # 初始化 RAG 引擎
        self.rag_engine = RAGEngine(
            persist_dir=persist_dir,
            embedding_provider=embedding_provider,
            llm_model=llm_model,
            llm_temperature=llm_temperature,
            search_k=search_k,
            request_timeout=request_timeout,
        )

        # 工具管理
        self._extra_tools: List[Tool] = []
        self._all_tools: List[Tool] = []
        self._tool_map: Dict[str, Callable] = {}
        self._rebuild_tools()

    # ================================================================
    # 工具管理
    # ================================================================

    def _create_rag_tool(self) -> Tool:
        """创建 RAG 知识库查询工具"""
        def rag_query(query_text: str) -> str:
            try:
                if self.verbose:
                    self.rag_engine.inspect_retrieval(query_text)

                answer, sources = self.rag_engine.query(query_text)

                result = f"【知识库查询结果】\n{answer}\n\n【参考来源】"
                for i, doc in enumerate(sources[:3]):
                    source_info = doc.metadata.get("source", "未知来源")
                    preview = doc.page_content[:100].replace("\n", " ")
                    result += f"\n[{i+1}] {source_info}: {preview}..."
                return result
            except Exception as e:
                return f"知识库查询出错: {str(e)}"

        return Tool(
            name="KnowledgeBase",
            func=rag_query,
            description=(
                "查询企业知识库。用于回答产品、政策、流程、价格、保修、"
                "技术支持等问题。输入：自然语言问句或关键词。"
            ),
        )

    def add_tool(self, name: str, func: Callable, description: str) -> None:
        """添加自定义工具"""
        tool = Tool(name=name, func=func, description=description)
        self._extra_tools.append(tool)
        self._rebuild_tools()

    def _rebuild_tools(self):
        """重新构建工具列表和映射"""
        self._all_tools = [self._create_rag_tool()] + self._extra_tools
        self._tool_map = {tool.name: tool.func for tool in self._all_tools}

    def _get_tool_descriptions(self) -> str:
        """生成工具描述文本（用于 system prompt）"""
        lines = []
        for tool in self._all_tools:
            lines.append(f"- **{tool.name}**: {tool.description}")
        return "\n".join(lines)

    def list_tools(self) -> List[Dict[str, str]]:
        """列出当前所有可用工具"""
        return [{"name": t.name, "description": t.description} for t in self._all_tools]

    # ================================================================
    # Agent 核心循环
    # ================================================================

    def run(self, query: str) -> Dict[str, Any]:
        """
        运行 Agent 循环处理用户查询。

        循环逻辑：
          1. 发送 system prompt + 对话历史 + 当前 query 给 LLM
          2. LLM 返回 JSON: {"action": "...", "action_input": "..."} 或 {"final_answer": "..."}
          3. 如果是 action → 执行工具 → 将结果加入对话 → 回到步骤 1
          4. 如果是 final_answer → 结束循环，返回答案
        """
        tool_descriptions = self._get_tool_descriptions()
        tool_names = [t.name for t in self._all_tools]
        system_prompt = _build_system_prompt(tool_descriptions, ", ".join(tool_names))

        messages = [SystemMessage(content=system_prompt)]

        # 首次用户输入
        messages.append(HumanMessage(content=query))

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"  🤖 Agent 开始处理: {query}")
            print(f"{'='*60}")

        for iteration in range(1, self.max_iterations + 1):
            # 调用 LLM
            response = self.llm.invoke(messages)
            raw_output = response.content.strip()

            if self.verbose:
                print(f"\n  📍 第 {iteration} 轮 LLM 输出:")
                print(f"  {'-'*40}")
                print(textwrap.indent(raw_output, "  | "))
                print(f"  {'-'*40}")

            # 解析 JSON
            parsed = self._parse_json_output(raw_output)

            if parsed is None:
                # JSON 解析失败：让 LLM 重试
                messages.append(HumanMessage(content=(
                    "ERROR: Your last response was not valid. You must output "
                    "EXACTLY ONE JSON object per turn. Do NOT output multiple "
                    "JSON objects, and do NOT simulate what the tool will return. "
                    "Just decide ONE action:\n"
                    '{"action": "ToolName", "action_input": "value"}\n'
                    'or {"final_answer": "your answer"}'
                )))
                continue

            if "final_answer" in parsed:
                if self.verbose:
                    print(f"\n  ✅ Agent 完成 (共 {iteration} 轮)")
                return {"input": query, "output": parsed["final_answer"]}

            if "action" in parsed:
                action_name = parsed["action"]
                action_input = parsed.get("action_input", "")

                # 执行工具
                if action_name not in self._tool_map:
                    tool_result = (
                        f"错误: 工具 '{action_name}' 不存在。"
                        f"可用工具: {', '.join(tool_names)}"
                    )
                else:
                    try:
                        if self.verbose:
                            print(f"  🔧 调用工具: {action_name}({action_input!r})")
                        tool_result = self._tool_map[action_name](action_input)
                        if self.verbose:
                            preview = tool_result[:200].replace("\n", " ")
                            print(f"  📋 工具返回: {preview}...")
                    except Exception as e:
                        tool_result = f"工具执行出错: {str(e)}"

                # 将工具调用和结果加入对话
                messages.append(HumanMessage(content=(
                    f"Tool call result:\n"
                    f"Tool: {action_name}\n"
                    f"Input: {action_input}\n"
                    f"Result:\n{tool_result}\n\n"
                    f"Based on this result, either call another tool or give your final_answer."
                )))
                continue

            # 既不是 action 也不是 final_answer
            messages.append(HumanMessage(content=(
                "Invalid JSON format. Your response must have either 'action' or 'final_answer'. "
                'Example: {"action": "KnowledgeBase", "action_input": "your question"} '
                'or {"final_answer": "your complete answer"}'
            )))

        # 超过最大迭代次数
        return {
            "input": query,
            "output": "抱歉，处理超时。请尝试简化您的问题，或稍后再试。"
        }

    def _parse_json_output(self, raw: str) -> Optional[Dict[str, Any]]:
        """
        解析 LLM 输出的 JSON。只提取第一个完整的 JSON 对象。

        处理 LLM 常见问题：
        - 一次输出多个 JSON 对象（模拟对话）→ 只取第一个
        - Markdown 代码块包裹
        - 前后的多余文本
        - 单引号替代双引号
        """
        # 去掉 Markdown 代码块
        code_block_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', raw, re.DOTALL)
        if code_block_match:
            raw = code_block_match.group(1).strip()

        # 用括号平衡算法提取第一个完整 JSON 对象
        # (避免贪婪匹配吞掉多个对象)
        start = raw.find('{')
        if start == -1:
            if self.verbose:
                print(f"  ⚠️ 未找到 JSON 对象: {raw[:200]}")
            return None

        depth = 0
        in_string = False
        escape_next = False
        for i in range(start, len(raw)):
            ch = raw[i]
            if escape_next:
                escape_next = False
                continue
            if ch == '\\':
                escape_next = True
                continue
            if ch == '"' and not escape_next:
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    first_object = raw[start:i + 1]
                    return self._try_parse_json(first_object)

        # 没找到匹配的右括号，尝试用贪婪匹配回退
        brace_match = re.search(r'\{.*\}', raw, re.DOTALL)
        if brace_match:
            return self._try_parse_json(brace_match.group(0))

        if self.verbose:
            print(f"  ⚠️ JSON 解析失败: {raw[:200]}")
        return None

    @staticmethod
    def _try_parse_json(json_str: str) -> Optional[Dict[str, Any]]:
        """尝试解析 JSON 字符串，自动修复常见问题"""
        for attempt in range(3):
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                if attempt == 0:
                    # 单引号 → 双引号
                    json_str = json_str.replace("'", '"')
                elif attempt == 1:
                    # 修复字符串值中的裸换行符
                    json_str = json_str.replace('\n', '\\n')
            except Exception:
                break
        return None


# ================================================================
# 示例：自定义工具
# ================================================================

def create_order_query_tool():
    """示例自定义工具：订单查询"""
    mock_orders = {
        "ORD-001": {
            "status": "已完成",
            "product": "NeoCompute Standard 包月",
            "amount": "¥1,500",
            "date": "2025-01-15",
        },
        "ORD-002": {
            "status": "配送中",
            "product": "NeoServer R2 服务器",
            "amount": "¥28,000",
            "date": "2025-01-20",
        },
        "ORD-003": {
            "status": "待支付",
            "product": "NeoStorage 标准存储 100GB",
            "amount": "¥120",
            "date": "2025-01-22",
        },
    }

    def _extract_order_id(raw_input: str) -> str:
        """从杂讯输入中提取订单 ID"""
        match = re.search(r'(ORD-\d+)', raw_input, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        return raw_input.strip().upper()

    def query_order(order_id: str) -> str:
        order_id = _extract_order_id(order_id)
        if order_id in mock_orders:
            order = mock_orders[order_id]
            return (
                f"订单 {order_id}:\n"
                f"  状态: {order['status']}\n"
                f"  产品: {order['product']}\n"
                f"  金额: {order['amount']}\n"
                f"  日期: {order['date']}"
            )
        else:
            available = ", ".join(mock_orders.keys())
            return f"未找到订单 {order_id}。可用的订单号: {available}"

    return Tool(
        name="OrderQuery",
        func=query_order,
        description=(
            "查询订单状态和详情。"
            "输入：订单ID（如 ORD-001）"
        ),
    )


def create_email_tool():
    """示例自定义工具：模拟邮件发送"""
    def send_email(recipient_info: str) -> str:
        return (
            f"✅ 邮件已模拟发送成功！\n"
            f"  收件信息: {recipient_info}\n"
            f"  (实际项目中会调用真实邮件服务)"
        )

    return Tool(
        name="SendEmail",
        func=send_email,
        description=(
            "向指定收件人发送邮件。"
            "输入格式: '收件人: xxx@example.com, 主题: xxx, 内容: xxx'"
        ),
    )


# ================================================================
# 便捷函数
# ================================================================

def create_rag_agent(
    persist_dir: str = "./chroma_db",
    embedding_provider: str = "bge",
    llm_model: str | None = None,
    search_k: int = 3,
    extra_tools: Optional[List[Tool]] = None,
) -> RAGAgent:
    """创建 RAG Agent 的便捷函数"""
    agent = RAGAgent(
        persist_dir=persist_dir,
        embedding_provider=embedding_provider,
        llm_model=llm_model,
        search_k=search_k,
    )
    if extra_tools:
        for tool in extra_tools:
            agent.add_tool(name=tool.name, func=tool.func, description=tool.description)
    return agent
