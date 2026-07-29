"""
======================================================================
  第五步：Agent 智能体 —— 从"回答问题"到"自主行动"
======================================================================
本脚本展示 Agent 如何：
  1. 自动判断应该使用哪个工具
  2. 调用知识库查询相关信息
  3. 组合多个工具完成复杂任务
  4. 自主决策"思考→行动→观察→再思考"的循环

前置条件：已运行 03_build_knowledge_base.py 构建了知识库

运行方式：
  # 基础 Agent 演示（只有知识库工具）
  python scripts/05_agent_demo.py

  # Agent + 订单查询 + 邮件工具
  python scripts/05_agent_demo.py --with-extras

  # 交互式 Agent
  python scripts/05_agent_demo.py --interactive

======================================================================
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.agent import RAGAgent, create_order_query_tool, create_email_tool


def print_separator(title: str):
    print()
    print("=" * 65)
    print(f"  {title}")
    print("=" * 65)
    print()


def check_prerequisites():
    """检查前置条件"""
    if not os.path.exists("./chroma_db") or not os.listdir("./chroma_db"):
        print("  ⚠️  未找到知识库！")
        print()
        print("  请先运行:")
        print("    python scripts/03_build_knowledge_base.py")
        print()
        sys.exit(1)


def run_demo(with_extras: bool = False):
    """运行 Agent 演示"""
    print_separator("Agent 智能体演示")

    # 创建 Agent（LLM 配置从 .env 读取）
    print("  🤖 正在初始化 Agent...")
    agent = RAGAgent(
        persist_dir="./chroma_db",
        embedding_provider="bge",
        search_k=3,
        verbose=True,  # 显示 Agent 的思考过程
    )

    # 可选：添加额外工具
    if with_extras:
        agent.add_tool(
            name="OrderQuery",
            func=create_order_query_tool().func,
            description=create_order_query_tool().description,
        )
        agent.add_tool(
            name="SendEmail",
            func=create_email_tool().func,
            description=create_email_tool().description,
        )

    # 列出可用工具
    print()
    print("  🛠️  可用工具:")
    for tool_info in agent.list_tools():
        print(f"     - {tool_info['name']}: {tool_info['description'][:60]}...")
    print()

    # ================================================================
    # 场景 1：简单知识库查询
    # ================================================================
    print_separator("场景 1: 简单知识库查询")

    question1 = "产品保修期是多久？"
    print(f"  🙋 用户: {question1}")
    print()

    response1 = agent.run(question1)
    print()
    print(f"  🤖 Agent 最终回答:\n  {response1['output']}")

    # ================================================================
    # 场景 2：需要搜索的问题
    # ================================================================
    print_separator("场景 2: 复杂信息查询")

    question2 = "我想了解退款政策和数据安全措施"
    print(f"  🙋 用户: {question2}")
    print()

    response2 = agent.run(question2)
    print()
    print(f"  🤖 Agent 最终回答:\n  {response2['output']}")

    # ================================================================
    # 场景 3：多工具协作（仅在 --with-extras 时可用）
    # ================================================================
    if with_extras:
        print_separator("场景 3: 多工具协作")

        question3 = "帮我查一下订单 ORD-001 的状态，然后告诉我这个订单里的产品保修政策"
        print(f"  🙋 用户: {question3}")
        print()

        response3 = agent.run(question3)
        print()
        print(f"  🤖 Agent 最终回答:\n  {response3['output']}")

    print_separator("演示完成")
    print()
    print("  💡 关键观察：Agent 的思考过程（Thought → Action → Observation）")
    print("     这就是 Agent 区别于普通 RAG 的核心——它能自主决策！")
    print()


def run_interactive():
    """交互式 Agent 模式"""
    print_separator("Agent 交互模式")

    # 创建带所有工具的 Agent
    agent = RAGAgent(
        persist_dir="./chroma_db",
        embedding_provider="bge",
        verbose=True,
    )

    # 添加额外工具
    agent.add_tool(
        name="OrderQuery",
        func=create_order_query_tool().func,
        description=create_order_query_tool().description,
    )
    agent.add_tool(
        name="SendEmail",
        func=create_email_tool().func,
        description=create_email_tool().description,
    )

    print()
    print("  🛠️  可用工具:")
    for tool_info in agent.list_tools():
        print(f"     - {tool_info['name']}")
    print()
    print("  💡 试试这些指令:")
    print("     - '产品保修期是多久？'")
    print("     - '查一下订单 ORD-001，然后告诉我对应的退款政策'")
    print("     - '告诉我安全合规信息，然后给客户发一封说明邮件'")
    print()
    print("  输入 'quit' 或 'exit' 退出")
    print()

    while True:
        try:
            user_input = input("\n  🙋 你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  再见！")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            print("  再见！")
            break

        print()
        response = agent.run(user_input)
        print(f"\n  🤖 Agent: {response['output']}")


def main():
    parser = argparse.ArgumentParser(description="RAG Agent 智能体")
    parser.add_argument(
        "--with-extras",
        action="store_true",
        help="添加订单查询和邮件发送工具",
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="交互式 Agent 模式",
    )
    parser.add_argument(
        "--persist-dir",
        default="./chroma_db",
        help="向量数据库路径",
    )

    args = parser.parse_args()

    # 前置检查
    check_prerequisites()

    if args.interactive:
        run_interactive()
    else:
        run_demo(with_extras=args.with_extras)


if __name__ == "__main__":
    main()
