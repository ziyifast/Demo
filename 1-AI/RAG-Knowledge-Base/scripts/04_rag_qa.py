"""
======================================================================
  第四步：RAG 问答 —— 检索增强生成的完整流程
======================================================================
本脚本展示 RAG 在实际问答中的表现：

  用户问题 → 向量化 → 检索 Top-K 文档 → 注入 Prompt → LLM 生成答案
                                              ↑
                                    这是 RAG 与传统 LLM 的核心区别

前置条件：已运行 03_build_knowledge_base.py 构建了知识库

所有 LLM 配置统一通过项目根目录 .env 文件管理。

运行方式：
  # 使用 .env 中配置的 LLM（推荐）
  python scripts/04_rag_qa.py --interactive

  # 查看当前配置是否正确（不调用 LLM）
  python scripts/04_rag_qa.py --check-config

  # 演示模式（预设问题）
  python scripts/04_rag_qa.py

  # 通过命令行覆盖 .env 中的设置
  python scripts/04_rag_qa.py --llm-model deepseek-chat --llm-api-key sk-xxx --llm-api-base https://api.deepseek.com/v1

  # 自定义检索数量
  python scripts/04_rag_qa.py --k 5
======================================================================
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import print_config_banner, resolve_llm_config


# 预设的演示问题
DEMO_QUESTIONS = [
    "产品保修期是多久？",
    "如果我想退款，有什么政策？",
    "NeoAI Max 的价格是多少？",
    "数据安全方面有哪些措施？",
    "如何联系技术支持？",
    "新用户有什么优惠？",
]


def print_separator(title: str):
    print()
    print("=" * 65)
    print(f"  {title}")
    print("=" * 65)
    print()


def check_prerequisites():
    if not os.path.exists("./chroma_db") or not os.listdir("./chroma_db"):
        print("  ⚠️  未找到知识库！")
        print()
        print("  请先运行:")
        print("    python scripts/03_build_knowledge_base.py")
        print()
        sys.exit(1)


def run_demo(engine):
    """运行演示问答"""
    print_separator("RAG 问答演示")

    for i, question in enumerate(DEMO_QUESTIONS):
        print(f"\n  {'─'*60}")
        print(f"  🙋 问题 {i+1}: {question}")
        print(f"  {'─'*60}")

        # 检索
        print(f"\n  📊 检索结果（Top-{engine.search_k} 最相关文档）:")
        engine.inspect_retrieval(question)

        # RAG
        print(f"\n  🔍 正在调用 LLM 生成回答...")
        try:
            answer, sources = engine.query(question)
            print(f"\n  🤖 RAG 回答:")
            print(f"  {'─'*55}")
            for line in answer.strip().split("\n"):
                print(f"  {line}")
            print(f"  {'─'*55}")
            print(f"\n  📚 引用了 {len(sources)} 篇源文档")
        except Exception as e:
            print(f"\n  ❌ {e}")

    print_separator("演示完成")


def run_interactive(engine):
    """交互式问答"""
    print_separator("RAG 交互式问答")
    print()
    print("  输入问题后回车，系统将：")
    print("  1. 在知识库中检索最相关文档")
    print("  2. 基于检索到的文档生成回答")
    print()
    print("  输入 'quit' / 'exit' 退出")
    print("  输入 'debug <问题>' 查看详细检索信息（不调用 LLM）")
    print()

    while True:
        try:
            user_input = input("  🙋 你的问题: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  再见！")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("  再见！")
            break
        if user_input.lower().startswith("debug "):
            engine.inspect_retrieval(user_input[6:].strip())
            continue

        print(f"\n  🔍 正在检索...")
        try:
            answer, sources = engine.query(user_input)
            print(f"\n  🤖 回答:")
            print(f"  {'─'*55}")
            for line in answer.strip().split("\n"):
                print(f"  {line}")
            print(f"  {'─'*55}")
            print(f"  📚 引用 {len(sources)} 篇源文档")
            print()
        except Exception as e:
            print(f"\n  ❌ {e}\n")


def main():
    parser = argparse.ArgumentParser(
        description="RAG 问答系统（LLM 配置统一通过 .env 文件管理）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 检查配置
  python scripts/04_rag_qa.py --check-config

  # 交互式问答
  python scripts/04_rag_qa.py --interactive

  # 演示模式
  python scripts/04_rag_qa.py

  # 临时覆盖 .env 中的配置
  python scripts/04_rag_qa.py --llm-api-key sk-xxx --llm-api-base https://api.deepseek.com/v1 --llm-model deepseek-chat
        """,
    )
    parser.add_argument("--check-config", action="store_true", help="仅打印当前配置并退出（不调用 LLM）")
    parser.add_argument("--interactive", "-i", action="store_true", help="交互式问答模式")
    parser.add_argument("--embedding", choices=["bge", "remote"], default="bge", help="Embedding 来源 (默认 bge 本地免费)")
    parser.add_argument("--llm-model", default=None, help="LLM 模型名（覆盖 .env 中的 LLM_MODEL_NAME）")
    parser.add_argument("--llm-api-key", default=None, help="LLM API Key（覆盖 .env 中的 LLM_API_KEY）")
    parser.add_argument("--llm-api-base", default=None, help="LLM API Base URL（覆盖 .env 中的 LLM_API_BASE）")
    parser.add_argument("--request-timeout", type=int, default=60, help="API 请求超时秒数（默认 60）")
    parser.add_argument("--k", type=int, default=3, help="检索文档数（默认 3）")
    parser.add_argument("--persist-dir", default="./chroma_db", help="向量数据库路径")

    args = parser.parse_args()

    # --check-config：仅打印配置并退出
    if args.check_config:
        print_config_banner()
        return

    # 命令行参数覆盖环境变量（临时生效）
    if args.llm_api_key:
        os.environ["LLM_API_KEY"] = args.llm_api_key
    if args.llm_api_base:
        os.environ["LLM_API_BASE"] = args.llm_api_base
    if args.llm_model:
        os.environ["LLM_MODEL_NAME"] = args.llm_model

    # 打印当前配置，让用户确认
    print_config_banner()

    check_prerequisites()

    from app.rag_engine import RAGEngine

    engine = RAGEngine(
        persist_dir=args.persist_dir,
        embedding_provider=args.embedding,
        # 明确传 None 让 get_llm_config 从 .env 读取，不用硬编码默认值
        llm_model=args.llm_model,        # None or user-provided
        llm_api_key=args.llm_api_key,
        llm_api_base=args.llm_api_base,
        search_k=args.k,
        request_timeout=args.request_timeout,
    )

    if args.interactive:
        run_interactive(engine)
    else:
        run_demo(engine)


if __name__ == "__main__":
    main()
