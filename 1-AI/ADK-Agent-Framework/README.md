# 🤖 ADK Agent 开发框架 — 从入门到实战

> 基于 Google ADK（Agent Development Kit）的 Agent 开发教程，包含最简单的入门示例和最完整的渗透测试实战。

---

## 📖 这是什么？

**Google ADK** 是 Google 推出的 Agent 开发框架，类似于 LangChain 的 Agent 模块。它帮你管理 LLM 调用、工具绑定、会话状态、多 Agent 编排等底层细节，让你专注于定义 Agent 的行为。

本目录包含两个子项目，按难度递进：

| 子项目 | 难度 | 简介 |
|--------|------|------|
| [**simple_agent**](simple_agent/) | ⭐⭐ | 30 行代码搭建第一个 LLM Agent |
| [**penetration_agent**](penetration_agent/) | ⭐⭐⭐ | 多 Agent 渗透测试编排系统 |

---

## 🗺️ 学习路线

```
simple_agent/                          penetration_agent/
    │                                       │
    ├── LlmAgent 基础定义                    ├── 4 个子 Agent 编排
    ├── LiteLlm 模型配置                     ├── SequentialAgent 顺序执行
    ├── .env 环境变量管理                    ├── 自定义工具（scan_tool/report_tool）
    │                                       ├── 状态传递（output_key）
    │                                       ├── Session 管理 + Runner 执行
    │                                       └── 报告自动生成与保存
    └───────────────────────────────────────┘
     先理解一个 Agent 能干什么 → 再看多个 Agent 怎么协作
```

> 💡 **建议先花 30 分钟跑通 simple_agent，理解 Agent 的基本结构，再花 1 小时深入 penetration_agent。**

---

## 🚀 快速开始

### 1️⃣ 环境要求

- Python 3.10+
- LLM API Key（本示例使用腾讯云 DeepSeek 接口）

### 2️⃣ 安装依赖

```bash
cd 1-AI/ADK-Agent-Framework
pip install google-adk python-dotenv requests
```

### 3️⃣ 跑起来

```bash
# 第一步：Simple Agent 入门
cd simple_agent
# 编辑 .env 填入 API Key
python agent.py

# 第二步：渗透测试 Agent 实战
cd ../penetration_agent
# 编辑 .env 填入 API Key
python agent.py
# 输入测试目标 URL（如 http://localhost/Less-1/）
```

### 4️⃣ 配置 API

两个子项目各有一个 `.env` 文件，填入：

```bash
TENCENT_API_KEY=your-actual-api-key
```

---

## 📦 项目对比

| 对比维度 | simple_agent | penetration_agent |
|----------|-------------|-------------------|
| Agent 数量 | 1 个 | 4 个（Planner/Scanner/Reviewer/Reporter） |
| 编排方式 | 无 | SequentialAgent 顺序编排 |
| 自定义工具 | 无 | 端口扫描、SQL注入检测、报告保存 |
| 状态管理 | 无 | Session + output_key 跨 Agent 传递 |
| Web 模式 | ❌ | ✅ 支持 ADK Web 模式 |
| 代码量 | ~30 行 | ~300+ 行 |
| 适合人群 | 第一次接触 Agent | 想做完整 Agent 项目 |

---

## 🔗 参考资源

- [Google ADK 官方文档](https://google.github.io/adk-docs/)
- [LiteLLM 支持的模型列表](https://docs.litellm.ai/docs/providers)

---

> 🎯 **Agent 的核心能力是调用工具 + 自主决策。simple_agent 让你知道 Agent 长什么样，penetration_agent 让你理解 Agent 能做什么。**
>
> 下一步：同目录下的 [RAG-Knowledge-Base](../RAG-Knowledge-Base/) 会教你更复杂的工具调用和思考链。
