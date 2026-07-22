# ADK 渗透测试 Agent

基于 Google ADK (Agent Development Kit) 框架构建的企业级渗透测试 Agent，包含任务规划、漏洞扫描、漏洞验证和报告输出四个子 Agent，展示了如何使用 ADK 框架开发企业级多 Agent 系统。

## 🚀 快速开始

### 环境要求

- Python 3.10+
- ADK 框架
- 腾讯云 Maas API Key

### 安装依赖

```bash
pip install google-adk python-dotenv requests
```

### 配置环境变量

创建 `.env` 文件：

```env
TENCENT_API_KEY=your_api_key_here
```

### 运行方式

#### 方式一：命令行模式

```bash
cd 1-AI/ADK-Agent-Framework/penetration_agent
python agent.py
```

然后按照提示输入测试目标 URL：

```
============================================================
          ADK 渗透测试 Agent
============================================================

请输入测试目标URL（如 http://localhost/Less-1/）：http://localhost/Less-1/
```

#### 方式二：ADK Web 模式

```bash
cd /path/to/ADK
adk web --port 8000
```

在浏览器中访问 `http://localhost:8000`，选择 `hack_agent`，输入测试指令：

```
请对 http://localhost/Less-1/ 进行安全测试
```

> **注意**：ADK Web 框架要求主入口变量名必须为 `root_agent`，否则会报 `No root_agent found` 错误。

## 📁 项目结构

```
hack_agent/
├── agent.py                    # 主入口文件，编排所有子 Agent
├── .env                        # 环境变量配置
├── .gitignore                  # Git 忽略配置
├── __init__.py                 # Python 包初始化
├── reports/                    # 渗透测试报告输出目录
│   └── pen_test_report_*.md    # 自动生成的测试报告
├── sub_agent/                  # 子 Agent 目录
│   ├── __init__.py
│   ├── planner_agent.py        # 任务规划 Agent
│   ├── scanner_agent.py        # 漏洞扫描 Agent
│   ├── reviewer_agent.py       # 漏洞验证 Agent
│   └── reporter_agent.py       # 报告输出 Agent
└── tools/                      # 工具函数目录
    ├── __init__.py
    ├── scan_tool.py            # 漏洞扫描工具
    └── report_tool.py          # 报告保存工具
```

## 🏗️ 架构设计

### Agent 工作流

```
用户输入 → Planner Agent → Scanner Agent → Reviewer Agent → Reporter Agent → 生成报告
     │            │              │                │                 │
     │            ▼              ▼                ▼                 ▼
     │       制定测试       执行漏洞扫描      验证漏洞真实性     生成渗透测试报告
     │       计划              │                │                 │
     │                         ▼                ▼                 ▼
     │                    使用扫描工具       排除误报         保存报告到本地
     │                    (scan_tool.py)     评估风险
     └─────────────────────────────────────────────────────────────────→ 返回结果
```

### 各组件职责

| 组件 | 类型 | 职责 | 关键特性 |
|------|------|------|----------|
| **agent.py** | 编排器 | 初始化模型、创建子Agent、编排工作流 | SequentialAgent 顺序执行 |
| **planner_agent** | LlmAgent | 分析目标，制定渗透测试计划 | 提取 target_url，输出 test_plan |
| **scanner_agent** | LlmAgent | 执行漏洞扫描，检测安全漏洞 | 调用 run_comprehensive_scan 工具 |
| **reviewer_agent** | LlmAgent | 验证漏洞真实性，排除误报 | 评估可利用性，输出 validation_results |
| **reporter_agent** | LlmAgent | 生成完整渗透测试报告 | 调用 save_report 工具保存报告 |
| **scan_tool.py** | 工具 | 端口扫描、SQL注入检测、XSS检测、目录枚举 | 真实网络请求，模拟安全扫描 |
| **report_tool.py** | 工具 | 将报告保存到本地文件 | 自动创建 reports 目录 |

## 📝 ADK 企业级 Agent 开发指南

### 步骤一：环境准备

1. **安装 ADK 框架**：
   ```bash
   pip install google-adk
   ```

2. **配置模型服务**：
   ```python
   from google.adk.models.lite_llm import LiteLlm
   
   model = LiteLlm(
       model="tencent/deepseek-v4-flash",
       api_key=os.getenv("TENCENT_API_KEY"),
       api_base="https://tokenhub.tencentmaas.com/v1"
   )
   ```

### 步骤二：定义工具函数

创建 `tools/` 目录，编写可被 Agent 调用的工具函数：

```python
# tools/scan_tool.py
def run_comprehensive_scan(target: str) -> str:
    """执行综合安全扫描"""
    # 实现扫描逻辑
    return scan_result
```

**工具开发要点**：
- 使用标准 Python 函数定义
- 添加清晰的文档字符串（docstring）
- 返回值应为字符串，便于 LLM 处理

### 步骤三：创建子 Agent

每个子 Agent 使用 `LlmAgent` 定义：

```python
# sub_agent/planner_agent.py
from google.adk.agents import LlmAgent

def create_planner_agent(model):
    return LlmAgent(
        name="planner_agent",
        model=model,
        description="分析测试目标，制定渗透测试计划",
        instruction="""你是一名专业的渗透测试规划师...""",
        output_key="test_plan",
        include_contents="default"
    )
```

**LlmAgent 参数说明**：

| 参数 | 说明 |
|------|------|
| `name` | Agent 名称，用于标识和日志 |
| `model` | 使用的 LLM 模型实例 |
| `description` | Agent 功能描述，用于编排器理解 |
| `instruction` | 详细指令，指导 Agent 行为 |
| `tools` | 可调用的工具列表（可选） |
| `output_key` | 输出结果存储的 session 状态 key |
| `include_contents` | 是否包含会话历史（`default` 或 `none`） |

### 步骤四：编排工作流

使用 `SequentialAgent` 编排多个子 Agent：

```python
# agent.py
from google.adk.agents import SequentialAgent

# 变量名必须为 root_agent，ADK Web 框架通过此名称查找入口 Agent
root_agent = SequentialAgent(
    name="penetration_test_orchestrator",
    sub_agents=[
        planner_agent,
        scanner_agent,
        reviewer_agent,
        reporter_agent,
    ],
    description="渗透测试工作流编排器"
)
```

> **重要**：ADK Web 框架会按顺序搜索 `hack_agent.agent.root_agent`、`hack_agent.root_agent` 和 `hack_agent/root_agent.yaml`，因此主入口变量名**必须**为 `root_agent`。

### 步骤五：实现状态传递

通过 `output_key` 和 `{variable}` 模板实现 Agent 间数据传递：

```python
# planner_agent 输出
output_key="test_plan"

# scanner_agent 使用
instruction="""测试计划：{test_plan}"""
```

### 步骤六：处理运行模式

确保 Agent 同时支持命令行和 ADK Web 模式：

```python
# agent.py
async def main():
    session_service = InMemorySessionService()
    runner = Runner(agent=agent, session_service=session_service, app_name="pen_test")
    
    # 命令行模式：创建 session 并运行
    _ = await session_service.create_session(
        app_name="pen_test",
        user_id="test_user",
        session_id="test_session",
        state={}
    )
    
    async for event in runner.run_async(
        user_id="test_user",
        session_id="test_session",
        new_message=user_message
    ):
        # 处理事件输出
        pass

if __name__ == "__main__":
    asyncio.run(main())
```

### 步骤七：导入路径处理

确保 ADK Web 模式下能正确导入模块：

```python
# agent.py 顶部
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# sub_agent/*.py 顶部
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
```

### 步骤八：.env 文件路径处理

确保 ADK Web 模式下能正确加载 `.env` 文件：

```python
# agent.py - 使用绝对路径加载 .env
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
load_dotenv(dotenv_path=env_path)
```

> **注意**：当 ADK Web 从父目录加载 `hack_agent` 模块时，`load_dotenv()` 默认从当前工作目录查找 `.env`，而不是从 `hack_agent/` 目录。必须使用 `dotenv_path` 参数指定绝对路径。

## 🧪 测试验证

### 命令行测试

```bash
echo "http://localhost/Less-1/" | python agent.py
```

预期输出：
- Planner Agent 输出测试计划
- Scanner Agent 输出扫描结果（发现 SQL 注入等漏洞）
- Reviewer Agent 输出验证结果
- Reporter Agent 输出最终报告并保存到 `reports/` 目录

### ADK Web 测试

1. 启动服务：`adk web --port 8000`
2. 在浏览器中选择 `hack_agent`
3. 输入：`请对 http://localhost/Less-1/ 进行安全测试`
4. 观察页面上各 Agent 的输出

### 测试结果示例

```
[planner_agent]:
## 渗透测试计划
目标分析：http://localhost/Less-1/
target_url: http://localhost/Less-1/
...

[scanner_agent]:
## 扫描结果汇总
发现 SQL 注入漏洞（CVSS: 8.5）
...

[reviewer_agent]:
## 漏洞验证报告
确认 SQL 注入漏洞存在，可利用性高
...

[reporter_agent]:
报告已保存到 reports/pen_test_report_20260721_170509.md
```

## 📊 报告输出

报告自动保存到 `reports/` 目录，包含：

1. **执行摘要** — 测试概述、总体风险评级
2. **测试目标** — 目标系统描述、测试范围
3. **测试方法** — 使用的工具和技术
4. **漏洞详情** — 按严重程度排序的漏洞列表
5. **安全建议** — 立即修复、短期修复、长期改进
6. **结论** — 测试总结和后续行动建议

## 🔧 扩展建议

### 集成真实扫描工具

将模拟扫描替换为真实工具：

- **端口扫描**：使用 `nmap` 或 `python-nmap`
- **Web 漏洞扫描**：集成 OWASP ZAP API
- **目录枚举**：使用 `gobuster` 或 `dirsearch`

### 添加权限控制

实现用户认证和授权机制：

- 基于角色的访问控制（RBAC）
- API 密钥管理
- 会话权限验证

### 持久化存储

替换内存存储为数据库：

- 使用 SQLite 或 PostgreSQL 存储会话
- 实现报告版本管理
- 添加数据备份和恢复功能

### Web 界面增强

开发专业的前端界面：

- 可视化展示测试流程
- 实时进度条
- 漏洞统计图表
- 报告在线预览和下载

## ⚠️ 安全注意事项

1. **仅限授权测试**：仅对授权目标进行测试，遵守法律法规
2. **API 密钥保护**：使用环境变量管理敏感信息，不要硬编码
3. **报告加密存储**：测试报告可能包含敏感信息，建议加密存储
4. **最小权限原则**：限制 Agent 执行权限，避免越权操作

## 📖 ADK 框架核心概念

### Agent

ADK 中的核心组件，封装了特定任务的执行逻辑。

### LlmAgent

基于大语言模型的 Agent，通过 instruction 指导其行为。

### SequentialAgent

顺序执行多个子 Agent 的编排器，适用于线性工作流。

### Session

管理 Agent 执行过程中的状态和上下文。

### Runner

负责执行 Agent 的运行器，处理会话管理和事件流。

### Tool

可被 Agent 调用的外部函数或服务。

## 📚 参考文档

- [ADK 官方文档](https://developers.google.com/adk)
- [ADK GitHub 仓库](https://github.com/google/adk)
- [LiteLLM 文档](https://docs.litellm.ai/)

## 📄 许可证

本项目仅供学习和研究使用。