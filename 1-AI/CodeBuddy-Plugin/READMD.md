设想一个场景：我们已经在用 Skill 封装团队的代码规范，用 MCP 连接内部 API。在我们的机器上，一切跑得很顺。然后新人入职了，或者我们要把这套配置复制到第二个项目。这时发现自己开始写 wiki、截图、录屏，反复解释“先装这个、再配那个、MCP 要填这个路径”。我们配置好的不是一套“能力”，而是一堆需要被重新组装一次的零件。



**这时，Plugin 才成为必要的答案。它解决的不是“Claude 能不能做到某件事”，而是“这套能力怎么被完整地交给另一个人或另一个项目”。**



Plugin 是 Anthropic 于2025年10月为 Claude Code 提出的概念，以公开测试形式发布，同期推出的还有 Agent Skills。Plugin 的核心定位是将已有的 Skill、Hook、MCP 配置等组件打包分发，而非另起炉灶的新能力。具体可参考：https://code.claude.com/docs/en/plugins/overview。



## 1. 三个概念，一张表说清
| 机制 | 一句话定义 | 回答的问题 | 触发方式 |
|:---|:---|:---|:---|
| **MCP** | 连接外部系统的开放标准 | 模型**能触达**什么？ | 模型判断任务需要外部工具时调用 |
| **Skill** | 按需加载的程序性知识包 | 模型**知道怎么做**？ | 用户输入 `/命令`，或模型根据任务语义自动匹配 |
| **Plugin** | 打包多个能力的可分发容器 | 怎么**把能力交给团队**？ | 没有自己的触发机制；启动时注册，内部组件各自激活 |

**关键关系：**&#x50;lugin 可以打包 Skill 和 MCP 配置；Skill 可以教模型如何正确使用 MCP 暴露的工具。三者是可组合的，不是竞争的。



**什么时候用什么——反过来判断同样重要：**

* Claude 读不到某个外部系统 → MCP

* 同一段指令你说了不止一遍 → Skill

* 这套配置要交给第二个人或第二个项目 → Plugin

* 只是自己临时用一下 → 都不用，直接写 prompt 或放进 `CLAUDE.md`



## 2.  一个完整 Plugin 长什么样

这里以 CodeBuddy 为例。CodeBuddy Plugin 与 Claude Code 规范类似，只是在 Claude Code 规则上做了部分自适应适配。

Plugin 最小可以只含一个 `plugin.json`，但完整形态是多个组件并存。CodeBuddy 官方将 Plugin 定义为一个“自包含的组件目录”，组件包括 Skills、Agents、Hooks、MCP 服务器和 LSP 服务器。

以团队分发的“故障诊断工具箱”为例：

```json
incident-toolkit/
├── .codebuddy-plugin/
│   └── plugin.json           # 必需：插件清单
├── skills/
│   └── diagnose/
│       └── SKILL.md          # 编排核心：定义诊断 SOP
├── hooks/
│   └── hooks.json            # 治理：诊断结束后写审计日志
└── .mcp.json                 # 连接：声明外部 MCP 服务器地址
```

`.codebuddy-plugin/plugin.json` 是唯一必需的文件，所有其他目录（`skills/`、`hooks/`、`.mcp.json`）都位于插件根目录层级。Plugin 本身不提供触发逻辑——它在启动时发现并注册这些组件，之后每个组件按自己的规则独立工作。把三者打包在一起的价值是：安装者一次性获得连接、知识、自动化，无需分别配置。



**为什么不能只有 MCP？**&#x5047;设你把两个 MCP 服务器配好了：`query_logs`（查日志）和 `search_code`（搜代码）。模型看到的是两个孤立的工具。它可能会只用其中一个就下结论——比如只查了日志，没去代码里验证，就报告"根因是数据库超时"。



这不是假设。真实团队在把 MCP 接入诊断流程时，正是因为踩过"半截诊断"的坑，才把完整的 SOP 写成 Skill："查日志 → 找关键信息 → 扫描代码 → 定位问题"，并明确规定"分页全量拉取，禁止只查第一页就下结论"。MCP 提供"手"，Skill 提供"判断链"——工具描述本身无法表达"必须交叉验证两个来源才算完"这种团队规范。

下面的教程会让你亲手跑一遍这个过程。



> 拓展：关于 Claude Code 的兼容性。CodeBuddy 与 Claude Code 的插件结构高度一致，核心差异仅在清单文件夹的命名——CodeBuddy 使用 `.codebuddy-plugin/`，Claude Code 使用 `.claude-plugin/`。组件目录（`skills/`、`hooks/`、`.mcp.json`）的位置和格式在两者之间是共通的，为 Claude Code 编写的 Skill 通常可以直接复用。CodeBuddy 在未检测到自身清单格式时，也会读取 Claude 格式的 manifest。如果你的团队同时使用两者，只需注意清单路径这一处差异即可。





## 3. CodeBuddy Plugin 从零开发教程

> 这里我们分两步。第一步做一个只含 Skill + Hook 的最小插件，先跑通"目录即插件"的体感；第二步加入 MCP，可以体会 Skill 如何把两个孤立的 MCP 工具编排成一条完整的业务闭环。MCP 不是硬加进来的——它是这个场景走到第二步时的自然需要。



**背景：**&#x5047;设你们团队的日常痛点是线上告警来了，工程师要在日志平台和 IDE 之间来回切换，先查日志、再从日志里复制类名方法名、再去代码里找对应位置、最后拼出根因。这个过程固定但耗时间。

**诉求：**&#x4F60;想把这件事做成一个团队可安装的 Plugin。拆开看，它需要三样东西：

* 两个 MCP 工具：一个查日志（`query_logs`），一个搜代码（`search_code`）。它们是"手"，各自能干一件事，但彼此不知道对方的存在。

* 一个 Skill：规定"先查日志、再从日志里提取代码线索、再去代码里验证、最后交叉引用出报告"的顺序和完成标准。它是"脑"。

* 一个 Hook：诊断结束后自动写一条审计日志。它是"条件反射"。



### 3.1 最小可运行版本（Skill + Hook）

1. **创建目录与清单：**

```bash
mkdir -p ~/my-first-plugin/.codebuddy-plugin
mkdir -p ~/my-first-plugin/skills/diagnose
mkdir -p ~/my-first-plugin/hooks
```

* **创建plugin标识文件：** `my-first-plugin/.codebuddy-plugin/plugin.json`

```json
{
  "name": "my-first-plugin",
  "description": "A minimal diagnosis plugin to learn the basics",
  "version": "1.0.0",
  "author": { "name": "Ziyi" }
}
```

> **必填字段：`name` 是唯一标识符，也是 Skill 命令的命名空间前缀（`/my-first-plugin:diagnose`），用于避免插件之间的冲突。**



* **创建 Skill（`my-first-plugin/skills/diagnose/SKILL.md`）：**

```markdown
---
description: Diagnose an issue by following a fixed sequence
disable-model-invocation: true
---

You are a diagnostician. Follow this sequence.

## Step 1
Ask the user for a trace_id or alert description if not provided.

## Step 2
Explain what you would check next. (In the next section, this step will call real MCP tools.)
```

> `disable-model-invocation: true` 表示只允许用户手动 `/my-first-plugin:diagnose` 调用；去掉这一行则允许模型在合适时自动使用。



* **创建 Hook（`my-first-plugin/hooks/hooks.json`）：指定触发时间**

> 特定事件前后执行特定操作，例如：日志审计、敏感词过滤、环境初始化等

```json
{
  "hooks": {
    "Stop": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "echo \"[$(date -Iseconds)] diagnosis ended\" >> ${CODEBUDDY_PLUGIN_ROOT}/audit.log"
          }
        ]
      }
    ]
  }
}
```

> * `Stop` 事件在模型完成一次响应时触发，向插件根目录的 `audit.log` 追加记录。`${CODEBUDDY_PLUGIN_ROOT}` 是保留变量，指向插件安装根目录。

> * `Stop` 事件不支持 `matcher`——`matcher` 仅用于 `PreToolUse` 和 `PostToolUse` 的工具名筛选。CodeBuddy 完整支持 27+ 种 Hook 事件，常见的有 `PreToolUse`（工具调用前，可阻断）、`PostToolUse`（工具调用后）、`UserPromptSubmit`（用户提交提示后）、`SessionStart` / `SessionEnd`、`PreCompact` 等。完整事件清单与配置结构见 [Hook 参考指南](https://www.codebuddy.cn/docs/cli/hooks)。



* **本地加载：**

```bash
codebuddy --plugin-dir ./my-first-plugin
```

输入 `/my-first-plugin:diagnose trace-abc123`，你应该看到 Skill 被触发，模型按照 Step 1→2 的指令行动。此时还没有 MCP——Skill 只是在"说"，并没有"做"。

> 如果 `/插件名:技能名` 形式在当前版本中不响应，直接使用 `/技能名` 调用即可——这是官方支持的兜底路径。技能本身是否加载成功，以 `/skills` 列表中能否看到它为准。

![](https://my.feishu.cn/space/api/box/stream/download/asynccode/?code=MGIwYWFmMDE1NWY4NWZhODFkY2E5MmQ5ODkxY2JmODlfY3NGekFjQ0NyelJDNlBhSTJZeWtvNVRRN3JSOVpsZVVfVG9rZW46QjBsQ2J1WXdTb2V4RE54cGNEV2NXZzJkbmdiXzE3OTQ4NDAxMDg6MTc5NDg0MzcwOF9WNA)



### 3.2 完整版：加入 MCP，让 Skill 真正"指挥"

现在把两个 MCP 服务器接进来。它们各自暴露一个工具，彼此独立。

> MCP 服务器不放进插件目录。在真实业务中，日志查询和代码检索这类涉及业务鉴权的服务，通常由平台团队独立维护和部署。Plugin 只负责声明“去哪里连接”，不负责携带服务器代码。这样 Plugin 安装后就是“一切就绪”的——Skill 和 Hook 是纯配置，不需要任何额外环境准备。



1. **创建 MCP 配置（`my-first-plugin/.mcp.json`）：**

```json
{
  "mcpServers": {
    "incident-logs": {
      "type": "http",
      "url": "http://127.0.0.1:8000/mcp"
    },
    "incident-code": {
      "type": "http",
      "url": "http://127.0.0.1:8001/mcp"
    }
  }
}
```

CodeBuddy 的 MCP 配置使用 `mcpServers` 对象，`type` 支持 `stdio`、`sse` 和 `http`。`http` 表示通过 HTTP 流式传输与远程服务通信。两个服务器使用不同端口，避免冲突。如果希望模拟鉴权行为，可以在每个服务器配置中加入 `headers`：

```bash
"headers": {
  "Authorization": "Bearer mock-token-for-testing"
}
```

这里不定义工具参数——`.mcp.json` 只负责“连到哪个地址”。工具的参数定义（如 `trace_id` 是什么类型、是否必填）由 MCP 服务器在连接后通过 `tools/list` 返回给 CodeBuddy，客户端配置中不需要重复声明。



> 为什么 `.mcp.json` 里不需要写工具参数？
>
> 1. 因为工具发现不是由模型发起的，也不是由你手动配置的，而是客户端在连接服务器后自动完成的一步协议握手。
>
> 2. 按照 MCP 规范，声明了 `tools` 能力的服务器必须响应 `tools/list` 请求，返回当前可用工具的集合；而发送这个请求的，正是客户端——在 CodeBuddy 中，就是 CodeBuddy 框架本身。CodeBuddy 启动时，会向所有已配置的 MCP Server 发送 `tools/list`，拿到完整的工具清单；每个工具包含 `name`、`description` 和 `inputSchema` 三个字段。
>
> 3. 这份清单随后被注入模型的 system prompt——模型“看到”的不是函数指针，而是一段文本描述：“你可以调用 `search_code`，它接受 `symbol`（必填，string）……”。
>
> 因此，你只需要在 `.mcp.json` 中给出服务器地址，剩下的工具暴露和参数发现由协议自动完成。你在 `@mcp.tool()` 装饰的函数签名中写的 `symbol: str`，就是 `inputSchema` 的来源；`MCPServer` 会自动把它包装成 `tools/list` 的响应内容，CodeBuddy 连接后会自动拉取并交给模型。客户端配置中不需要、也不应该重复声明参数。
>
> MCP官方协议地址：https://modelcontextprotocol.io/specification/2026-07-28/server/tools
>
> ![](https://my.feishu.cn/space/api/box/stream/download/asynccode/?code=OTIzNzhkMGM1Yjc1ZGQ1OWI5ZTk1ZTkyM2RiZmViNmNfOFJqRkFoQWJqZHEzYmFGRVFMRFNhNGdUdGljb3JNa0FfVG9rZW46WVp5N2J6YTdqb25VZWJ4VXFUYmNvblpVbnFjXzE3OTQ4NDAxMDg6MTc5NDg0MzcwOF9WNA)







* **在本地模拟远程 MCP 服务器**

为了在本地演示“访问远程 MCP”的效果，我们启动两个本地 HTTP 服务器来模拟远程服务。

```bash
# 创建mcp server目录，并安装依赖
mkdir -p ~/mcp-server-demo/servers
cd ~/mcp-server-demo

# 配置依赖文件
vim pyproject.toml

# 依赖文件内容
[project]
name = "mcp-server-demo"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "mcp[cli]>=2.0.0",
]
```

> CodeBuddy 不会自动为插件中的自定义 Python MCP 服务器安装 Python 包。这一步需要手动完成。示例服务器为了简洁省略了鉴权；生产环境的 MCP 服务器应接入团队统一的身份验证与访问控制。





**`servers/log_server.py：`**——暴露 `query_logs` 工具，返回模拟日志：

```python
# /// script
# requires-python = ">=3.12"
# dependencies = ["mcp[cli]>=2.0.0"]
# ///
from mcp.server import MCPServer
import json

mcp = MCPServer("incident-logs")

MOCK_LOGS = {
    "trace-abc123": [
        {"level": "ERROR", "message": "NullPointerException at OrderService.calculateTotal(OrderService.java:87)"},
        {"level": "WARN", "message": "Retry attempt 1 for external payment gateway"},
        {"level": "ERROR", "message": "Failed to calculate order total for orderId=ORD-9981"},
    ]
}

@mcp.tool()
def query_logs(trace_id: str) -> str:
    """Query application logs by trace_id. Use this FIRST in any diagnosis."""
    return json.dumps(MOCK_LOGS.get(trace_id, []), indent=2)

if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="127.0.0.1", port=8000)
```

`servers/code_server.py：`——暴露 `search_code` 工具，返回模拟代码搜索结果：

```python
# /// script
# requires-python = ">=3.12"
# dependencies = ["mcp[cli]>=2.0.0"]
# ///
from mcp.server import MCPServer
import json

mcp = MCPServer("incident-code")

MOCK_CODE = {
    "OrderService.calculateTotal": [
        {
            "file": "src/main/java/com/example/OrderService.java",
            "line": 87,
            "snippet": "BigDecimal total = order.getItems().stream()\n    .map(item -> item.getPrice().multiply(item.getQuantity()))\n    .reduce(BigDecimal.ZERO, BigDecimal::add);\n// BUG: discount not applied before reduce"
        }
    ]
}

@mcp.tool()
def search_code(symbol: str) -> str:
    """Search the codebase by symbol name. Use this AFTER extracting signals from logs."""
    return json.dumps(MOCK_CODE.get(symbol, []), indent=2)

if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="127.0.0.1", port=8001)
```

`MCPServer 的 streamable-http 传输启动后，客户端通过 http://127.0.0.1:<端口>/mcp 连接。@mcp.tool()` 装饰器把普通函数暴露为模型可调用的工具，函数的 `trace_id: str` 签名就是模型看到的参数定义，docstring 就是工具描述。







**启动两个服务器（打开两个终端，分别在 `servers/` 所在目录执行）：**

```bash
uv sync        # 解析并安装 pyproject.toml 中的依赖
# 终端1
uv run servers/log_server.py

# 终端2
uv run servers/code_server.py
```



* **改造 Skill：**&#x8BA9;它真正编排两个 MCP 工具（替换`my-first-plugin/skills/diagnose/SKILL.md`的内容）：

```markdown
---
description: Diagnose a production issue by querying logs and searching code. Use when the user provides a trace_id.
disable-model-invocation: true
---

You are an incident diagnostician. Follow this exact sequence. Do not skip steps.

## Step 1: Query logs
Call the `query_logs` tool with the user's trace_id.

## Step 2: Extract code signals
From the log entries, identify FULL symbol names — class name and method name
joined by a dot (e.g., `OrderService.calculateTotal`, not `OrderService` or
`calculateTotal` alone). These full symbols are your entry points into the
codebase.

## Step 3: Search code
For each full symbol from Step 2, call `search_code` with that EXACT full symbol
string. Do not split it into separate class and method queries. Read the returned
snippet. If the snippet contains a comment starting with "BUG", flag it
explicitly.

## Step 4: Synthesize
A valid diagnosis must cite BOTH a log entry AND a code location (file:line)
returned by `search_code`. If Step 3 returns nothing, say so explicitly — do not
fabricate.

Important: a file:line string that merely appears inside a log message is NOT a
confirmed code location. Only a result returned by `search_code` counts as code
evidence. If no code search result is available, the diagnosis is incomplete —
state that clearly instead of borrowing the location mentioned in the log.
```

关键变化在 Step 1 和 Step 3：Skill 不再只是“解释你会查什么”，而是明确指令模型调用 `query_logs` 和 `search_code`，并规定 Step 4 必须同时引用日志和代码两个来源的证据。



* **重新加载并验证：**

```bash
codebuddy --plugin-dir ./my-first-plugin
```

再次输入 `/diagnose trace-abc123`。这次你应该观察到模型依次调用 `query_logs` → 从日志中提取 `OrderService.calculateTotal` → 调用 `search_code` → 最后给出一个同时引用日志行和代码行（`OrderService.java:87`）的诊断报告。

> 关于调用形式：带插件命名空间的 `/my-first-plugin:diagnose` 是插件正式安装后的标准形式；本地用 `--plugin-dir` 开发时，直接使用技能名 `/diagnose` 更稳定。两种形式触发的是同一个 Skill，不影响后续 MCP 编排的行为。



**测试效果：**

![](https://my.feishu.cn/space/api/box/stream/download/asynccode/?code=ZDM3YjM1MzAwYTYzNzU2Y2MxMjQ3MmIyY2Y2N2IyNWVfcmhMZVN5djBwUXpZUFZuaEZFUVdlQnU2aWg1Wk5OZ0pfVG9rZW46R1V3c2I1TkNhb1BoUVZ4SzN0UGNQU2c5bmJ4XzE3OTQ4NDAxMDg6MTc5NDg0MzcwOF9WNA)



### 3.3 整体目录结构

```bash
my-first-plugin/
├── .codebuddy-plugin/
│   └── plugin.json
├── skills/
│   └── diagnose/
│       └── SKILL.md          # 编排两个 MCP 工具
├── hooks/
│   └── hooks.json            # Stop → audit.log
└── .mcp.json                 # 声明两个 HTTP MCP 服务器地址
```

验证时关注三件事：

1. Skill 是否被触发：`/my-first-plugin:diagnose` 能唤起诊断流程。

2. MCP 是否被编排：模型是否按 Step 1→3 依次调用两个工具，而不是只用其中一个。

3. Hook 是否执行：插件根目录下是否出现了 `audit.log`。



Skill 的加入不是多余的——它把"先查什么、再查什么、什么才算完成"这条无法硬编码进工具 schema 的团队规范，变成了模型必须遵循的指令。



Plugin 则把"两个工具 + 一条判断链 + 一条审计规则"打包成一个可安装的单元，让团队每个人拿到的不是零件，而是一套已经组装好的诊断能力。





### 3.4 分发与安装

> Plugin 的价值在分发时才真正体现。CodeBuddy 遵循 "先添加市场，再安装插件" 的两步流程。市场（Marketplace）是一个包含 `.codebuddy-plugin/marketplace.json` 的目录或 Git 仓库，用于集中发现和分发插件。



### 基础分发：一个plugin一个git仓库

> “本示例中插件位于仓库根目录，因此 `source` 写 `"./"`。如果仓库中有多个插件分别放在不同子目录，应使用进阶分发中的 `git-subdir` 写法。”

要在自己的仓库里提供市场清单，在插件仓库根目录创建 `my-first-plugin/.codebuddy-plugin/marketplace.json`：

```json
{
  "name": "team-tools",
  "owner": { "name": "Platform Engineering" },
  "plugins": [
    {
      "name": "my-first-plugin",
      "source": "./",
      "description": "Diagnosis toolkit: logs + code, orchestrated by a skill"
    }
  ]
}
```

**团队分发：**

> **方式一：提交到 CodeBuddy 官方市场（面向全社区）。**&#x5B98;方插件市场仓库（`codebuddy-plugins-official`）接受 PR 贡献。合并后全社区可见，用户无需手动添加市场即可安装。
>
> **方式二：自建市场（公司内部/团队专有，推荐用于业务插件）。**&#x6D89;及公司内部平台或专有业务逻辑的插件，适合自建市场。在仓库根目录的 `.codebuddy-plugin/` 下放一个 `marketplace.json`，团队成员两步完成安装：

```bash
# 1. 添加市场（GitHub 仓库或内部 Git 地址）
/plugin marketplace add your-org/your-marketplace-repo

# 2. 安装插件
/plugin install my-first-plugin@team-tools
```

* 安装作用域：默认安装到用户作用域（对你所有项目生效）；交互界面中也可选择项目作用域，写入 `.codebuddy/settings.json`，协作者克隆仓库后自动获得。

* 安装 ≠ 立即生效：新安装的插件需要 `/reload-plugins` 或重启才会在当前会话中加载。加载后，内部组件各自按自身规则激活：Skill 等待匹配，Hook 监听事件，MCP 工具等待模型调用。

* 本地开发：`codebuddy --plugin-dir ./my-first-plugin` 可临时加载目录，不涉及安装。



### 进阶分发：在共享仓库中分发插件

> 常用于部门内部有多个 plugin、希望维护在同一个 git 仓库中的场景。

基础分发假设插件位于仓库根目录。但实际开发中，插件往往和一个用于本地调试的 MCP 服务器仓库、或其他配套工具放在同一个 Git 仓库的不同子目录里。例如：

```bash
Demo/                         ← Git 仓库根目录
└── 1-AI/
    └── CodeBuddy-Plugin/
        ├── my-first-plugin/  ← Plugin 子目录（只分发这个）
        └── mcp-server-demo/  ← 本地调试用，不随 Plugin 分发
```

如果 `source` 写成相对路径 `"./my-first-plugin"`，CodeBuddy 需要拥有整个 `Demo` 仓库才能解析路径，用户会下载到 `mcp-server-demo` 等无关内容。为避免这一点，可以使用 `git-subdir` 源类型：

```json
{
  "name": "team-tools",
  "owner": { "name": "Ziyi" },
  "plugins": [
    {
      "name": "my-first-plugin",
      "source": {
        "source": "git-subdir",
        "url": "https://github.com/ziyifast/Demo.git",
        "path": "1-AI/CodeBuddy-Plugin/my-first-plugin"
      },
      "description": "Diagnosis toolkit: logs + code, orchestrated by a skill"
    }
  ]
}
```

> `git-subdir` 通过稀疏检出（sparse checkout） 只获取 `path` 指定的子目录，`mcp-server-demo` 和 Demo 仓库中的其他文件不会被下载。这是 CodeBuddy 官方在 v2.77.0 插件系统升级中明确支持的多来源安装方式之一。
>
> ![](https://my.feishu.cn/space/api/box/stream/download/asynccode/?code=MDQ2ZTc3NzM0NmE4ZjJmMjI2ZTE5ZWI0Zjk5NDE4ZjJfam5mWXRONHRIVW14S3gzMzFoSm1jWWxSaG5DY3ZrcXFfVG9rZW46SXZVT2I0bUpDb1FnaWJ4UHRsU2NKMXJYbmdoXzE3OTQ4NDAxMDg6MTc5NDg0MzcwOF9WNA)
>
>

用户安装命令不变：

```bash
/plugin marketplace add ziyifast/Demo
/plugin install my-first-plugin@team-tools
```







在我们案例场景下，安装完成后，团队每个工程师遇到线上告警，都用同一条命令、走同一套诊断标准——而不是各自凭经验在多个平台之间手动切换、用不同的方式向 AI 描述问题——观测平台仍然是数据和权限的源头，但诊断的编排和代码上下文验证，被收拢到了同一条可复用的命令里，这就是Plugin的价值。

