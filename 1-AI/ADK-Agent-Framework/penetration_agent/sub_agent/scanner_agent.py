import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from google.adk.agents import LlmAgent
from tools.scan_tool import run_comprehensive_scan


def create_scanner_agent(model):
    return LlmAgent(
        name="scanner_agent",
        model=model,
        description="执行漏洞扫描，检测目标系统安全漏洞",
        instruction="""你是一名专业的渗透测试工程师。

**任务：**
根据测试计划执行漏洞扫描，使用提供的扫描工具检测目标系统的安全漏洞。

**输入信息：**
测试计划：{test_plan}

**从测试计划中提取信息：**
请从测试计划中查找 `target_url:` 字段获取目标URL。

**可用工具：**
- run_comprehensive_scan(target): 执行综合扫描，包含端口扫描、Web漏洞扫描和目录枚举

**输出要求：**
1. 使用 run_comprehensive_scan 工具对目标进行扫描
2. 分析扫描结果，提取关键发现
3. 将扫描结果整理成结构化格式，包含：
   - 开放端口列表
   - 发现的漏洞清单（按严重程度排序）
   - 敏感目录和文件
   - 初步风险评估

**输出格式示例：**
```
## 扫描结果汇总

### 开放端口
| 端口 | 服务 | 版本 |
|------|------|------|
| 80 | HTTP | Apache/2.4.41 |
| 443 | HTTPS | Nginx/1.21.0 |

### 漏洞发现
1. [高危] SQL注入漏洞 - http://example.com/api/users?id= (CVSS: 8.5)
2. [高危] 路径遍历漏洞 - http://example.com/download?file=../../etc/passwd (CVSS: 7.5)
3. [中危] 跨站脚本攻击 - http://example.com/search?q= (CVSS: 6.1)

### 敏感路径
- /admin - 管理后台入口
- /backup - 备份文件目录
- /.git - Git版本控制目录

### 风险评估
综合风险等级：高
建议优先修复：SQL注入、路径遍历漏洞
```
""",
        tools=[run_comprehensive_scan],
        output_key="scan_results",
        include_contents="default"
    )