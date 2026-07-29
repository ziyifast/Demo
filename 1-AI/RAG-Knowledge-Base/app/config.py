"""
======================================================================
  统一配置模块
======================================================================
集中管理 API 配置，支持任意 OpenAI 兼容接口的 LLM 服务商。

使用方式:
    # 方式 1：通过 .env 文件配置（推荐）
    #   在项目根目录创建 .env 文件，填入你的 API 信息

    # 方式 2：通过环境变量配置
    export LLM_API_KEY="your-api-key"
    export LLM_API_BASE="https://api.openai.com/v1"
    export LLM_MODEL_NAME="gpt-4o"

    # 方式 3：直接在代码中传参
    from app.config import get_llm_config
    llm = ChatOpenAI(**get_llm_config(
        api_key="your-key",
        api_base="https://your-api.com/v1",
        model="your-model",
    ))

所有配置统一通过 .env 文件或环境变量管理，不做任何写死的默认值。
======================================================================
"""

import os

# 在模块加载时自动读取 .env 文件
try:
    from dotenv import load_dotenv
    _dotenv_loaded = load_dotenv()
except ImportError:
    _dotenv_loaded = False


# ================================================================
#  环境变量读取工具
# ================================================================

def _env(key: str, default: str = "") -> str:
    """读取环境变量，去除首尾空白和引号"""
    val = os.getenv(key, default)
    if val and val.startswith('"') and val.endswith('"'):
        val = val[1:-1]
    if val and val.startswith("'") and val.endswith("'"):
        val = val[1:-1]
    return val.strip()


# ================================================================
#  LLM 配置
# ================================================================

# 如果客户端传了 model，用它；否则读环境变量；都没有就报错。
# 不再写死任何默认模型名，避免用户配了 .env 却被忽略。
def get_llm_config(
    model: str | None = None,
    temperature: float = 0.0,
    api_base: str | None = None,
    api_key: str | None = None,
    request_timeout: int | None = None,
) -> dict:
    """
    获取创建 ChatOpenAI 所需的参数字典。
    优先级：函数参数 > 环境变量 > 报错（不留硬编码默认值）。

    Args:
        model:    模型名称。None=从环境变量 LLM_MODEL_NAME 读取
        temperature: 温度 (0=确定, 1=创造)
        api_base: API 地址。None=从环境变量 LLM_API_BASE 读取
        api_key:  API 密钥。None=从环境变量 LLM_API_KEY 读取
        request_timeout: 请求超时秒数。None=从环境变量 LLM_REQUEST_TIMEOUT 读取（回退 60）

    Returns:
        可解包给 ChatOpenAI(**result) 的参数字典

    Raises:
        ValueError: 必要配置缺失时给出可操作的中文提示
    """
    # ---- 解析各项配置 ----
    resolved_key = api_key or _env("LLM_API_KEY")
    resolved_base = api_base or _env("LLM_API_BASE")
    resolved_model = model or _env("LLM_MODEL_NAME")
    timeout = request_timeout or int(_env("LLM_REQUEST_TIMEOUT") or 60)

    # ---- 缺失检查（友好报错）----
    missing = []
    if not resolved_key:
        missing.append("LLM_API_KEY  (API 密钥)")
    if not resolved_base:
        missing.append("LLM_API_BASE (API 地址)")
    if not resolved_model:
        missing.append("LLM_MODEL_NAME (模型名称)")

    if missing:
        lines = ["以下 LLM 配置项未设置，请在 .env 文件中填写：", ""]
        for m in missing:
            lines.append(f"  • {m}")
        lines.append("")
        lines.append("编辑项目根目录的 .env 文件，例如：")
        lines.append("  LLM_API_KEY=sk-your-key")
        lines.append("  LLM_API_BASE=https://api.deepseek.com/v1")
        lines.append("  LLM_MODEL_NAME=deepseek-chat")
        lines.append("")
        lines.append("也可以设置环境变量后重新运行。")
        raise ValueError("\n".join(lines))

    return {
        "model": resolved_model,
        "temperature": temperature,
        "openai_api_base": resolved_base,
        "openai_api_key": resolved_key,
        "request_timeout": timeout,
    }


def get_embedding_config(
    model: str | None = None,
    api_base: str | None = None,
    api_key: str | None = None,
) -> dict:
    """
    Embedding API 配置（仅在 embedding_provider="remote" 时使用）。
    默认推荐本地 BGE 模型，此函数为备用。

    Args:
        model:    模型名，回退 EMBEDDING_MODEL_NAME → "text-embedding-3-small"
        api_base: API 地址，回退 EMBEDDING_API_BASE → LLM_API_BASE
        api_key:  API 密钥，回退 EMBEDDING_API_KEY → LLM_API_KEY
    """
    resolved_key = api_key or _env("EMBEDDING_API_KEY") or _env("LLM_API_KEY")
    resolved_base = api_base or _env("EMBEDDING_API_BASE") or _env("LLM_API_BASE")

    return {
        "model": model or _env("EMBEDDING_MODEL_NAME") or "text-embedding-3-small",
        "openai_api_base": resolved_base,
        "openai_api_key": resolved_key,
    }


# ================================================================
#  配置诊断工具
# ================================================================

def resolve_llm_config() -> dict:
    """
    返回当前解析后的 LLM 配置摘要（隐藏密钥），供脚本启动时打印确认。
    不做缺失检查——调用 get_llm_config 才会报错。
    """
    key = _env("LLM_API_KEY")
    base = _env("LLM_API_BASE")
    model = _env("LLM_MODEL_NAME")
    timeout = _env("LLM_REQUEST_TIMEOUT") or "60"

    masked = ""
    if key:
        masked = key[:8] + "..." + key[-4:] if len(key) > 12 else "***"

    return {
        "api_base": base or "(未设置)",
        "model": model or "(未设置)",
        "api_key_masked": masked or "(未设置)",
        "timeout": timeout,
    }


def print_config_banner() -> None:
    """打印当前配置概览，帮助用户在调用 LLM 前确认配置。"""
    cfg = resolve_llm_config()
    print()
    print("=" * 65)
    print("  ⚙️  当前 LLM 配置")
    print("=" * 65)
    print(f"  API Base:  {cfg['api_base']}")
    print(f"  Model:     {cfg['model']}")
    print(f"  API Key:   {cfg['api_key_masked']}")
    print(f"  超时:      {cfg['timeout']}s")
    print("=" * 65)
