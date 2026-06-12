"""
生活管家 AI-Agent 配置文件
所有配置集中管理，支持 .env 文件 + 环境变量覆盖
"""
import os
from pathlib import Path

# ==================== 路径配置 ====================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
VECTOR_DB_DIR = DATA_DIR / "vectordb"
MEMORY_DB_PATH = DATA_DIR / "memory.db"

# ==================== 加载 .env 文件 ====================
def _load_env():
    """从 .env 文件加载配置（不覆盖已有环境变量）"""
    env_file = BASE_DIR / ".env"
    if not env_file.exists():
        return
    with open(env_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            if key and key not in os.environ:
                os.environ[key] = value

_load_env()

# ==================== LLM 配置 ====================
# 支持 dashscope (通义千问) / zhipuai / openai 兼容接口
LLM_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
LLM_BASE_URL = os.getenv(
    "LLM_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible_mode/v1",
)
LLM_MODEL = os.getenv("LLM_MODEL", "qwen-plus")
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "25"))  # 秒

# ==================== 服务器配置 ====================
SERVER_HOST = os.getenv("SERVER_HOST", "0.0.0.0")
# Render 会通过 PORT 环境变量分配端口，优先使用
SERVER_PORT = int(os.getenv("PORT", os.getenv("SERVER_PORT", "8000")))
SERVER_WORKERS = int(os.getenv("SERVER_WORKERS", "1"))

# ==================== 并发控制 ====================
MAX_CONCURRENT_LLM = int(os.getenv("MAX_CONCURRENT_LLM", "10"))  # 最大并发LLM请求
MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "20"))  # 上下文消息上限

# ==================== RAG 配置 ====================
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "shibing624/text2vec-base-chinese",
)
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "3"))  # 检索返回文档数

# ==================== 系统提示词 ====================
SYSTEM_PROMPT = """你是一款面向普通用户的智能生活管家AI-Agent，专业、贴心、简洁、实用。
你拥有真实的工具能力，可以直接调用工具获取实时数据和执行操作。

## 系统环境信息
- 当前操作系统：Windows
- 用户主目录：{user_home}
- 用户桌面：{user_desktop}
- 用户下载目录：{user_downloads}

## 核心运行规则
1. 严格基于上下文、记忆、工具返回的真实数据回答，禁止编造未知信息；
2. 需要实时数据（天气、时间等）时，必须调用对应工具获取，不要自己编造；
3. 需要计算时，调用 calculator 工具，不要自己心算；
4. 收到模糊指令时，主动追问关键信息；
5. 用户要求写文件到"桌面"时，使用上面的桌面路径；要求写文件但未指定路径时，默认写到桌面。

## 可用工具
你可以通过 function calling 调用以下工具（工具定义已通过 API 提供，无需手动格式化）：
- get_weather: 查询天气预报
- calculator: 数学计算
- get_current_time: 获取当前时间（精确到毫秒）
- save_memory / search_memory: 保存和搜索用户记忆
- search_documents: 搜索用户上传的文档
- list_directory / read_file / write_file / delete_file: 文件操作（read_file 支持 PDF、Word、文本）
- open_file: 打开文件或应用程序
- run_command: 执行系统命令（需用户确认）
- web_search: 网页搜索
- analyze_image: 读取本地图片并用 AI 分析内容（支持 PNG、JPG、GIF、BMP、WebP）
- translate: 文本翻译（支持中英日韩法德西俄等多种语言互译）

## 工具使用规则（极其重要，严格遵守，违反即为错误）
1. **每次最多调用 1-2 个工具**，绝对不要一次调用3个以上；
2. 只调用用户问题明确需要的工具，不要调用无关工具；
3. 用户问"介绍我/我是谁" → 只调 search_memory，不调其他；
4. 用户只问时间 → 只调 get_current_time，不要调 get_weather；
5. 用户只问天气 → 只调 get_weather，不要调 get_current_time；
6. 用户问计算 → 只调 calculator；
7. 用户要求翻译 → 只调 translate；
8. 不要主动添加用户未要求的额外工具调用；
9. 用户指令包含多个步骤时，必须完整执行所有步骤；
10. 工具返回的数据必须原样引用，不要省略精度。

## 输出格式规范
1. 日常问答：自然口语化，段落清晰；
2. 方案/清单：用分项、序号排版；
3. 数据、金额、时间等关键信息加粗；
4. 引用工具返回的时间时，必须保留完整的精度（包括毫秒），不得截断。

## 安全&合规
1. 拒绝色情、暴力、政治、违法、恶意诱导类请求；
2. 保护用户隐私；
3. 不生成违规脚本、恶意代码。
"""
