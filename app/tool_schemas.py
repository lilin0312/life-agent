"""
工具 Schema 定义 — 供 LangGraph bind_tools() 使用
每个工具对应 OpenAI function calling 格式，参数名与 tool_service.py 的 _tool_* 方法一一对应
"""

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询城市天气预报，支持未来7天预报，返回天气状况、温度、风速、降水量等信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名，如：北京、上海、广州、深圳、成都等",
                    },
                    "date": {
                        "type": "string",
                        "description": "查询日期（可选），格式 YYYY-MM-DD，不填则返回7天预报",
                    },
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "安全数学表达式计算，支持加减乘除、幂运算、取余等",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "数学表达式，如：(5000-2000)/12、2**10、100*1.08",
                    },
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "获取当前日期和时间，支持不同时区",
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string",
                        "description": "时区（可选），如 Asia/Shanghai、Asia/Tokyo、America/New_York、Europe/London、UTC，默认 Asia/Shanghai",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_memory",
            "description": "保存用户偏好、习惯或重要信息到长期记忆，后续对话中可自动回忆",
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "记忆标签/分类，如：饮食偏好、生日、常用地址",
                    },
                    "content": {
                        "type": "string",
                        "description": "记忆的具体内容",
                    },
                },
                "required": ["key", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_memory",
            "description": "从长期记忆中搜索之前保存的信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {
                        "type": "string",
                        "description": "搜索关键词",
                    },
                },
                "required": ["keyword"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": "从用户上传的文档中检索相关内容（基于语义相似度）",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索查询词或问题",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "列出指定目录下的文件和子目录",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "目录路径，如 C:/Users/用户名/Desktop",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "读取文件内容，支持文本文件",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "文件路径",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "创建新文件或写入内容到指定文件",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "文件路径",
                    },
                    "content": {
                        "type": "string",
                        "description": "要写入的内容",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_file",
            "description": "删除文件或文件夹（危险操作，需要用户确认）",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "要删除的文件或文件夹路径",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "open_file",
            "description": "用系统默认程序打开文件或启动应用程序（如 chrome、微信、notepad 等）",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "文件路径或应用名称，如 chrome、微信、notepad、C:/Users/用户名/Desktop/report.docx",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "执行系统命令（危险操作，需要用户确认，有安全限制）",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "要执行的命令字符串",
                    },
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "使用搜索引擎搜索网页信息，返回相关结果标题和摘要",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词",
                    },
                },
                "required": ["query"],
            },
        },
    },
    # ---- 图片分析 ----
    {
        "type": "function",
        "function": {
            "name": "analyze_image",
            "description": "读取本地图片文件并使用 AI 视觉模型分析图片内容，支持 PNG、JPG、GIF、BMP、WebP 格式",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "图片文件路径",
                    },
                    "question": {
                        "type": "string",
                        "description": "关于图片的问题或分析要求（可选），默认为'请详细描述这张图片的内容'",
                    },
                },
                "required": ["path"],
            },
        },
    },
    # ---- 翻译 ----
    {
        "type": "function",
        "function": {
            "name": "translate",
            "description": "将文本翻译为指定语言，支持中英日韩法德西俄等多种语言互译",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "要翻译的文本内容",
                    },
                    "target_lang": {
                        "type": "string",
                        "description": "目标语言，如：中文、英文、日语、韩语、法语、德语、西班牙语",
                    },
                    "source_lang": {
                        "type": "string",
                        "description": "源语言（可选，不填则自动检测），如：英文、中文",
                    },
                },
                "required": ["text", "target_lang"],
            },
        },
    },
]
