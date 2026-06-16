"""
工具 Schema 定义 — 供 LangGraph bind_tools() 使用
每个工具对应 OpenAI function calling 格式，参数名与 tool_service.py 的 _tool_* 方法一一对应
"""

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询全球任意城市天气预报（支持中文名、拼音、英文名），未来7天。不填城市则自动IP定位",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名（可选），如：北京、上海、广州。不填则自动使用IP定位的城市",
                    },
                    "date": {
                        "type": "string",
                        "description": "查询日期（可选），格式 YYYY-MM-DD，不填则返回7天预报",
                    },
                },
                "required": [],
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
    # ---- 图片生成 ----
    {
        "type": "function",
        "function": {
            "name": "generate_image",
            "description": "根据文字描述生成图片，支持各种场景：人物、风景、动漫、Logo等",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "图片描述，尽量详细，如：一只可爱的橘猫坐在窗台上，阳光照射，温馨风格",
                    },
                    "size": {
                        "type": "string",
                        "description": "图片尺寸（可选）：1024x1024（默认）、512x512、1024x768、768x1024",
                    },
                },
                "required": ["prompt"],
            },
        },
    },
    # ---- 一站式输入 ----
    {
        "type": "function",
        "function": {
            "name": "app_write_text",
            "description": "一站式：打开应用(如未打开)+等待加载+输入文字。用户说'打开XX写YYY'时，直接调用此工具一步完成，不要再分步操作",
            "parameters": {
                "type": "object",
                "properties": {
                    "app": {"type": "string", "description": "应用名，如：记事本、notepad、word"},
                    "text": {"type": "string", "description": "要输入的文字内容"},
                    "window_title": {"type": "string", "description": "窗口标题关键词(可选)，默认同app名"},
                },
                "required": ["app", "text"],
            },
        },
    },
    # ---- 截图 + 视觉分析 ----
    {
        "type": "function",
        "function": {
            "name": "app_screenshot",
            "description": "截取屏幕并可选调用AI视觉模型分析截图内容。让AI能'看到'屏幕上有什么——按钮在哪里、弹窗内容是什么、软件界面长什么样。截图+分析后，可用app_click_at点击指定坐标",
            "parameters": {
                "type": "object",
                "properties": {
                    "region": {
                        "type": "string",
                        "enum": ["full", "active_window"],
                        "description": "截图区域：full=全屏, active_window=当前活动窗口",
                    },
                    "question": {
                        "type": "string",
                        "description": "要问视觉模型的问题（可选）。例如：'这个界面有什么按钮？''播放按钮在哪里？''弹窗上写的什么？'。不填则只截图不分析",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "app_click_at",
            "description": "在屏幕指定坐标位置点击鼠标。先通过app_screenshot+视觉分析确定目标位置，再用此工具点击（需用户确认）",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "integer", "description": "X 坐标（像素）"},
                    "y": {"type": "integer", "description": "Y 坐标（像素）"},
                    "button": {"type": "string", "enum": ["left", "right", "middle"], "description": "鼠标按键，默认left"},
                    "clicks": {"type": "integer", "description": "点击次数，1=单击 2=双击，默认1"},
                },
                "required": ["x", "y"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "app_clipboard",
            "description": "剪贴板操作：复制(copy=Ctrl+C)、粘贴(paste=Ctrl+V)、读取当前剪贴板内容(get)、写入文字(set)",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["copy", "paste", "get", "set"], "description": "操作类型"},
                    "text": {"type": "string", "description": "要写入剪贴板的文字（仅action=set时需要）"},
                },
                "required": ["action"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "app_list_windows",
            "description": "列出当前桌面所有打开的窗口及其标题和尺寸",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "app_drag",
            "description": "鼠标拖拽操作：从起始坐标拖动到目标坐标（需用户确认）",
            "parameters": {
                "type": "object",
                "properties": {
                    "x1": {"type": "integer", "description": "起始 X"},
                    "y1": {"type": "integer", "description": "起始 Y"},
                    "x2": {"type": "integer", "description": "目标 X"},
                    "y2": {"type": "integer", "description": "目标 Y"},
                    "duration": {"type": "number", "description": "拖拽持续时间（秒），默认0.5"},
                },
                "required": ["x1", "y1", "x2", "y2"],
            },
        },
    },
]
