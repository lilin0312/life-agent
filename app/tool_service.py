"""
工具服务 - 真正可用的工具集
包含：天气、计算器、时间、记忆、文档检索、文件操作、打开应用、执行命令、网页搜索、图片分析、翻译
"""
import ast
import base64
import json
import logging
import operator
import os
import re
import sqlite3
import subprocess
import shutil
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Optional

import httpx

from app.config import UPLOAD_DIR, MEMORY_DB_PATH, LLM_API_KEY, LLM_BASE_URL, LLM_MODEL

# 通义千问视觉模型配置（用于图片分析）
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
DASHSCOPE_VISION_URL = "https://dashscope.aliyuncs.com/compatible_mode/v1"
DASHSCOPE_VISION_MODEL = "qwen-vl-plus"

logger = logging.getLogger(__name__)

# 安全的数学运算符
SAFE_OPERATORS = {
    ast.Add: operator.add, ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
    ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos,
}

# ==================== 城市坐标映射 ====================
CITY_COORDS = {
    "北京": (39.90, 116.40), "上海": (31.23, 121.47), "广州": (23.13, 113.26),
    "深圳": (22.54, 114.06), "成都": (30.57, 104.07), "杭州": (30.27, 120.15),
    "武汉": (30.59, 114.31), "南京": (32.06, 118.80), "西安": (34.26, 108.94),
    "重庆": (29.56, 106.55), "天津": (39.13, 117.20), "苏州": (31.30, 120.62),
    "长沙": (28.23, 112.94), "郑州": (34.75, 113.65), "青岛": (36.07, 120.38),
    "大连": (38.91, 121.60), "厦门": (24.48, 118.09), "昆明": (25.04, 102.68),
    "哈尔滨": (45.75, 126.65), "沈阳": (41.80, 123.43), "济南": (36.65, 116.98),
    "合肥": (31.82, 117.23), "福州": (26.07, 119.30), "南宁": (22.82, 108.37),
    "贵阳": (26.65, 106.63), "太原": (37.87, 112.55), "石家庄": (38.04, 114.51),
    "兰州": (36.06, 103.83), "乌鲁木齐": (43.83, 87.62), "拉萨": (29.65, 91.17),
    "海口": (20.04, 110.35), "三亚": (18.25, 109.50), "香港": (22.32, 114.17),
    "台北": (25.03, 121.57),
}

WEATHER_CODES = {
    0: "晴", 1: "大部晴", 2: "多云", 3: "阴天",
    45: "雾", 48: "雾凇", 51: "小毛毛雨", 53: "毛毛雨", 55: "大毛毛雨",
    61: "小雨", 63: "中雨", 65: "大雨", 66: "冻雨", 67: "大冻雨",
    71: "小雪", 73: "中雪", 75: "大雪", 77: "雪粒",
    80: "阵雨", 81: "中阵雨", 82: "大阵雨",
    85: "阵雪", 86: "大阵雪",
    95: "雷暴", 96: "雷暴+冰雹", 99: "强雷暴+冰雹",
}

# ==================== 系统保护路径（禁止删/改）====================
PROTECTED_PATHS = [
    r"c:\windows", r"c:\program files", r"c:\program files (x86)",
    r"c:\programdata", r"c:\users\all users",
]


def _safe_eval(expr: str) -> float:
    """安全数学表达式计算"""
    expr = expr.replace("×", "*").replace("÷", "/").replace("％", "%").replace("^", "**").strip()
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        raise ValueError(f"无效表达式: {expr}")

    def _eval_node(node):
        if isinstance(node, ast.Expression):
            return _eval_node(node.body)
        elif isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        elif isinstance(node, ast.Num):
            return node.n
        elif isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in SAFE_OPERATORS:
                raise ValueError(f"不支持的运算符: {op_type.__name__}")
            return SAFE_OPERATORS[op_type](_eval_node(node.left), _eval_node(node.right))
        elif isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type not in SAFE_OPERATORS:
                raise ValueError(f"不支持的运算符: {op_type.__name__}")
            return SAFE_OPERATORS[op_type](_eval_node(node.operand))
        else:
            raise ValueError(f"不支持的表达式: {type(node).__name__}")

    result = _eval_node(tree)
    return int(result) if isinstance(result, float) and result == int(result) else round(result, 6)


class ToolService:
    """工具执行服务 - 支持人机确认机制"""

    # 危险工具列表（需要用户确认才能执行）
    DANGEROUS_TOOLS = {"delete_file", "run_command"}

    def __init__(self, memory_service=None, rag_service=None):
        self.memory_service = memory_service
        self.rag_service = rag_service
        # LLM 客户端（用于图片分析和翻译）
        self._llm_client = None

    def set_llm_client(self, client):
        """设置 LangChain ChatOpenAI 客户端"""
        self._llm_client = client

    async def execute(
        self, tool_name: str, args: dict, user_id: str = "", confirmed: bool = False
    ) -> dict:
        """
        执行工具并返回结果。
        返回 {"content": str, "need_confirm": bool, "pending_id": str|None}
        """
        try:
            handler = getattr(self, f"_tool_{tool_name}", None)
            if handler is None:
                available = [
                    "get_weather", "calculator", "get_current_time",
                    "save_memory", "search_memory", "search_documents",
                    "list_directory", "read_file", "write_file",
                    "delete_file", "open_file", "run_command", "web_search",
                    "analyze_image", "translate",
                ]
                return {"content": f"未知工具: {tool_name}。可用: {', '.join(available)}", "need_confirm": False}

            # 危险操作 → 需要确认
            if tool_name in self.DANGEROUS_TOOLS and not confirmed:
                pending_id = self._save_pending_action(user_id, tool_name, args)
                preview = self._preview_dangerous_action(tool_name, args)
                logger.info(f"[确认等待] {tool_name} pending_id={pending_id}")
                return {
                    "content": preview,
                    "need_confirm": True,
                    "pending_id": pending_id,
                }

            # 安全操作 或 已确认 → 直接执行
            logger.info(f"执行工具: {tool_name}({args})")
            import asyncio
            result = handler(args, user_id)
            if asyncio.iscoroutine(result):
                result = await result
            logger.info(f"工具结果: {str(result)[:200]}")
            return {"content": str(result), "need_confirm": False}

        except Exception as e:
            logger.error(f"工具执行异常 [{tool_name}]: {e}")
            return {"content": f"工具执行出错: {e}", "need_confirm": False}

    # ==================== 确认机制 ====================

    def _save_pending_action(self, user_id: str, tool_name: str, args: dict) -> str:
        """将待确认操作存入 SQLite，返回 pending_id"""
        import uuid
        pending_id = str(uuid.uuid4())[:8]
        conn = self._get_pending_db()
        try:
            conn.execute(
                "INSERT INTO pending_actions (id, user_id, tool_name, args_json, created_at) VALUES (?, ?, ?, ?, ?)",
                (pending_id, user_id, tool_name, json.dumps(args, ensure_ascii=False), datetime.now().isoformat()),
            )
            conn.commit()
        finally:
            conn.close()
        return pending_id

    def get_pending_action(self, pending_id: str) -> Optional[dict]:
        """根据 ID 获取待确认操作"""
        conn = self._get_pending_db()
        try:
            row = conn.execute(
                "SELECT id, user_id, tool_name, args_json FROM pending_actions WHERE id = ?",
                (pending_id,),
            ).fetchone()
            if row:
                return {"id": row[0], "user_id": row[1], "tool_name": row[2], "args": json.loads(row[3])}
            return None
        finally:
            conn.close()

    def remove_pending_action(self, pending_id: str):
        """删除已处理的待确认操作"""
        conn = self._get_pending_db()
        try:
            conn.execute("DELETE FROM pending_actions WHERE id = ?", (pending_id,))
            conn.commit()
        finally:
            conn.close()

    def _get_pending_db(self) -> sqlite3:
        """获取 pending_actions 表的连接"""
        db_path = os.path.join(os.path.dirname(MEMORY_DB_PATH), "memory.db")
        conn = sqlite3.connect(str(db_path), timeout=10)
        conn.execute("""CREATE TABLE IF NOT EXISTS pending_actions (
            id TEXT PRIMARY KEY, user_id TEXT NOT NULL, tool_name TEXT NOT NULL,
            args_json TEXT NOT NULL, created_at TEXT NOT NULL
        )""")
        conn.commit()
        return conn

    def _preview_dangerous_action(self, tool_name: str, args: dict) -> str:
        """生成危险操作的预览描述"""
        if tool_name == "delete_file":
            path = args.get("path", args.get("file", ""))
            return f"⚠️ **即将删除**: `{path}`\n\n此操作不可撤销，确认要删除吗？"
        elif tool_name == "run_command":
            cmd = args.get("command", args.get("cmd", ""))
            return f"⚠️ **即将执行命令**: `{cmd}`\n\n确认要执行吗？"
        return f"⚠️ 即将执行 **{tool_name}**，确认吗？"

    # ==================== 天气查询 ====================

    def _tool_get_weather(self, args: dict, user_id: str = "") -> str:
        city = args.get("city", args.get("location", "北京"))
        date_str = args.get("date", "")

        coords = CITY_COORDS.get(city)
        if not coords:
            return f"暂不支持查询「{city}」的天气。支持: {', '.join(list(CITY_COORDS.keys())[:15])}等"

        lat, lon = coords
        try:
            resp = httpx.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": lat, "longitude": lon,
                    "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max",
                    "timezone": "Asia/Shanghai", "forecast_days": 7,
                },
                timeout=15,
            )
            data = resp.json()
        except Exception as e:
            return f"天气数据获取失败: {e}"

        daily = data.get("daily", {})
        dates = daily.get("time", [])
        if not dates:
            return "天气数据暂时不可用"

        t_max = daily.get("temperature_2m_max", [])
        t_min = daily.get("temperature_2m_min", [])
        weather_codes = daily.get("weather_code", [])
        wind = daily.get("wind_speed_10m_max", [])
        rain = daily.get("precipitation_sum", [])

        lines = [f"**{city}** 天气预报\n"]
        today = datetime.now().strftime("%Y-%m-%d")
        tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

        for i, d in enumerate(dates):
            if date_str and date_str != d:
                continue
            w_desc = WEATHER_CODES.get(weather_codes[i], "未知") if i < len(weather_codes) else "?"
            high = t_max[i] if i < len(t_max) else "?"
            low = t_min[i] if i < len(t_min) else "?"
            w = wind[i] if i < len(wind) else "?"
            r = rain[i] if i < len(rain) else "?"

            day_label = d
            if d == today:
                day_label = f"{d}（今天）"
            elif d == tomorrow:
                day_label = f"{d}（明天）"

            lines.append(f"**{day_label}**: {w_desc}，{low}°C ~ {high}°C，风速{w}km/h，降水{r}mm")
            if date_str:
                break

        return "\n".join(lines)

    # ==================== 计算器 ====================

    def _tool_calculator(self, args: dict, user_id: str = "") -> str:
        expression = args.get("expression", "")
        if not expression:
            return "请提供要计算的表达式"
        try:
            result = _safe_eval(expression)
            return f"**{expression} = {result}**"
        except ValueError as e:
            return f"计算失败: {e}"
        except Exception as e:
            return f"计算出错: {e}"

    # ==================== 当前时间 ====================

    def _tool_get_current_time(self, args: dict, user_id: str = "") -> str:
        tz_name = args.get("timezone", "Asia/Shanghai")
        tz_map = {"Asia/Shanghai": 8, "Asia/Tokyo": 9, "America/New_York": -4, "Europe/London": 1, "UTC": 0}
        offset = tz_map.get(tz_name, 8)
        now = datetime.now(timezone(timedelta(hours=offset)))
        weekdays = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
        return f"**{now.strftime('%Y年%m月%d日')} {weekdays[now.weekday()]} {now.strftime('%H:%M:%S.%f')}**（{tz_name}）"

    # ==================== 记忆 ====================

    async def _tool_save_memory(self, args: dict, user_id: str = "") -> str:
        if not self.memory_service:
            return "记忆服务不可用"
        key = args.get("key", "")
        content = args.get("content", "")
        if not key or not content:
            return "请提供记忆标签和内容"
        await self.memory_service.save_memory(user_id, key, content)
        return f"已记住: **{key}** → {content}"

    async def _tool_search_memory(self, args: dict, user_id: str = "") -> str:
        if not self.memory_service:
            return "记忆服务不可用"
        keyword = args.get("keyword", args.get("query", ""))
        results = await self.memory_service.search_memories(user_id, keyword)
        if not results:
            return "未找到相关记忆"
        return "\n".join(results)

    # ==================== 文档检索 ====================

    def _tool_search_documents(self, args: dict, user_id: str = "") -> str:
        if not self.rag_service or not self.rag_service.is_ready:
            return "文档检索服务不可用。请先上传文档。"
        query = args.get("query", "")
        results = self.rag_service.search(user_id, query)
        if not results:
            return "未找到相关文档内容。"
        return "\n\n".join(f"[片段{i+1}] {r}" for i, r in enumerate(results))

    # ==================== 文件操作 ====================

    def _tool_list_directory(self, args: dict, user_id: str = "") -> str:
        path = args.get("path", args.get("dir", ""))
        if not path:
            return "请提供目录路径，例如: path='C:/Users/李琳/Desktop'"

        real = self._resolve_path(path)
        if not real:
            return f"路径不存在: {path}"

        if not os.path.isdir(real):
            return f"不是目录: {path}"

        try:
            items = os.listdir(real)
            if not items:
                return f"目录为空: {path}"

            lines = [f"**{path}** ({len(items)}项)\n"]
            for item in sorted(items)[:80]:
                full = os.path.join(real, item)
                try:
                    if os.path.isdir(full):
                        lines.append(f"  📂 {item}/")
                    else:
                        size = os.path.getsize(full)
                        size_str = f"{size/1024:.1f}KB" if size > 1024 else f"{size}B"
                        lines.append(f"  📄 {item} ({size_str})")
                except OSError:
                    lines.append(f"  🔒 {item}")

            if len(items) > 80:
                lines.append(f"\n  ... 还有{len(items)-80}项")
            return "\n".join(lines)
        except PermissionError:
            return f"没有权限: {path}"
        except Exception as e:
            return f"读取失败: {e}"

    def _tool_read_file(self, args: dict, user_id: str = "") -> str:
        path = args.get("path", args.get("file", ""))
        if not path:
            return "请提供文件路径"

        real = self._resolve_path(path)
        if not real:
            return f"文件不存在: {path}"

        if os.path.isdir(real):
            return f"这是目录不是文件: {path}"

        if os.path.getsize(real) > 5 * 1024 * 1024:
            return f"文件太大（>5MB）: {path}"

        ext = Path(real).suffix.lower()

        # PDF 文件
        if ext == ".pdf":
            try:
                from PyPDF2 import PdfReader
                reader = PdfReader(real)
                pages = []
                for i, page in enumerate(reader.pages[:50]):
                    text = page.extract_text()
                    if text:
                        pages.append(f"--- 第{i+1}页 ---\n{text}")
                content = "\n\n".join(pages) if pages else "PDF 文件无法提取文本"
                return f"**{path}** ({len(reader.pages)}页)\n\n{content[:20000]}"
            except Exception as e:
                return f"PDF 读取失败: {e}"

        # Word 文件
        if ext in (".docx", ".doc"):
            try:
                from docx import Document
                doc = Document(real)
                paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
                content = "\n".join(paragraphs)
                # 也读取表格
                tables_text = []
                for table in doc.tables:
                    for row in table.rows:
                        cells = [cell.text for cell in row.cells]
                        tables_text.append(" | ".join(cells))
                if tables_text:
                    content += "\n\n--- 表格 ---\n" + "\n".join(tables_text)
                return f"**{path}**\n\n{content[:20000]}"
            except Exception as e:
                return f"Word 文件读取失败: {e}"

        # 普通文本文件
        try:
            with open(real, "r", encoding="utf-8", errors="replace") as f:
                content = f.read(20000)
            return f"**{path}**\n\n{content}"
        except Exception as e:
            return f"读取失败: {e}"

    def _tool_write_file(self, args: dict, user_id: str = "") -> str:
        path = args.get("path", args.get("file", ""))
        content = args.get("content", "")
        if not path:
            return "请提供文件路径和内容"

        # 解析路径（允许新文件）
        target = os.path.expanduser(path)
        target = os.path.expandvars(target)
        target = os.path.normpath(target)

        if self._is_protected(target):
            return f"不允许写入系统目录: {path}"

        try:
            parent = os.path.dirname(target)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(target, "w", encoding="utf-8") as f:
                f.write(content)
            return f"✅ 已创建/写入文件: **{path}** ({len(content)}字符)"
        except Exception as e:
            return f"写入失败: {e}"

    def _tool_delete_file(self, args: dict, user_id: str = "") -> str:
        path = args.get("path", args.get("file", ""))
        if not path:
            return "请提供要删除的文件路径"

        real = self._resolve_path(path)
        if not real:
            return f"文件不存在: {path}"

        if self._is_protected(real):
            return f"⛔ 禁止删除系统文件: {path}"

        if not os.path.exists(real):
            return f"不存在: {path}"

        try:
            if os.path.isdir(real):
                shutil.rmtree(real)
                return f"✅ 已删除文件夹: **{path}**"
            else:
                os.remove(real)
                return f"✅ 已删除文件: **{path}**"
        except PermissionError:
            return f"没有删除权限: {path}"
        except Exception as e:
            return f"删除失败: {e}"

    # ==================== 打开文件/应用 ====================

    def _tool_open_file(self, args: dict, user_id: str = "") -> str:
        """
        用系统默认程序打开文件，或启动应用程序
        Windows 用 os.startfile，Linux/Mac 用 subprocess
        """
        path = args.get("path", args.get("file", args.get("app", args.get("application", ""))))
        if not path:
            return "请提供文件路径或应用程序名称"

        # 尝试直接路径
        real = self._resolve_path(path)

        # 如果不是直接路径，尝试作为应用程序名搜索
        if not real:
            app_path = shutil.which(path)
            if not app_path:
                # Windows 常见应用搜索
                app_path = self._find_windows_app(path)
            if app_path:
                real = app_path

        if not real:
            return f"找不到文件或应用: {path}"

        try:
            if os.name == "nt":  # Windows
                os.startfile(real)
            else:  # Linux/Mac
                subprocess.Popen(["xdg-open" if os.name != "darwin" else "open", real])
            return f"✅ 已打开: **{path}**"
        except Exception as e:
            return f"打开失败: {e}"

    def _find_windows_app(self, name: str) -> str:
        """在 Windows 常见路径中搜索应用程序"""
        name_lower = name.lower()

        # 常见应用映射
        app_map = {
            "chrome": r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            "firefox": r"C:\Program Files\Mozilla Firefox\firefox.exe",
            "edge": r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            "微信": r"C:\Program Files\Tencent\WeChat\WeChat.exe",
            "wechat": r"C:\Program Files\Tencent\WeChat\WeChat.exe",
            "qq": r"C:\Program Files\Tencent\QQ\Bin\QQ.exe",
            "vscode": r"C:\Users\李琳\AppData\Local\Programs\Microsoft VS Code\new_Code.exe",
            "code": r"C:\Users\李琳\AppData\Local\Programs\Microsoft VS Code\new_Code.exe",
            "notepad": "notepad.exe",
            "记事本": "notepad.exe",
            "calc": "calc.exe",
            "计算器": "calc.exe",
            "explorer": "explorer.exe",
            "word": r"C:\Program Files\Microsoft Office\root\Office16\WINWORD.EXE",
            "excel": r"C:\Program Files\Microsoft Office\root\Office16\EXCEL.EXE",
            "ppt": r"C:\Program Files\Microsoft Office\root\Office16\POWERPNT.EXE",
            "powerpoint": r"C:\Program Files\Microsoft Office\root\Office16\POWERPNT.EXE",
            "wps": r"C:\Users\李琳\AppData\Local\Kingsoft\WPS Office\wps.exe",
            "typora": r"C:\Program Files\Typora\Typora.exe",
        }

        # 精确匹配
        if name_lower in app_map:
            p = app_map[name_lower]
            if os.path.exists(p):
                return p

        # 模糊匹配
        for key, p in app_map.items():
            if name_lower in key or key in name_lower:
                if os.path.exists(p):
                    return p

        # 在 PATH 和常见目录中搜索
        for ext in [".exe", ".lnk", ".url"]:
            result = shutil.which(name + ext) or shutil.which(name)
            if result:
                return result

        # 搜索桌面快捷方式
        desktop = os.path.expanduser("~/Desktop")
        for f in os.listdir(desktop):
            if name_lower in f.lower():
                return os.path.join(desktop, f)

        # 搜索开始菜单
        start_menu = os.path.expanduser(
            r"~\AppData\Roaming\Microsoft\Windows\Start Menu\Programs"
        )
        for root, dirs, files in os.walk(start_menu):
            for f in files:
                if name_lower in f.lower():
                    return os.path.join(root, f)

        return ""

    # ==================== 执行命令 ====================

    def _tool_run_command(self, args: dict, user_id: str = "") -> str:
        """执行系统命令（有安全限制）"""
        command = args.get("command", args.get("cmd", ""))
        if not command:
            return "请提供要执行的命令"

        # 安全黑名单
        blocked = ["format", "del /", "rmdir /s", "rd /s", "shutdown", "taskkill /f",
                    "reg delete", "reg add", "net user", "net localgroup",
                    "takeown /f c:", "cipher /w", "diskpart"]
        cmd_lower = command.lower()
        for b in blocked:
            if b in cmd_lower:
                return f"⛔ 禁止执行危险命令: {command}"

        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30,
                encoding="utf-8",
                errors="replace",
            )
            output = result.stdout.strip()
            error = result.stderr.strip()

            if result.returncode == 0:
                msg = f"✅ 命令执行成功"
                if output:
                    msg += f":\n{output[:3000]}"
                return msg
            else:
                msg = f"⚠️ 命令返回码 {result.returncode}"
                if error:
                    msg += f":\n{error[:1000]}"
                return msg
        except subprocess.TimeoutExpired:
            return "⏱️ 命令执行超时（30秒限制）"
        except Exception as e:
            return f"执行失败: {e}"

    # ==================== 网页搜索 ====================

    def _tool_web_search(self, args: dict, user_id: str = "") -> str:
        query = args.get("query", args.get("q", args.get("keyword", "")))
        if not query:
            return "请提供搜索关键词"

        try:
            resp = httpx.get(
                "https://html.duckduckgo.com/html/",
                params={"q": query},
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=10,
            )
            results = []
            title_pattern = r'<a[^>]*class="result__a"[^>]*>(.*?)</a>'
            snippet_pattern = r'<a[^>]*class="result__snippet"[^>]*>(.*?)</a>'

            titles = re.findall(title_pattern, resp.text, re.DOTALL)
            snippets = re.findall(snippet_pattern, resp.text, re.DOTALL)

            for i in range(min(5, len(titles))):
                title = re.sub(r'<[^>]+>', '', titles[i]).strip()
                snippet = re.sub(r'<[^>]+>', '', snippets[i]).strip() if i < len(snippets) else ""
                if title:
                    results.append(f"{i+1}. **{title}**\n   {snippet[:150]}")

            if not results:
                return f"未找到关于「{query}」的结果"

            return f"搜索: **{query}**\n\n" + "\n\n".join(results)
        except Exception as e:
            return f"搜索失败: {e}"

    # ==================== 路径处理 ====================

    def _resolve_path(self, path: str) -> str:
        """解析路径，存在则返回真实路径，不存在返回空串"""
        if not path:
            return ""
        p = os.path.expanduser(path)
        p = os.path.expandvars(p)
        p = os.path.normpath(p)
        if os.path.exists(p):
            return p
        return ""

    def _is_protected(self, path: str) -> bool:
        """检查是否是受保护的系统路径"""
        p = os.path.normpath(path).lower()
        for prot in PROTECTED_PATHS:
            if p.startswith(prot.lower()):
                return True
        return False

    # ==================== 图片分析 ====================

    def _tool_analyze_image(self, args: dict, user_id: str = "") -> str:
        """读取图片文件并使用 LLM 分析描述"""
        path = args.get("path", args.get("file", ""))
        question = args.get("question", "请详细描述这张图片的内容")
        if not path:
            return "请提供图片文件路径"

        real = self._resolve_path(path)
        if not real:
            return f"文件不存在: {path}"

        # 检查文件大小（限制 10MB）
        if os.path.getsize(real) > 10 * 1024 * 1024:
            return f"图片太大（>10MB）: {path}"

        # 读取图片并转 base64
        try:
            ext = Path(real).suffix.lower()
            mime_map = {
                ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                ".gif": "image/gif", ".bmp": "image/bmp", ".webp": "image/webp",
            }
            mime_type = mime_map.get(ext, "image/jpeg")

            with open(real, "rb") as f:
                img_data = f.read()
            b64 = base64.b64encode(img_data).decode("utf-8")
            data_url = f"data:{mime_type};base64,{b64}"
        except Exception as e:
            return f"读取图片失败: {e}"

        # 调用通义千问视觉模型
        try:
            from openai import OpenAI
            client = OpenAI(
                api_key=DASHSCOPE_API_KEY,
                base_url=DASHSCOPE_VISION_URL,
                timeout=30.0,
            )
            response = client.chat.completions.create(
                model=DASHSCOPE_VISION_MODEL,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }],
                temperature=0.3,
            )
            description = response.choices[0].message.content or "无法分析该图片"
            return f"**图片分析** ({path})\n\n{description}"
        except Exception as e:
            return f"图片分析失败: {e}"

    # ==================== 翻译 ====================

    def _tool_translate(self, args: dict, user_id: str = "") -> str:
        """使用 LLM 翻译文本"""
        text = args.get("text", "")
        target_lang = args.get("target_lang", args.get("target", "中文"))
        source_lang = args.get("source_lang", args.get("source", ""))

        if not text:
            return "请提供要翻译的文本"

        source_hint = f"从{source_lang}" if source_lang else ""
        prompt = (
            f"你是一位专业翻译。请将以下文本{source_hint}翻译为**{target_lang}**。\n"
            f"要求：\n"
            f"1. 翻译准确、自然、符合{target_lang}表达习惯\n"
            f"2. 保留原文的格式（如列表、加粗等）\n"
            f"3. 只输出翻译结果，不要解释\n\n"
            f"原文：\n{text}"
        )

        try:
            from openai import OpenAI
            client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL, timeout=30.0)
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            result = response.choices[0].message.content or "翻译失败"
            return f"**翻译结果** ({source_lang or '自动检测'} → {target_lang})\n\n{result}"
        except Exception as e:
            return f"翻译失败: {e}"
