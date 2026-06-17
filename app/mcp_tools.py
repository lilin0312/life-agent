"""
MCP 工具桥接 — 将 tool_schemas.py 的 Function Calling Schema 转为 MCP Tool 格式
"""
import json
import logging

logger = logging.getLogger(__name__)


def get_mcp_tools(exclude_dangerous: bool = True) -> list[dict]:
    """将 TOOL_DEFINITIONS 转为 MCP 兼容的 Tool 列表"""
    from app.tool_schemas import TOOL_DEFINITIONS
    from app.tool_service import ToolService

    dangerous = ToolService.DANGEROUS_TOOLS
    tools = []
    for td in TOOL_DEFINITIONS:
        func = td["function"]
        name = func["name"]
        if exclude_dangerous and name in dangerous:
            continue
        tools.append({
            "name": name,
            "description": func["description"],
            "inputSchema": {
                "type": "object",
                "properties": func["parameters"].get("properties", {}),
                "required": func["parameters"].get("required", []),
            },
        })
    return tools


async def call_mcp_tool(tool_name: str, arguments: dict, user_id: str = "mcp") -> list[dict]:
    """执行 MCP 工具调用，返回 MCP Content 列表"""
    from app.tool_service import ToolService

    ts = ToolService(None, None)
    result = await ts.execute(tool_name, arguments, user_id, confirmed=False)

    content = result.get("content", "")
    if result.get("need_confirm"):
        content = (
            f"[需要确认] {content}\n"
            f"请通过 pending_id={result['pending_id']} 二次调用确认。"
        )

    return [{"type": "text", "text": content}]
