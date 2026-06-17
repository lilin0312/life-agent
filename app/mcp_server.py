"""
MCP Server — 将 20 个工具暴露为标准 MCP 协议
其他 AI 客户端（Claude Desktop 等）通过 stdio 连接使用
"""
import asyncio
import json
import logging
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from app.mcp_tools import get_mcp_tools, call_mcp_tool

logger = logging.getLogger(__name__)

server = Server("life-agent-mcp")


@server.list_tools()
async def handle_list_tools() -> list[Tool]:
    """列出所有可用工具"""
    tools = get_mcp_tools(exclude_dangerous=True)
    mcp_tools = []
    for t in tools:
        mcp_tools.append(Tool(
            name=t["name"],
            description=t["description"],
            inputSchema=t["inputSchema"],
        ))
    logger.info(f"MCP: 列出 {len(mcp_tools)} 个工具")
    return mcp_tools


@server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[TextContent]:
    """执行工具调用"""
    logger.info(f"MCP call: {name}({json.dumps(arguments, ensure_ascii=False)[:200]})")
    try:
        result = await call_mcp_tool(name, arguments)
        return [TextContent(type="text", text=item["text"]) for item in result]
    except Exception as e:
        logger.error(f"MCP error [{name}]: {e}")
        return [TextContent(type="text", text=f"工具执行出错: {e}")]


async def run_mcp_server():
    """启动 MCP stdio 服务器"""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())
