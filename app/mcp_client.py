"""
MCP Client — 连接外部 MCP Server，动态扩展工具集
"""
import asyncio
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


class MCPClientManager:
    """管理多个 MCP Server 连接，合并工具并代理调用"""

    def __init__(self, servers_config: list[dict]):
        self.servers_config = servers_config
        self.sessions: dict[str, Any] = {}
        self.tools: list[dict] = []
        self._tool_name_to_server: dict[str, str] = {}
        self._transports: list = []

    async def connect_all(self):
        """连接所有配置的 MCP 服务器并发现工具"""
        from mcp.client.stdio import stdio_client
        from mcp import ClientSession

        for cfg in self.servers_config:
            name = cfg.get("name", "unknown")
            command = cfg.get("command", "")
            args = cfg.get("args", [])

            if not command:
                logger.warning(f"MCP '{name}': no command, skipping")
                continue

            try:
                # stdio_client returns (read, write)
                transport = stdio_client(command, args)
                read_stream, write_stream = await transport.__aenter__()
                self._transports.append(transport)

                session = ClientSession(read_stream, write_stream)
                await session.__aenter__()
                await session.initialize()

                tools_result = await session.list_tools()
                count = 0
                for tool in tools_result.tools:
                    self._tool_name_to_server[tool.name] = name
                    self.tools.append({
                        "name": tool.name,
                        "description": tool.description or "",
                        "inputSchema": tool.inputSchema or {},
                    })
                    count += 1

                self.sessions[name] = session
                logger.info(f"MCP '{name}': connected, {count} tools")
            except Exception as e:
                logger.warning(f"MCP '{name}': failed - {e}")

    def get_tool_schemas(self) -> list[dict]:
        """获取所有外部工具的 OpenAI function calling Schema"""
        schemas = []
        for tool in self.tools:
            props = tool.get("inputSchema", {}).get("properties", {})
            req = tool.get("inputSchema", {}).get("required", [])

            openai_props = {}
            for key, val in props.items():
                p = {}
                for k in ("type", "description", "enum"):
                    if k in val:
                        p[k] = val[k]
                openai_props[key] = p

            schemas.append({
                "type": "function",
                "function": {
                    "name": f"mcp_{tool['name']}",
                    "description": f"[MCP] {tool.get('description', tool['name'])}",
                    "parameters": {
                        "type": "object",
                        "properties": openai_props,
                        "required": req,
                    },
                },
            })
        return schemas

    async def call_tool(self, tool_name: str, arguments: dict) -> str:
        """调用 MCP 工具（去掉 mcp_ 前缀）"""
        name = tool_name.replace("mcp_", "", 1)
        server_name = self._tool_name_to_server.get(name)
        if not server_name:
            return f"MCP tool '{name}' not found"

        session = self.sessions.get(server_name)
        if not session:
            return f"MCP server '{server_name}' not connected"

        result = await session.call_tool(name, arguments)
        texts = []
        for item in result.content:
            if hasattr(item, "text"):
                texts.append(item.text)
        return "\n".join(texts) if texts else str(result.content)

    async def close_all(self):
        """关闭所有连接"""
        for session in self.sessions.values():
            try:
                await session.__aexit__(None, None, None)
            except Exception:
                pass
        self.sessions.clear()
        for t in self._transports:
            try:
                await t.__aexit__(None, None, None)
            except Exception:
                pass
        self._transports.clear()
        logger.info("MCP connections closed")


# 全局单例
_mcp_manager: MCPClientManager | None = None


async def get_mcp_manager() -> MCPClientManager:
    global _mcp_manager
    if _mcp_manager is None:
        from app.config import MCP_SERVERS_JSON
        try:
            servers = json.loads(MCP_SERVERS_JSON)
        except json.JSONDecodeError:
            servers = []
        _mcp_manager = MCPClientManager(servers)
        await _mcp_manager.connect_all()
    return _mcp_manager


async def close_mcp():
    global _mcp_manager
    if _mcp_manager:
        await _mcp_manager.close_all()
        _mcp_manager = None
