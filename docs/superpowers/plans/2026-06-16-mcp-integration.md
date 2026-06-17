# MCP Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add MCP (Model Context Protocol) support — 1) Expose 20 existing tools as MCP Server for external AI agents, 2) MCP Client to connect external MCP servers and dynamically expand toolset.

**Architecture:** Use `mcp` Python SDK. MCP Server wraps existing ToolService tools via stdio transport (standalone process). MCP Client connects to configured servers at startup, discovers tools, merges into TOOL_DEFINITIONS, and proxies tool calls.

**Tech Stack:** Python 3.12, `mcp` SDK, stdio transport, existing FastAPI + LangGraph + ToolService

**Design decisions:**
- Transport: stdio (simplest, no port conflicts, Claude Desktop compatible)
- MCP Server: standalone entrypoint `run_mcp_server.py`, importable as module
- MCP Client: starts in lifespan, merges tools before AgentGraph init
- Content types: text only for now (images via base64 data URL)

---

## File Structure

```
Create:  app/mcp_tools.py       # Tool definitions in MCP format (bridges tool_schemas)
Create:  app/mcp_server.py      # MCP Server — wraps existing tools
Create:  app/mcp_client.py      # MCP Client — connects external servers
Create:  run_mcp_server.py      # Standalone MCP Server entrypoint
Modify:  app/config.py          # MCP server endpoints config
Modify:  app/main.py            # Init MCP client in lifespan, merge tools
Modify:  app/tool_service.py    # Route MCP tool calls through client
Modify:  requirements.txt       # Add mcp>=1.0.0
```

---

### Task 1: Install dependency

**Files:** `requirements.txt`

- [ ] Add `mcp>=1.0.0` to requirements.txt after `pyautogui>=0.9.0`

```python
mcp>=1.0.0
```

- [ ] Install and verify

Run: `pip install mcp>=1.0.0`
Expected: Successfully installed mcp

- [ ] Commit

```bash
git add requirements.txt
git commit -m "chore: add mcp SDK dependency"
```

---

### Task 2: Add MCP config

**Files:** `app/config.py`

- [ ] Add MCP server endpoints config — insert after SILICONFLOW_SPEECH_MODEL line (~line 63)

```python
# ==================== MCP 配置 ====================
# 要连接的 MCP 服务器列表，JSON 格式：
# [{"name":"filesystem","command":"npx","args":["-y","@modelcontextprotocol/server-filesystem","/tmp"]}]
MCP_SERVERS_JSON = os.getenv("MCP_SERVERS", "[]")
```

- [ ] Commit

```bash
git add app/config.py
git commit -m "feat: add MCP server config"
```

---

### Task 3: Create MCP tool definitions bridge

**Files:** Create `app/mcp_tools.py`

This file converts existing tool_schemas to MCP-compatible tool definitions and provides a dispatch function.

```python
"""
MCP 工具定义桥接层 — 将 tool_schemas.py 的 Schema 转为 MCP Tool 格式
"""
import json
from typing import Any

def get_mcp_tools(exclude_dangerous: bool = True) -> list[dict]:
    """
    将 TOOL_DEFINITIONS 转为 MCP 兼容的 Tool 列表
    默认排除危险工具，MCP 协议要求每个操作都需显式确认
    """
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
    """
    执行 MCP 工具调用，返回 MCP 格式的内容列表
    """
    from app.tool_service import ToolService

    ts = ToolService(None, None)  # MCP 调用时不带 memory/rag
    result = await ts.execute(tool_name, arguments, user_id, confirmed=False)

    content = result.get("content", "")
    if result.get("need_confirm"):
        content = f"[需要确认] {content}\n请通过 pending_id={result['pending_id']} 二次调用确认。"

    return [{"type": "text", "text": content}]
```

- [ ] Commit

```bash
git add app/mcp_tools.py
git commit -m "feat: add MCP tool definitions bridge"
```

---

### Task 4: Create MCP Server

**Files:** Create `app/mcp_server.py`

Wraps existing tools as an MCP stdio server. Other AI clients (Claude Desktop, etc.) can connect to use the agent's tools.

```python
"""
MCP Server — 将 20 个工具暴露为标准 MCP 协议
其他 AI 客户端可通过 stdio 连接使用这些工具
"""
import asyncio
import json
import logging
from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationCapabilities
from mcp.server.stdio import stdio_server

from app.mcp_tools import get_mcp_tools, call_mcp_tool

logger = logging.getLogger(__name__)

server = Server("life-agent-mcp")


@server.list_tools()
async def handle_list_tools() -> list:
    """列出所有可用工具"""
    return get_mcp_tools(exclude_dangerous=True)


@server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list:
    """执行工具调用"""
    logger.info(f"MCP tool call: {name}({json.dumps(arguments, ensure_ascii=False)[:200]})")
    try:
        result = await call_mcp_tool(name, arguments)
        return result
    except Exception as e:
        logger.error(f"MCP tool error [{name}]: {e}")
        return [{"type": "text", "text": f"工具执行出错: {e}"}]


async def run_mcp_server():
    """启动 MCP stdio 服务器"""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationCapabilities(
                sampling={},
                experimental={},
                roots={},
            ),
            NotificationOptions(),
        )
```

- [ ] Commit

```bash
git add app/mcp_server.py
git commit -m "feat: add MCP Server wrapping existing tools"
```

---

### Task 5: Create standalone MCP Server entrypoint

**Files:** Create `run_mcp_server.py`

```python
"""
MCP Server 独立启动入口
用法: python run_mcp_server.py
其他 AI 客户端通过 stdio 连接此进程即可使用 life-agent 的工具
"""
import asyncio
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [MCP] %(message)s",
    stream=sys.stderr,  # stdio transport uses stdout for JSON-RPC
)

if __name__ == "__main__":
    from app.mcp_server import run_mcp_server
    print("Life-Agent MCP Server starting...", file=sys.stderr)
    asyncio.run(run_mcp_server())
```

- [ ] Commit

```bash
git add run_mcp_server.py
git commit -m "feat: add standalone MCP Server entrypoint"
```

---

### Task 6: Create MCP Client

**Files:** Create `app/mcp_client.py`

Connects to external MCP servers, discovers their tools, and merges them into the agent's toolset.

```python
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
        self.sessions: dict[str, Any] = {}       # server_name -> ClientSession
        self.tools: list[dict] = []               # MCP-format tools from all servers
        self._tool_name_to_server: dict[str, str] = {}  # tool_name -> server_name

    async def connect_all(self):
        """连接所有配置的 MCP 服务器并发现工具"""
        from mcp.client.stdio import stdio_client, StdioServerParameters

        for cfg in self.servers_config:
            name = cfg.get("name", "unknown")
            command = cfg.get("command", "")
            args = cfg.get("args", [])
            env = cfg.get("env", {})

            if not command:
                logger.warning(f"MCP server '{name}': no command configured, skipping")
                continue

            try:
                params = StdioServerParameters(command=command, args=args)
                # stdio_client is a context manager that returns (read, write, session)
                # We use it as an async context manager
                transport = await stdio_client(params).__aenter__()
                read_stream, write_stream = transport[0], transport[1]
                
                from mcp.client.session import ClientSession
                session = ClientSession(read_stream, write_stream)
                await session.initialize()
                
                # Discover tools
                tools_result = await session.list_tools()
                for tool in tools_result.tools:
                    self._tool_name_to_server[tool.name] = name
                    self.tools.append({
                        "name": tool.name,
                        "description": tool.description or "",
                        "inputSchema": tool.inputSchema or {},
                    })
                
                self.sessions[name] = session
                logger.info(f"MCP server '{name}': connected, {len(tools_result.tools)} tools")
            except Exception as e:
                logger.warning(f"MCP server '{name}': connection failed: {e}")

    def get_tool_schemas(self) -> list[dict]:
        """获取所有外部工具的 LangChain function calling Schema"""
        schemas = []
        for tool in self.tools:
            props = tool.get("inputSchema", {}).get("properties", {})
            required = tool.get("inputSchema", {}).get("required", [])
            # Convert JSON Schema props to OpenAI function calling format
            openai_props = {}
            for key, val in props.items():
                prop = {}
                if "type" in val:
                    prop["type"] = val["type"]
                if "description" in val:
                    prop["description"] = val["description"]
                if "enum" in val:
                    prop["enum"] = val["enum"]
                openai_props[key] = prop

            schemas.append({
                "type": "function",
                "function": {
                    "name": f"mcp_{tool['name']}",  # 前缀避免与内置工具冲突
                    "description": f"[MCP:{tool['name']}] {tool.get('description','')}",
                    "parameters": {
                        "type": "object",
                        "properties": openai_props,
                        "required": required,
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
        # Extract text from MCP content items
        texts = []
        for item in result.content:
            if hasattr(item, "text"):
                texts.append(item.text)
        return "\n".join(texts) if texts else str(result.content)

    async def close_all(self):
        """关闭所有连接"""
        for name, session in self.sessions.items():
            try:
                await session.close()
            except Exception:
                pass
        self.sessions.clear()
        logger.info("All MCP connections closed")


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
```

- [ ] Commit

```bash
git add app/mcp_client.py
git commit -m "feat: add MCP Client for external server connections"
```

---

### Task 7: Integrate MCP into app startup

**Files:** `app/main.py`

- [ ] In lifespan startup, after AgentGraph init but before mount, init MCP manager and merge tools

Add to imports:
```python
from app.mcp_client import get_mcp_manager, close_mcp
```

After `call_service = CallService(...)`:
```python
    # 初始化 MCP 客户端（连接外部 MCP 服务器，扩展工具）
    mcp_manager = await get_mcp_manager()
    mcp_tools = mcp_manager.get_tool_schemas()
    if mcp_tools:
        # 将 MCP 工具合并到 Agent 的工具列表中
        from app.tool_schemas import TOOL_DEFINITIONS
        TOOL_DEFINITIONS.extend(mcp_tools)
        logger.info(f"  MCP 工具: ✅ 已加载 {len(mcp_tools)} 个外部工具")

    agent_graph = AgentGraph(llm_service.client, tool_service, zhipu_llm=llm_service.zhipu_client)
    # Store mcp_manager on app.state for tool routing
    app.state.mcp_manager = mcp_manager
```

In shutdown (before `memory_service.close()`):
```python
    await close_mcp()
```

- [ ] Commit

```bash
git add app/main.py
git commit -m "feat: integrate MCP client into app startup"
```

---

### Task 8: Route MCP tool calls in ToolService

**Files:** `app/tool_service.py`

- [ ] Add `app_mcp` as available tool name, and add handler. Insert before the available list and add `_tool_app_mcp`:

In `DANGEROUS_TOOLS` line ~109, add the new tool to available list. Add method:

```python
    def _tool_app_mcp(self, args: dict, user_id: str = "") -> str:
        """MCP 工具代理——将调用转发到外部 MCP Server"""
        # tool_name 由 execute() 传入，这里需要从外部获取
        return "MCP 工具调用出错：未找到 MCP 管理器"
```

- [ ] Modify `execute()` to intercept `mcp_*` tool calls:

After `handler = getattr(self, f"_tool_{tool_name}", None)`:

```python
            if handler is None:
                # 检查是否为 MCP 外部工具
                if tool_name.startswith("mcp_"):
                    return await self._execute_mcp_tool(tool_name, args)
                available = [...]
                return {"content": f"未知工具: {tool_name}", "need_confirm": False}
```

- [ ] Add `_execute_mcp_tool()` method:

```python
    async def _execute_mcp_tool(self, tool_name: str, args: dict) -> dict:
        """执行 MCP 外部工具"""
        try:
            from app.mcp_client import get_mcp_manager
            manager = await get_mcp_manager()
            result = await manager.call_tool(tool_name, args)
            logger.info(f"MCP工具结果 [{tool_name}]: {str(result)[:200]}")
            return {"content": str(result), "need_confirm": False}
        except Exception as e:
            logger.error(f"MCP工具异常 [{tool_name}]: {e}")
            return {"content": f"MCP工具执行出错: {e}", "need_confirm": False}
```

- [ ] Commit

```bash
git add app/tool_service.py
git commit -m "feat: add MCP tool routing in ToolService"
```

---

### Task 9: Test MCP Server locally

- [ ] Verify MCP server starts and lists tools

Run:
```bash
cd E:\work\专高六\life-agent
echo '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}' | python run_mcp_server.py 2>nul
```
Expected: JSON response with tool list including get_weather, calculator, etc.

- [ ] Verify tool execution

Run:
```bash
echo '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"calculator","arguments":{"expression":"1+1"}}}' | python run_mcp_server.py 2>nul
```
Expected: JSON response with "1+1 = 2"

---

### Task 10: End-to-end verification

- [ ] Restart server: `python run.py`
- [ ] Check startup logs for "MCP 工具: ✅ 已加载"
- [ ] Send a chat message: "1+1等于几"
- [ ] Verify calculator tool still works
- [ ] Add a test MCP server (e.g. filesystem):
  1. `npm install -g @modelcontextprotocol/server-filesystem`
  2. Set env var: `MCP_SERVERS=[{"name":"fs","command":"npx","args":["-y","@modelcontextprotocol/server-filesystem","/tmp"]}]`
  3. Restart, verify "MCP 工具: ✅ 已加载 N 个外部工具"
- [ ] Commit

```bash
git add -A
git commit -m "test: MCP end-to-end verification"
git push
```
