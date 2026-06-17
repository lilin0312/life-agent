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
    stream=sys.stderr,
)

if __name__ == "__main__":
    from app.mcp_server import run_mcp_server
    print("Life-Agent MCP Server starting...", file=sys.stderr)
    asyncio.run(run_mcp_server())
