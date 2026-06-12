"""
聊天编排服务 - 整合 LangGraph AgentGraph + 记忆 + RAG + 人机确认
"""
import logging
import os
from pathlib import Path
from typing import Optional

from app.config import SYSTEM_PROMPT
from app.memory_service import MemoryService
from app.rag_service import RAGService
from app.tool_service import ToolService
from app.agent_graph import AgentGraph

logger = logging.getLogger(__name__)


class ChatService:
    """聊天编排器：基于 LangGraph AgentGraph 的全异步服务"""

    def __init__(self, agent_graph: AgentGraph, memory_service: MemoryService, rag_service: RAGService):
        self.agent_graph = agent_graph
        self.memory = memory_service
        self.rag = rag_service
        self.tool_service = ToolService(memory_service, rag_service)

    async def handle_chat(
        self, user_id: str, message: str, session_id: Optional[str] = None,
        image_base64: Optional[str] = None,
    ) -> dict:
        """
        处理用户聊天（支持图片输入）
        """
        sid = await self.memory.get_or_create_session(user_id, session_id)
        await self.memory.save_message(sid, "user", message)

        system_prompt = await self._build_system_prompt(user_id, message)
        history = await self.memory.get_history(sid)

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(history)

        # 如果有图片，构造多模态用户消息
        if image_base64:
            user_msg = {
                "role": "user",
                "content": [
                    {"type": "text", "text": message if message else "请分析这张图片"},
                    {"type": "image_url", "image_url": {"url": image_base64}},
                ],
            }
            if messages and messages[-1]["role"] == "user":
                messages[-1] = user_msg
            else:
                messages.append(user_msg)

        # ---- 调用 LangGraph AgentGraph ----
        graph_result = await self.agent_graph.run(
            messages=messages,
            user_id=user_id,
            session_id=sid,
        )

        content = graph_result.get("final_response", "抱歉，回复生成失败。")
        tools_used_list = graph_result.get("tools_used", [])
        tool_used = ", ".join(tools_used_list) if tools_used_list else None

        # 保存助手回复
        await self.memory.save_message(sid, "assistant", content)

        response = {
            "content": content,
            "session_id": sid,
            "tool_used": tool_used,
        }

        # 如果有待确认操作，附加确认信息
        pending = graph_result.get("pending_confirmation")
        if pending:
            response["need_confirm"] = True
            response["pending_id"] = pending["pending_id"]
            response["confirm_preview"] = pending["preview"]

        return response

    async def confirm_action(self, pending_id: str, user_id: str) -> dict:
        """用户确认执行危险操作"""
        action = self.tool_service.get_pending_action(pending_id)
        if not action:
            return {"success": False, "content": "操作已过期或不存在，请重新发送指令。"}

        if action["user_id"] != user_id:
            return {"success": False, "content": "无权执行此操作。"}

        # 真正执行（confirmed=True）
        result = await self.tool_service.execute(
            action["tool_name"], action["args"], user_id, confirmed=True
        )

        # 清理 pending 记录
        self.tool_service.remove_pending_action(pending_id)

        logger.info(f"[已确认执行] {action['tool_name']} pending_id={pending_id}")
        return {"success": True, "content": result["content"]}

    async def reject_action(self, pending_id: str, user_id: str) -> dict:
        """用户拒绝执行"""
        action = self.tool_service.get_pending_action(pending_id)
        if action:
            self.tool_service.remove_pending_action(pending_id)
        return {"success": True, "content": "已取消操作。"}

    async def _build_system_prompt(self, user_id: str, message: str) -> str:
        # 填入系统环境信息
        user_home = str(Path.home()).replace("\\", "/")
        user_desktop = str(Path.home() / "Desktop").replace("\\", "/")
        user_downloads = str(Path.home() / "Downloads").replace("\\", "/")
        prompt = SYSTEM_PROMPT.format(
            user_home=user_home,
            user_desktop=user_desktop,
            user_downloads=user_downloads,
        )
        memory_ctx = await self.memory.build_memory_context(user_id)
        if memory_ctx:
            prompt += memory_ctx
        if self.rag.is_ready:
            rag_ctx = self.rag.build_rag_context(user_id, message)
            if rag_ctx:
                prompt += rag_ctx
        return prompt

    async def get_chat_history(self, session_id: str) -> list[dict]:
        return await self.memory.get_full_history(session_id)

    async def get_user_sessions(self, user_id: str) -> list[dict]:
        return await self.memory.get_user_sessions(user_id)
