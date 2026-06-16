"""
聊天编排服务 - 整合 LangGraph AgentGraph + 记忆 + RAG + 人机确认
"""
import base64
import logging
import os
from pathlib import Path
from typing import Optional

from app.config import SYSTEM_PROMPT, ZHIPU_API_KEY, ZHIPU_BASE_URL, CHAT_IMAGES_DIR
from app.memory_service import MemoryService
from app.rag_service import RAGService
from app.tool_service import ToolService
from app.agent_graph import AgentGraph

logger = logging.getLogger(__name__)


async def _vision_describe(image_base64: str, question: str) -> str:
    """用视觉模型将图片转为文字描述。优先 SiliconFlow，智谱兜底"""
    from openai import AsyncOpenAI

    from app.config import SILICONFLOW_API_KEY, SILICONFLOW_BASE_URL

    # 防幻觉 + 有趣 Prompt
    guard_prompt = (
        "请详细描述这张图片：场景、物体、颜色、文字（逐字抄录）、人物表情动作、"
        "整体氛围（搞笑/温馨/可爱/沙雕/震惊等）、有趣的细节。不要猜测具体人物姓名。"
        f"用户问题：{question or '请详细描述这张图片'}"
    )

    # 图片过大时压缩，避免 API 超时/拒收
    compressed_image = image_base64
    if len(image_base64) > 300 * 1024:  # >300KB base64 就压缩
        try:
            import io as _io
            from PIL import Image as _Image
            # 从 data URL 解码
            if ',' in image_base64:
                header, data = image_base64.split(',', 1)
            else:
                header, data = 'data:image/jpeg;base64', image_base64
            raw = base64.b64decode(data)
            img = _Image.open(_io.BytesIO(raw))
            # 缩放：最长边不超过 1024
            w, h = img.size
            if max(w, h) > 1024:
                ratio = 1024 / max(w, h)
                img = img.resize((int(w * ratio), int(h * ratio)), _Image.LANCZOS)
            # 编码为 JPEG，质量 60
            buf = _io.BytesIO()
            img.convert('RGB').save(buf, format='JPEG', quality=60)
            new_b64 = base64.b64encode(buf.getvalue()).decode()
            compressed_image = f'data:image/jpeg;base64,{new_b64}'
            logger.info(f'[vision] 图片压缩: {len(data)}B → {len(buf.getvalue())}B')
        except Exception as e:
            logger.warning(f'[vision] 图片压缩失败，用原图: {e}')

    # 方案1: SiliconFlow Qwen3-VL (Key 已验可用)
    if SILICONFLOW_API_KEY:
        try:
            client = AsyncOpenAI(
                api_key=SILICONFLOW_API_KEY,
                base_url=SILICONFLOW_BASE_URL,
                timeout=45.0,  # 大图需要更长时间
            )
            response = await client.chat.completions.create(
                model="Qwen/Qwen3-VL-8B-Instruct",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": guard_prompt},
                        {"type": "image_url", "image_url": {"url": compressed_image}},
                    ],
                }],
                temperature=0.1,
                max_tokens=500,
            )
            return response.choices[0].message.content or "[图片识别失败]"
        except Exception as e:
            logger.warning(f"[vision] SiliconFlow 视觉模型失败: {e}，尝试智谱备选")

    # 方案2: 智谱 glm-4v-flash 兜底
    if ZHIPU_API_KEY:
        try:
            client = AsyncOpenAI(
                api_key=ZHIPU_API_KEY,
                base_url=ZHIPU_BASE_URL,
                timeout=30.0,
            )
            response = await client.chat.completions.create(
                model="glm-4v-flash",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": guard_prompt},
                        {"type": "image_url", "image_url": {"url": image_base64}},
                    ],
                }],
                temperature=0.1,
                max_tokens=500,
            )
            return response.choices[0].message.content or "[图片识别失败]"
        except Exception as e:
            logger.warning(f"[vision] 智谱视觉模型也失败: {e}")

    return "[图片识别不可用：所有视觉模型均无法访问，请检查 API Key]"


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
        voice_mode: bool = False,
    ) -> dict:
        """
        处理用户聊天（支持图片输入）
        """
        sid = await self.memory.get_or_create_session(user_id, session_id)

        # 保存图片到磁盘（刷新后仍可查看）
        image_path = ""
        if image_base64:
            try:
                import uuid as _uuid
                os.makedirs(str(CHAT_IMAGES_DIR), exist_ok=True)
                img_name = f"{sid[:8]}_{_uuid.uuid4().hex[:8]}.jpg"
                img_file = str(CHAT_IMAGES_DIR / img_name)
                # 解码 base64 并保存为 JPEG
                if ',' in image_base64:
                    _, data = image_base64.split(',', 1)
                else:
                    data = image_base64
                raw = base64.b64decode(data)
                from io import BytesIO as _BytesIO
                from PIL import Image as _Image
                img = _Image.open(_BytesIO(raw))
                # 压缩后保存
                w, h = img.size
                if max(w, h) > 800:
                    ratio = 800 / max(w, h)
                    img = img.resize((int(w*ratio), int(h*ratio)), _Image.LANCZOS)
                img.convert('RGB').save(img_file, 'JPEG', quality=70)
                image_path = f"/chat-images/{img_name}"
                logger.info(f"[chat] 图片已保存: {img_file}")
            except Exception as e:
                logger.warning(f"[chat] 保存图片失败: {e}")

        # 消息内容附加图片标记
        msg_to_save = message
        if image_path:
            msg_to_save = f"[IMG]{image_path}[/IMG]{message}"
        await self.memory.save_message(sid, "user", msg_to_save)

        system_prompt = await self._build_system_prompt(user_id, message)

        # 语音通话模式：追加口语化指令
        if voice_mode:
            system_prompt += self._get_voice_prompt()
        history = await self.memory.get_history(sid)

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(history)

        # 如果有图片，先用视觉模型识别为文字，再交给主LLM处理
        # （主模型 deepseek-chat 不支持多模态输入，必须先转文字）
        if image_base64:
            vision_question = message if message else "请详细描述这张图片的内容"
            try:
                image_description = await _vision_describe(image_base64, vision_question)
                logger.info("[chat] 图片已通过视觉模型转为文字描述")
            except Exception as e:
                logger.warning(f"[chat] 视觉模型识别失败: {e}，降级为纯文字")
                image_description = f"[图片识别失败: {e}]"

            # 把图片描述作为纯文本消息添加，主LLM即可正常处理
            # 加入有趣的回复引导
            reply_guide = (
                "请根据图片内容给出有趣、有温度的回复。"
                "像朋友聊天一样——可以幽默吐槽、可以表达情感、可以发现有趣的细节。"
                "不要像写报告一样干巴巴描述。如果图片搞笑就一起笑，如果可爱就夸，如果有槽点就吐。"
                "回复控制在3-5句话以内。"
            )
            vision_msg = {
                "role": "user",
                "content": (
                    f"[用户上传了一张图片，视觉模型识别结果如下]\n{image_description}\n\n"
                    f"{reply_guide}\n用户说：{message if message else '看看这张图'}"
                ),
            }
            if messages and messages[-1]["role"] == "user":
                messages[-1] = vision_msg
            else:
                messages.append(vision_msg)

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
        memory_ctx = await self.memory.build_memory_context(user_id, query=message)
        if memory_ctx:
            prompt += memory_ctx
        if self.rag.is_ready:
            # RAG 检索是同步阻塞调用，丢到线程池避免卡住事件循环
            import asyncio
            try:
                rag_ctx = await asyncio.get_event_loop().run_in_executor(
                    None, self.rag.build_rag_context, user_id, message
                )
                if rag_ctx:
                    prompt += rag_ctx
            except Exception as e:
                logger.warning(f"RAG 上下文构建失败（已跳过）: {e}")
        return prompt

    @staticmethod
    def _get_voice_prompt() -> str:
        return """
[最高优先级] 你现在处于语音通话模式。你的文字将被 TTS 转为语音。

说话风格:
- 你已经不是在打字了，你是在跟朋友打语音电话
- 像真人聊天一样说话，用"嗯""哦""那个""就是说""对了""行吧""好嘞"
- 句尾带语气词：啊、哦、呢、吧、嘛
- 1到3句话说完，不要啰嗦
- 可以有思考停顿，用"嗯…"来表示

绝对禁止:
- 不要用任何书面结构：首先其次最后、建议您、根据分析、综上所述 都不准出现
- 不要在回复里用任何 markdown 格式，不要加粗不要标题不要列表
- 不要原样复述工具返回的数据，换成自己的话说
- 数字、时间、金额不需要标记，直接念出来就行
"""

    async def get_chat_history(self, session_id: str) -> list[dict]:
        return await self.memory.get_full_history(session_id)

    async def get_user_sessions(self, user_id: str) -> list[dict]:
        return await self.memory.get_user_sessions(user_id)
