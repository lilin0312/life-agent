"""
语音通话服务 — WebSocket + STT + TTS 流水线
处理实时语音通话的全流程：音频接收 → 语音识别 → LLM 回复 → 语音合成 → 音频推送
"""
import asyncio
import base64
import json
import logging
import tempfile
import os

import httpx
from fastapi import WebSocket, WebSocketDisconnect

from app.config import (
    SILICONFLOW_API_KEY, SILICONFLOW_BASE_URL,
    TTS_VOICE, TTS_RATE, TTS_PITCH,
)

logger = logging.getLogger(__name__)


class CallService:
    """语音通话服务：STT → LLM → TTS 完整流水线"""

    def __init__(self, chat_service, tool_service):
        """
        Args:
            chat_service: ChatService 实例（复用 LLM 对话能力）
            tool_service: ToolService 实例（复用语音识别能力）
        """
        self.chat_service = chat_service
        self.tool_service = tool_service

    # ==================== STT: 语音识别 ====================

    async def speech_to_text(self, audio_bytes: bytes) -> str:
        """调用 SiliconFlow SenseVoiceSmall 将音频转为文字"""
        if not SILICONFLOW_API_KEY:
            logger.warning("[Call] STT 未配置 SiliconFlow API Key")
            return ""

        if len(audio_bytes) < 100:
            logger.warning(f"[Call] 音频数据太小 ({len(audio_bytes)} bytes)，跳过识别")
            return ""

        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(audio_bytes)
                tmp_path = f.name

            try:
                # 在线程池中执行同步 HTTP 请求，避免阻塞事件循环
                def _do_stt():
                    with open(tmp_path, "rb") as f:
                        return httpx.post(
                            f"{SILICONFLOW_BASE_URL}/audio/transcriptions",
                            headers={"Authorization": f"Bearer {SILICONFLOW_API_KEY}"},
                            files={"file": ("audio.wav", f, "audio/wav")},
                            data={"model": "FunAudioLLM/SenseVoiceSmall"},
                            timeout=30,
                        )

                resp = await asyncio.to_thread(_do_stt)
            finally:
                os.unlink(tmp_path)

            if resp.status_code != 200:
                logger.error(f"[Call] STT API 错误: {resp.status_code} {resp.text[:200]}")
                return ""

            data = resp.json()
            text = data.get("text", "").strip()

            # 安全过滤：如果 STT 返回了 HTML/代码内容，说明识别失败
            if text and (text.startswith("<!") or text.startswith("<html") or "<!DOCTYPE" in text):
                logger.error(f"[Call] STT 返回异常内容（疑似HTML）: {text[:200]}")
                return ""

            logger.info(f"[Call] STT 识别结果 ({len(text)}字): {text[:100]}")
            return text

        except Exception as e:
            logger.error(f"[Call] STT 失败: {e}")
            return ""

    # ==================== TTS: 文字转语音 ====================

    @staticmethod
    def _clean_for_speech(text: str) -> str:
        """清洗 LLM 输出中会被 TTS 逐字念出的 markdown/符号"""
        import re
        # 去掉 markdown 加粗 **text** / __text__
        text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
        text = re.sub(r'__(.+?)__', r'\1', text)
        # 去掉 markdown 斜体 *text* / _text_
        text = re.sub(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', r'\1', text)
        # 去掉 markdown 标题 # ## ###
        text = re.sub(r'^#{1,6}\s*', '', text, flags=re.MULTILINE)
        # 去掉 markdown 链接 [text](url) → text
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
        # 去掉 markdown 图片 ![alt](url)
        text = re.sub(r'!\[[^\]]*\]\([^)]+\)', '', text)
        # 去掉 markdown 列表标记 - * +
        text = re.sub(r'^[\s]*[-*+]\s+', '', text, flags=re.MULTILINE)
        # 去掉代码块 ```
        text = re.sub(r'```[\s\S]*?```', '', text)
        text = re.sub(r'`([^`]+)`', r'\1', text)
        # 去掉 emoji 和特殊符号（保留中文标点、字母、数字）
        # 去掉残留的 HTML 标签
        text = re.sub(r'<[^>]+>', '', text)
        # 多个连续换行合并为一个
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    def _build_ssml(self, text: str) -> str:
        """
        构建 SSML（语音合成标记语言），轻量优化：
        - 只做语速和音高微调，不做自动断句
        - 自动插入的 <break> 会让节奏变得机械，交给 TTS 引擎自己处理韵律
        """
        # 清洗 markdown 残留
        cleaned = self._clean_for_speech(text)

        # 转义 XML 特殊字符
        escaped = cleaned.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

        return (
            f'<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="zh-CN">'
            f'<voice name="{TTS_VOICE}">'
            f'<prosody rate="{TTS_RATE}" pitch="{TTS_PITCH}">'
            f'{escaped}'
            f'</prosody>'
            f'</voice>'
            f'</speak>'
        )

    async def text_to_speech(self, text: str) -> bytes:
        """使用 Edge TTS 将文本转为 MP3 音频字节"""
        if not text:
            return b""

        try:
            import edge_tts

            ssml = self._build_ssml(text)
            logger.info(f"[Call] TTS 文本 ({len(text)}字): {text[:60]}...")

            communicate = edge_tts.Communicate(ssml, TTS_VOICE)

            chunks = []
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    chunks.append(chunk["data"])

            audio_data = b"".join(chunks)
            logger.info(f"[Call] TTS 生成音频: {len(audio_data)} bytes")
            return audio_data

        except ImportError:
            logger.error("[Call] edge-tts 未安装，请执行 pip install edge-tts")
            return b""
        except Exception as e:
            logger.error(f"[Call] TTS 失败: {e}")
            return b""

    # ==================== 完整处理流水线 ====================

    async def process_turn(
        self, audio_bytes: bytes, user_id: str, session_id: str
    ) -> dict:
        """
        处理一轮语音对话：STT → LLM → TTS

        Returns:
            {"text": str, "audio": bytes, "session_id": str, "error": str|None}
        """
        # ---- Step 1: STT ----
        if not audio_bytes or len(audio_bytes) < 100:
            return {"text": "", "audio": b"", "session_id": session_id, "error": "音频数据为空"}

        user_text = await self.speech_to_text(audio_bytes)
        if not user_text:
            return {"text": "", "audio": b"", "session_id": session_id, "error": "语音识别失败，请重试"}

        logger.info(f"[Call] 用户说 ({user_id}): {user_text[:120]}")

        # ---- Step 2: LLM ----
        try:
            result = await self.chat_service.handle_chat(
                user_id=user_id,
                message=user_text,
                session_id=session_id,
                voice_mode=True,  # 启用口语化 prompt
            )
            response_text = result.get("content", "")
            session_id = result.get("session_id", session_id)
        except Exception as e:
            logger.error(f"[Call] LLM 失败: {e}")
            return {"text": "", "audio": b"", "session_id": session_id, "error": f"AI 处理失败: {e}"}

        if not response_text:
            return {"text": "", "audio": b"", "session_id": session_id, "error": "AI 未生成回复"}

        logger.info(f"[Call] AI 回复 ({user_id}): {response_text[:120]}")

        # 如果 LLM 触发了需要确认的危险操作，自动拒绝（通话中不便确认）
        if result.get("need_confirm"):
            pending_id = result.get("pending_id")
            if pending_id:
                try:
                    await self.chat_service.reject_action(pending_id, user_id)
                    logger.info(f"[Call] 通话中自动拒绝危险操作: {pending_id}")
                except Exception:
                    pass
            response_text = "这个操作需要在聊天界面确认，通话中暂时无法执行。你可以挂断后在聊天里告诉我。"

        # ---- Step 3: TTS ----
        audio_data = await self.text_to_speech(response_text)

        return {
            "user_text": user_text,
            "text": response_text,
            "audio": audio_data,
            "session_id": session_id,
            "error": None,
        }

    # ==================== WebSocket 主循环 ====================

    async def handle_websocket(self, websocket: WebSocket):
        """
        WebSocket 通话主循环

        协议:
        Client → Server:
          {"type":"start", "user_id":"...", "session_id":"..."}
          {"type":"audio", "data":"<base64>"}
          {"type":"interrupt"}
          {"type":"ping"}
          {"type":"end"}

        Server → Client:
          {"type":"status", "state":"listening"|"thinking"|"speaking"}
          {"type":"audio", "data":"<base64>", "index":0, "total":1}
          {"type":"text", "content":"...", "role":"user"|"assistant"}
          {"type":"error", "message":"..."}
          {"type":"pong"}
        """
        user_id = None
        session_id = None

        try:
            while True:
                try:
                    raw = await websocket.receive_text()
                except WebSocketDisconnect:
                    logger.info(f"[Call] 客户端断开: user={user_id}")
                    break
                except Exception as e:
                    logger.error(f"[Call] 接收消息失败: {e}")
                    break

                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    logger.warning(f"[Call] 无效 JSON: {raw[:100]}")
                    continue

                msg_type = msg.get("type", "")

                # ---- 开始通话 ----
                if msg_type == "start":
                    user_id = msg.get("user_id", "anonymous")
                    session_id = msg.get("session_id", "")
                    logger.info(f"[Call] 通话开始: user={user_id} session={session_id}")

                    await self._safe_send(websocket, {
                        "type": "status",
                        "state": "listening",
                        "message": "通话已连接，请说话",
                    })

                # ---- 音频数据 ----
                elif msg_type == "audio":
                    audio_b64 = msg.get("data", "")
                    if not audio_b64:
                        continue

                    try:
                        audio_bytes = base64.b64decode(audio_b64)
                    except Exception:
                        logger.warning("[Call] Base64 解码失败")
                        continue

                    # 通知前端正在处理
                    await self._safe_send(websocket, {
                        "type": "status",
                        "state": "thinking",
                    })

                    # 处理一轮对话
                    result = await self.process_turn(
                        audio_bytes, user_id or "anonymous", session_id or ""
                    )

                    if result.get("error"):
                        await self._safe_send(websocket, {
                            "type": "error",
                            "message": result["error"],
                        })
                        await self._safe_send(websocket, {
                            "type": "status",
                            "state": "listening",
                        })
                        continue

                    # 更新 session_id
                    if result.get("session_id"):
                        session_id = result["session_id"]

                    # 发送识别文字（用户说的）
                    if result.get("user_text"):
                        await self._safe_send(websocket, {
                            "type": "text",
                            "content": result["user_text"],
                            "role": "user",
                        })

                    # 发送 AI 回复文字
                    if result.get("text"):
                        await self._safe_send(websocket, {
                            "type": "text",
                            "content": result["text"],
                            "role": "assistant",
                        })

                    # 发送音频
                    if result.get("audio"):
                        audio_b64 = base64.b64encode(result["audio"]).decode("utf-8")
                        # 分块发送（每块约 32KB base64，适配 WebSocket 帧大小）
                        CHUNK_SIZE = 32 * 1024
                        total_chunks = (len(audio_b64) + CHUNK_SIZE - 1) // CHUNK_SIZE
                        for i in range(0, len(audio_b64), CHUNK_SIZE):
                            chunk_data = audio_b64[i:i + CHUNK_SIZE]
                            await self._safe_send(websocket, {
                                "type": "audio",
                                "data": chunk_data,
                                "index": i // CHUNK_SIZE,
                                "total": total_chunks,
                            })

                        await self._safe_send(websocket, {
                            "type": "status",
                            "state": "speaking",
                        })
                    else:
                        await self._safe_send(websocket, {
                            "type": "status",
                            "state": "listening",
                        })

                # ---- 用户打断 ----
                elif msg_type == "interrupt":
                    logger.info(f"[Call] 用户打断: user={user_id}")
                    await self._safe_send(websocket, {
                        "type": "status",
                        "state": "listening",
                    })

                # ---- 心跳 ----
                elif msg_type == "ping":
                    await self._safe_send(websocket, {"type": "pong"})

                # ---- 结束通话 ----
                elif msg_type == "end":
                    logger.info(f"[Call] 通话结束: user={user_id}")
                    break

                else:
                    logger.warning(f"[Call] 未知消息类型: {msg_type}")

        except Exception as e:
            logger.error(f"[Call] WebSocket 异常: {e}")
        finally:
            logger.info(f"[Call] 通话连接关闭: user={user_id}")

    async def _safe_send(self, websocket: WebSocket, data: dict):
        """安全发送消息，忽略连接断开错误"""
        try:
            await websocket.send_json(data)
        except Exception:
            pass
