"""
API 路由 - 全异步，并发安全
"""
import logging
import time
import sqlite3

from fastapi import APIRouter, Request, UploadFile, File, Form, HTTPException, WebSocket

from app.schemas import (
    ChatRequest, ChatResponse, ConfirmRequest,
    UploadResponse, MemoryResponse, HealthResponse,
)
from app.config import MEMORY_DB_PATH, VECTOR_DB_DIR, UPLOAD_DIR

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api")


# ==================== 健康检查 ====================

@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request):
    app = request.app
    return HealthResponse(
        status="ok",
        llm_ready=app.state.llm.is_ready,
        rag_ready=app.state.rag.is_ready,
        memory_ready=app.state.memory.is_ready,
    )


# ==================== 聊天接口 ====================

@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request):
    """核心聊天接口（异步，支持多用户并发）"""
    start_time = time.time()
    chat_service = request.app.state.chat_service

    try:
        result = await chat_service.handle_chat(
            user_id=req.user_id,
            message=req.message,
            session_id=req.session_id,
            image_base64=req.image_base64,
            voice_mode=req.voice_mode,
        )

        elapsed = time.time() - start_time
        logger.info(
            f"[Chat] user={req.user_id} session={result['session_id'][:8]} "
            f"tool={result.get('tool_used')} time={elapsed:.2f}s"
        )

        return ChatResponse(
            success=True,
            content=result["content"],
            session_id=result["session_id"],
            tool_used=result.get("tool_used"),
            need_confirm=result.get("need_confirm", False),
            pending_id=result.get("pending_id"),
            confirm_preview=result.get("confirm_preview"),
        )

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"[Chat] FAIL user={req.user_id} time={elapsed:.2f}s: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"处理失败: {e}")


# ==================== 会话历史 ====================

@router.get("/history/{session_id}")
async def get_history(session_id: str, request: Request):
    chat_service = request.app.state.chat_service
    history = await chat_service.get_chat_history(session_id)
    return {"success": True, "history": history}


# ==================== 文件上传（RAG）====================

@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    user_id: str = Form(...),
):
    allowed_ext = {".txt", ".md", ".csv", ".json", ".text", ".log"}
    filename = file.filename or "unknown.txt"
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ".txt"

    if ext not in allowed_ext:
        raise HTTPException(status_code=400, detail=f"不支持的文件类型 {ext}")

    content = await file.read()
    if len(content) > 5 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="文件大小不能超过 5MB")

    rag_service = request.app.state.rag
    if not rag_service.is_ready:
        raise HTTPException(status_code=503, detail="RAG 服务暂不可用")

    result = await rag_service.upload_document(user_id, filename, content)
    if not result["success"]:
        raise HTTPException(status_code=500, detail=result["message"])

    return UploadResponse(
        success=True, message=result["message"],
        filename=filename, chunks=result["chunks"],
    )


# ==================== 用户记忆 ====================

@router.get("/memory/{user_id}", response_model=MemoryResponse)
async def get_memories(user_id: str, request: Request):
    memory_service = request.app.state.memory
    memories = await memory_service.get_memories(user_id)
    return MemoryResponse(
        success=True,
        memories=[f"[{m['mem_key']}] {m['content']}" for m in memories],
    )


@router.delete("/memory/{user_id}/{mem_key}")
async def delete_memory(user_id: str, mem_key: str, request: Request):
    memory_service = request.app.state.memory
    await memory_service.delete_memory(user_id, mem_key)
    return {"success": True, "message": f"已删除记忆: {mem_key}"}


# ==================== 用户会话列表 ====================

@router.get("/sessions/{user_id}")
async def get_sessions(user_id: str, request: Request):
    chat_service = request.app.state.chat_service
    sessions = await chat_service.get_user_sessions(user_id)
    return {"success": True, "sessions": sessions}


# ==================== 人机确认：确认/拒绝危险操作 ====================

@router.post("/confirm")
async def confirm_action(req: ConfirmRequest, request: Request):
    """用户确认或拒绝执行危险操作"""
    chat_service = request.app.state.chat_service

    if req.action == "confirm":
        result = await chat_service.confirm_action(req.pending_id, req.user_id)
    else:
        result = await chat_service.reject_action(req.pending_id, req.user_id)

    return result


# ==================== 语音识别 ====================

@router.post("/speech-to-text")
async def speech_to_text(
    request: Request,
    file: UploadFile = File(...),
):
    """语音转文字"""
    tool_service = request.app.state.chat_service.tool_service
    audio_data = await file.read()
    if len(audio_data) == 0:
        raise HTTPException(status_code=400, detail="音频数据为空")
    text = tool_service.transcribe_audio(audio_data)
    return {"success": True, "text": text}


# ==================== 管理面板：数据库查看 ====================

@router.get("/admin/db-stats")
async def get_db_stats():
    """获取数据库统计信息"""
    conn = sqlite3.connect(str(MEMORY_DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        stats = {}
        for table in ["sessions", "chat_history", "user_memory", "pending_actions"]:
            try:
                cnt = conn.execute(f"SELECT COUNT(*) as c FROM {table}").fetchone()["c"]
                stats[table] = cnt
            except Exception:
                stats[table] = 0
        return {"success": True, "stats": stats}
    finally:
        conn.close()


@router.get("/admin/all-memories/{user_id}")
async def get_all_memories(user_id: str):
    """获取用户所有记忆（详细信息）"""
    conn = sqlite3.connect(str(MEMORY_DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT id, mem_key, content, created_at, updated_at FROM user_memory WHERE user_id = ? ORDER BY updated_at DESC",
            (user_id,),
        ).fetchall()
        return {"success": True, "memories": [dict(r) for r in rows]}
    finally:
        conn.close()


@router.get("/admin/all-sessions/{user_id}")
async def get_all_sessions(user_id: str):
    """获取用户所有会话（含消息数统计）"""
    conn = sqlite3.connect(str(MEMORY_DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """SELECT s.session_id, s.created_at, s.updated_at,
                      (SELECT COUNT(*) FROM chat_history WHERE session_id = s.session_id) as msg_count
               FROM sessions s WHERE s.user_id = ? ORDER BY s.updated_at DESC LIMIT 20""",
            (user_id,),
        ).fetchall()
        return {"success": True, "sessions": [dict(r) for r in rows]}
    finally:
        conn.close()


@router.get("/admin/vectordb")
async def get_vectordb_info(request: Request):
    """获取向量库信息"""
    import os
    rag_service = request.app.state.rag
    info = {"ready": rag_service.is_ready, "documents": 0, "files": []}

    # 上传文件列表
    if UPLOAD_DIR.exists():
        for f in UPLOAD_DIR.iterdir():
            if f.is_file():
                info["files"].append({
                    "name": f.name,
                    "size": f.stat().st_size,
                })

    # 向量库大小
    if VECTOR_DB_DIR.exists():
        total_size = sum(f.stat().st_size for f in VECTOR_DB_DIR.rglob("*") if f.is_file())
        info["vectordb_size"] = total_size

    return {"success": True, "info": info}


@router.delete("/admin/session/{session_id}")
async def delete_session(session_id: str):
    """删除会话及其聊天记录"""
    conn = sqlite3.connect(str(MEMORY_DB_PATH))
    try:
        conn.execute("DELETE FROM chat_history WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
        conn.commit()
        return {"success": True, "message": f"已删除会话: {session_id[:12]}..."}
    finally:
        conn.close()


# ==================== 语音通话 WebSocket ====================

@router.websocket("/call")
async def call_websocket(websocket: WebSocket):
    """语音通话 WebSocket 端点 — STT + LLM + TTS 全双工实时通信"""
    await websocket.accept()
    # 从 app.state 获取 CallService（通过 FastAPI 的 request 机制无法在 WebSocket 中直接获取，
    # 这里通过依赖注入模式，实际在 lifespan 中挂载到 websocket.app.state）
    call_service = websocket.app.state.call_service
    await call_service.handle_websocket(websocket)
