"""
API 路由 - 全异步，并发安全
"""
import logging
import time

from fastapi import APIRouter, Request, UploadFile, File, Form, HTTPException

from app.schemas import (
    ChatRequest, ChatResponse, ConfirmRequest,
    UploadResponse, MemoryResponse, HealthResponse,
)

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
    conn = memory_service._get_conn()
    try:
        conn.execute("DELETE FROM user_memory WHERE user_id = ? AND mem_key = ?", (user_id, mem_key))
        conn.commit()
        return {"success": True, "message": f"已删除记忆: {mem_key}"}
    finally:
        conn.close()


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
