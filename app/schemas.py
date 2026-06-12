"""
请求/响应数据模型 (Pydantic v2)
"""
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class ChatRequest(BaseModel):
    """聊天请求"""
    user_id: str = Field(..., description="用户唯一标识", min_length=1, max_length=64)
    message: str = Field(..., description="用户消息", min_length=1, max_length=2000)
    session_id: Optional[str] = Field(None, description="会话ID，不传则新建会话")
    image_base64: Optional[str] = Field(None, description="图片base64编码（含data:image前缀）")


class ChatResponse(BaseModel):
    """聊天响应"""
    success: bool = True
    content: str = Field(..., description="AI回复内容")
    session_id: str = Field(..., description="会话ID")
    tool_used: Optional[str] = Field(None, description="使用的工具名称")
    need_confirm: bool = False
    pending_id: Optional[str] = Field(None, description="待确认操作ID")
    confirm_preview: Optional[str] = Field(None, description="操作预览描述")
    timestamp: datetime = Field(default_factory=datetime.now)


class ConfirmRequest(BaseModel):
    """确认/拒绝操作请求"""
    pending_id: str = Field(..., description="待确认操作ID")
    user_id: str = Field(..., description="用户ID")
    action: str = Field("confirm", description="confirm 或 reject")


class UploadResponse(BaseModel):
    """文件上传响应"""
    success: bool = True
    message: str
    filename: str
    chunks: int = 0


class MemoryResponse(BaseModel):
    """记忆查询响应"""
    success: bool = True
    memories: list[str] = []


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str = "ok"
    llm_ready: bool = False
    rag_ready: bool = False
    memory_ready: bool = False
