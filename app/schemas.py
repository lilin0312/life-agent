"""
请求/响应数据模型 (Pydantic v2)
"""
from pydantic import BaseModel, Field #Basemodel自带自动类型校验、数据解析、报错提示，所有请求/响应模型都要继承它
from typing import Optional #代表字段可选，可以传null
from datetime import datetime #时间戳类型，用于记录接口响应时间


class ChatRequest(BaseModel):
    """聊天请求 用户发消息调用聊天接口时，前端必须传给后端的数据结构"""
    user_id: str = Field(..., description="用户唯一标识", min_length=1, max_length=64)
    #必填（... 代表必填），用户唯一 ID；长度 1~64，不能为空字符串，区分不同用户
    message: str = Field(..., description="用户消息", min_length=1, max_length=2000)
    #必填，用户输入的文字；限制最长 2000 字，防止超长文本压垮服务
    session_id: Optional[str] = Field(None, description="会话ID，不传则新建会话")
    #可选；会话标识，用来区分同用户下多轮对话上下文，不传就自动新开对话
    image_base64: Optional[str] = Field(None, description="图片base64编码（含data:image前缀）")
    #可选；图片 base64 字符串，支持多模态识图（对应上一份配置里的 analyze_image 看图工具）
    voice_mode: bool = Field(False, description="语音通话模式，启用口语化回复")
    #布尔值，默认关闭；开启后 AI 回复口语化，适配语音播报场景
    # 校验规则作用
    # 前端少传 user_id/message → 接口直接返回 422 参数错误，不用后端手动写 if 判断；
    # message 超过 2000 字会直接拦截，保护 LLM 接口。


class ChatResponse(BaseModel):
    """聊天响应 后端返回给前端的数据结构"""
    success: bool = True
    #接口状态，默认 true，代表正常响应；AI 报错时可改为 false
    content: str = Field(..., description="AI回复内容")
    #必填，AI 输出的文字回答
    session_id: str = Field(..., description="会话ID")
    #当前对话会话 ID，前端下次请求带上维持上下文
    tool_used: Optional[str] = Field(None, description="使用的工具名称")
    #本次 AI 调用了哪个工具（如 get_weather、app_click），无工具返回 null
    need_confirm: bool = False
    #关键开关：当 AI 要执行危险桌面操作（删文件、运行命令）时，设为 True，需要用户手动确认后才执行
    pending_id: Optional[str] = Field(None, description="待确认操作ID")
    #待确认操作唯一 ID；用户确认 / 拒绝时要携带这个 ID
    confirm_preview: Optional[str] = Field(None, description="操作预览描述")
    #给用户看的操作预览文案，比如「即将打开记事本并写入文本」
    timestamp: datetime = Field(default_factory=datetime.now)
    #响应生成时间，default_factory=datetime.now：实例化时自动填充当前时间


class ConfirmRequest(BaseModel):
    """确认/拒绝操作请求 用户弹窗点「确认 / 取消」时，前端传给后端的接口参数。"""
    pending_id: str = Field(..., description="待确认操作ID")
    #必填，和 ChatResponse 里的 pending_id 对应，后端找到挂起的操作；
    user_id: str = Field(..., description="用户ID")
    #校验操作归属人，防止越权；
    action: str = Field("confirm", description="confirm 或 reject")
    #只能传 confirm（执行操作）/ reject（放弃操作），默认值 confirm。

class UploadResponse(BaseModel):
    """文件上传响应 用户上传文档 / 图片后，后端返回的结果结构"""
    success: bool = True
    #上传是否成功；
    message: str
    #提示文案（如「上传成功」「文件格式不支持」）；
    filename: str
    #服务器保存后的文件名；
    chunks: int = 0
    #文档分块数量（对应 config 里 RAG 的 chunk 逻辑，PDF/Word 会切割成多段存入向量库）。


class MemoryResponse(BaseModel):
    """记忆查询响应 返回用户历史记忆"""
    success: bool = True
    memories: list[str] = []
    #字符串列表，存放检索到的用户记忆（比如「用户喜欢喝咖啡、家住天津」），无记忆返回空数组。


class HealthResponse(BaseModel):
    """健康检查响应 运维 / 前端用来检测服务、LLM、向量库是否正常的接口模型。"""
    status: str = "ok"
    #总状态，ok 代表服务存活；
    llm_ready: bool = False
    #大模型 API 是否连通可用；
    rag_ready: bool = False
    #知识库向量数据库是否正常；
    memory_ready: bool = False
    #用户长期记忆向量库是否正常。



# 整体设计目的 & 和上一份 config 代码的关联
# 接口标准化
# 所有前端和后端交互的数据都有固定结构，不会出现字段乱传、类型错乱问题，配合 FastAPI 自动生成交互式接口文档。
# 配套 AI Agent 业务流程
# 用户发消息 → ChatRequest
# AI 要操作电脑 / 删文件，需要确认 → 返回带 need_confirm=True 的 ChatResponse
# 用户点确认弹窗 → 提交 ConfirmRequest 执行工具
# 上传文档构建知识库 → 返回 UploadResponse
# 查询用户过往偏好记忆 → 返回 MemoryResponse
# 前端 / 监控检测服务是否挂了 → HealthResponse
# 约束 AI 桌面自动化风险
# 核心设计亮点：高危操作强制二次确认，通过 pending_id 串联聊天接口和确认接口，防止 AI 擅自删除文件、运行系统命令。