"""
生活管家 AI-Agent - FastAPI 应用入口
"""
import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import DATA_DIR, UPLOAD_DIR, VECTOR_DB_DIR, CHAT_IMAGES_DIR, BASE_DIR, ZHIPU_MODEL, SILICONFLOW_API_KEY, TTS_VOICE
from app.router import router
from app.llm_service import LLMService
from app.memory_service import MemoryService
from app.rag_service import RAGService
from app.tool_service import ToolService
from app.agent_graph import AgentGraph
from app.chat_service import ChatService
from app.call_service import CallService
from app.mcp_client import get_mcp_manager, close_mcp

# ==================== 日志配置 ====================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(DATA_DIR / "app.log"), encoding="utf-8"),
    ],
)
logger = logging.getLogger("life-agent")


# ==================== 应用生命周期 ====================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用启动和关闭时的初始化/清理"""
    # ---- 启动 ----
    logger.info("🚀 生活管家 AI-Agent 启动中...")

    # 确保目录存在
    for d in [DATA_DIR, UPLOAD_DIR, VECTOR_DB_DIR, CHAT_IMAGES_DIR]:
        os.makedirs(str(d), exist_ok=True)

    # 初始化各服务
    memory_service = MemoryService()
    rag_service = RAGService()
    llm_service = LLMService()
    tool_service = ToolService(memory_service, rag_service)
    tool_service.set_llm_client(llm_service.client)
    tool_service.set_zhipu_client(llm_service.zhipu_client)

    # 预加载IP定位（避免首次天气查询等待）
    ToolService.preload_location()

    # 构建 LangGraph Agent（千问用于规划/回答，智谱用于反思）
    agent_graph = AgentGraph(
        llm_service.client,
        tool_service,
        zhipu_llm=llm_service.zhipu_client,
    )
    chat_service = ChatService(agent_graph, memory_service, rag_service)
    call_service = CallService(chat_service, tool_service)

    # 初始化 MCP 客户端（连接外部 MCP Server，扩展工具集）
    try:
        mcp_manager = await get_mcp_manager()
        mcp_tools = mcp_manager.get_tool_schemas()
        if mcp_tools:
            from app.tool_schemas import TOOL_DEFINITIONS
            TOOL_DEFINITIONS.extend(mcp_tools)
            logger.info(f"  MCP 工具: ✅ 已加载 {len(mcp_tools)} 个外部工具")
        app.state.mcp_manager = mcp_manager
    except Exception as e:
        logger.warning(f"  MCP 工具: ⚠️ 初始化失败 ({e})，跳过")
        app.state.mcp_manager = None

    # 挂载到 app.state
    app.state.memory = memory_service
    app.state.rag = rag_service
    app.state.llm = llm_service
    app.state.agent_graph = agent_graph
    app.state.chat_service = chat_service
    app.state.call_service = call_service

    # 状态检查
    logger.info(f"  LLM 服务: {'✅ 就绪' if llm_service.is_ready else '❌ 未配置 API Key'}")
    if llm_service._zhipu_client:
        logger.info(f"  智谱 GLM:  ✅ 就绪 (model={ZHIPU_MODEL})")
    else:
        logger.info(f"  智谱 GLM:  ⚠️ 未配置，使用主模型兜底")
    logger.info(f"  记忆服务: ✅ 就绪")
    logger.info(f"  RAG 服务: {'✅ 就绪' if rag_service.is_ready else '⚠️ 功能降级'}")
    logger.info(f"  Agent 编排: LangGraph StateGraph")
    logger.info(f"  语音通话: {'✅ 就绪' if SILICONFLOW_API_KEY else '⚠️ 未配置 STT Key'} (TTS={TTS_VOICE})")
    logger.info("🚀 生活管家 AI-Agent 启动完成！")

    yield

    # ---- 关闭 ----
    logger.info("👋 生活管家 AI-Agent 正在关闭...")
    await close_mcp()
    memory_service.close()
    logger.info("👋 已安全关闭")


# ==================== 创建 FastAPI 应用 ====================
app = FastAPI(
    title="生活管家 AI-Agent",
    description="面向普通用户的智能生活管家，支持日程规划、收支计算、文档问答、旅行方案等",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS 中间件（允许跨域访问）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册 API 路由
app.include_router(router)

# 挂载聊天图片目录（必须在 "/" 之前注册）
if CHAT_IMAGES_DIR.exists():
    app.mount("/chat-images", StaticFiles(directory=str(CHAT_IMAGES_DIR)), name="chat-images")

# 挂载静态文件（前端页面）
static_dir = BASE_DIR / "static"
if static_dir.exists():
    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")
