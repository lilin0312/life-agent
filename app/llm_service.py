"""
LLM 服务层 — ChatOpenAI 封装
工具编排已迁移到 LangGraph AgentGraph，本模块仅提供 LLM 客户端实例
"""
import logging

from langchain_openai import ChatOpenAI

from app.config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, LLM_TIMEOUT, MAX_CONCURRENT_LLM

logger = logging.getLogger(__name__)


class LLMService:
    """LLM 服务：提供 ChatOpenAI 客户端实例"""

    def __init__(self):
        if not LLM_API_KEY:
            logger.warning("API Key 未设置，LLM 服务不可用")
            self.client = None
        else:
            self.client = ChatOpenAI(
                model=LLM_MODEL,
                base_url=LLM_BASE_URL,
                api_key=LLM_API_KEY,
                timeout=float(LLM_TIMEOUT),
                temperature=0.3,
                max_retries=2,
            )

    @property
    def is_ready(self) -> bool:
        return self.client is not None
