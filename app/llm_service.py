"""
LLM 服务层 — ChatOpenAI 封装
提供主模型（千问）和辅助模型（智谱 GLM）双客户端实例
"""
import logging

from langchain_openai import ChatOpenAI

from app.config import (
    LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, LLM_TIMEOUT,
    ZHIPU_API_KEY, ZHIPU_BASE_URL, ZHIPU_MODEL,
)

logger = logging.getLogger(__name__)


class LLMService:
    """LLM 服务：提供双 ChatOpenAI 客户端实例（主模型 + 智谱 GLM）"""

    def __init__(self):
        # ---- 主 LLM（通义千问 qwen-plus）----
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

        # ---- 辅助 LLM（智谱 GLM，用于反思 + 翻译）----
        if ZHIPU_API_KEY:
            self._zhipu_client = ChatOpenAI(
                model=ZHIPU_MODEL,
                base_url=ZHIPU_BASE_URL,
                api_key=ZHIPU_API_KEY,
                timeout=float(LLM_TIMEOUT),
                temperature=0.3,
                max_retries=2,
            )
            logger.info(f"智谱 GLM 已配置: model={ZHIPU_MODEL}")
        else:
            self._zhipu_client = None
            logger.info("智谱 GLM 未配置，反思/翻译将使用主模型")

    @property
    def is_ready(self) -> bool:
        return self.client is not None

    @property
    def zhipu_client(self):
        """智谱 GLM 客户端；未配置时回退到主客户端"""
        return self._zhipu_client or self.client
