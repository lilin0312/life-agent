"""
RAG 服务 - 文档上传、向量化、检索
基于 ChromaDB + DashScope Embedding API（国内直连，无需 HuggingFace）
"""
import logging
import os
from pathlib import Path
from typing import Optional

from app.config import (
    UPLOAD_DIR, VECTOR_DB_DIR, CHUNK_SIZE,
    CHUNK_OVERLAP, RAG_TOP_K, LLM_API_KEY,
    SILICONFLOW_API_KEY,
)

logger = logging.getLogger(__name__)


# ==================== 轻量级 Embedding 实现 ====================

class SiliconFlowEmbeddings:
    """基于硅基流动的 Embedding（OpenAI 兼容 API）"""

    def __init__(self, api_key: str, model: str = "BAAI/bge-large-zh-v1.5"):
        self.api_key = api_key
        self.model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        import httpx
        results = []
        batch_size = 16
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            resp = httpx.post(
                "https://api.siliconflow.cn/v1/embeddings",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={"model": self.model, "input": batch},
                timeout=15,
            )
            data = resp.json()
            if "data" in data:
                for item in data["data"]:
                    results.append(item["embedding"])
            else:
                raise RuntimeError(f"SiliconFlow embedding 失败: {data}")
        return results

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class DashScopeEmbeddings:
    """
    基于 DashScope Text Embedding API 的轻量 Embedding 实现
    无需下载本地模型，国内直连
    模型: text-embedding-v3 (dashscope)
    """

    def __init__(self, api_key: str, model: str = "text-embedding-v3"):
        self.api_key = api_key
        self.model = model
        self._dimension = 1024

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """批量文本向量化"""
        import httpx
        results = []
        # dashscope 批量 embedding，每批最多25条
        batch_size = 25
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            resp = httpx.post(
                "https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "input": {"texts": batch},
                    "parameters": {"dimension": self._dimension, "output_type": "floats"},
                },
                timeout=8,
            )
            data = resp.json()
            if "output" in data and "embeddings" in data["output"]:
                for emb in data["output"]["embeddings"]:
                    results.append(emb["embedding"])
            else:
                raise RuntimeError(f"DashScope embedding 失败: {data.get('message', data)}")
        return results

    def embed_query(self, text: str) -> list[float]:
        """单条文本向量化"""
        return self.embed_documents([text])[0]


# ==================== RAG 服务主体 ====================
class RAGService:
    """文档 RAG 服务 - 懒加载，不阻塞服务启动"""

    def __init__(self):
        self._vectorstore = None
        self._embedding = None
        self._initialized = False
        self._init_attempted = False
        os.makedirs(str(VECTOR_DB_DIR), exist_ok=True)
        os.makedirs(str(UPLOAD_DIR), exist_ok=True)
        logger.info("RAG 服务已创建（懒加载模式）")

    @property
    def is_ready(self) -> bool:
        if not self._initialized and not self._init_attempted:
            self._lazy_init()
        return self._initialized

    def _lazy_init(self):
        """懒加载：首次使用时初始化。优先硅基流动，DashScope 兜底"""
        if self._init_attempted:
            return
        self._init_attempted = True

        if not SILICONFLOW_API_KEY and not LLM_API_KEY:
            logger.warning("RAG: 未配置任何 API Key，文档检索不可用")
            return

        try:
            from langchain_chroma import Chroma

            # 优先硅基流动（key 已验证可用）
            if SILICONFLOW_API_KEY:
                self._embedding = SiliconFlowEmbeddings(
                    api_key=SILICONFLOW_API_KEY,
                    model="BAAI/bge-large-zh-v1.5",
                )
                logger.info("RAG Embedding: SiliconFlow (BAAI/bge-large-zh-v1.5)")
            else:
                self._embedding = DashScopeEmbeddings(
                    api_key=LLM_API_KEY,
                    model="text-embedding-v3",
                )
                logger.info("RAG Embedding: DashScope (text-embedding-v3)")

            self._vectorstore = Chroma(
                collection_name="life_agent_docs",
                embedding_function=self._embedding,
                persist_directory=str(VECTOR_DB_DIR),
            )
            self._initialized = True
            logger.info("RAG 服务初始化完成 (DashScope Embedding)")

        except Exception as e:
            logger.warning(f"RAG 初始化失败（功能不可用）: {e}")
            self._initialized = False

    async def upload_document(
        self, user_id: str, filename: str, file_content: bytes
    ) -> dict:
        """
        上传文档并建立向量索引
        支持 .txt, .md, .csv, .json 文件
        """
        if not self._initialized:
            return {"success": False, "message": "RAG 服务未初始化", "chunks": 0}

        # 保存文件
        safe_name = f"{user_id}_{filename}"
        file_path = UPLOAD_DIR / safe_name
        with open(file_path, "wb") as f:
            f.write(file_content)

        # 读取文本
        try:
            text = file_content.decode("utf-8", errors="ignore")
        except Exception:
            text = file_content.decode("gbk", errors="ignore")

        if not text.strip():
            return {"success": False, "message": "文件内容为空", "chunks": 0}

        # 分块
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", "。", "！", "？", ".", " ", ""],
        )
        chunks = splitter.create_documents(
            [text],
            metadatas=[{"user_id": user_id, "filename": filename}],
        )

        if not chunks:
            return {"success": False, "message": "文档分块失败", "chunks": 0}

        # 存入向量数据库
        try:
            self._vectorstore.add_documents(chunks)
            logger.info(f"文档 {filename} 已索引 ({len(chunks)} 个分块)")
            return {
                "success": True,
                "message": f"文档上传成功，已建立 {len(chunks)} 个知识分块",
                "chunks": len(chunks),
            }
        except Exception as e:
            logger.error(f"向量存储失败: {e}")
            return {"success": False, "message": f"索引建立失败: {e}", "chunks": 0}

    def search(self, user_id: str, query: str, top_k: int = RAG_TOP_K) -> list[str]:
        """在用户文档中检索相关内容"""
        if not self._initialized:
            return []

        try:
            results = self._vectorstore.similarity_search(
                query,
                k=top_k,
                filter={"user_id": user_id},
            )
            return [doc.page_content for doc in results]
        except Exception as e:
            logger.error(f"RAG 检索失败: {e}")
            return []

    def build_rag_context(self, user_id: str, query: str) -> str:
        """构建 RAG 上下文文本"""
        docs = self.search(user_id, query)
        if not docs:
            return ""
        lines = ["\n## 用户文档相关内容"]
        for i, doc in enumerate(docs, 1):
            lines.append(f"[文档{i}] {doc[:300]}")
        return "\n".join(lines)
