"""
用户记忆服务 - 异步 SQLite + 向量语义搜索，并发安全
支持:
  - 用户偏好/习惯记忆（SQLite 持久 + ChromaDB 语义索引）
  - 对话历史记录
  - 会话管理
  - 多用户并发安全（asyncio.Lock + WAL 模式）
"""
import asyncio
import logging
import sqlite3
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import partial
from typing import Optional

from app.config import (
    MEMORY_DB_PATH, MAX_HISTORY_MESSAGES,
    LLM_API_KEY, VECTOR_DB_DIR,
    MEMORY_VECTOR_COLLECTION, MEMORY_VECTOR_TOP_K,
    SILICONFLOW_API_KEY,
)

logger = logging.getLogger(__name__)

# 线程池：用于将同步 SQLite 操作转为异步
_executor = ThreadPoolExecutor(max_workers=4)


class MemoryService:
    """异步记忆服务：SQLite 持久 + ChromaDB 语义搜索"""

    def __init__(self):
        self._write_lock = asyncio.Lock()  # 写操作全局锁
        self._init_db()
        # --- 向量搜索索引（懒加载，同 RAGService 模式）---
        self._vectorstore = None
        self._embedding = None
        self._vec_initialized = False
        self._vec_init_attempted = False
        logger.info("记忆服务初始化完成（异步模式）")

    @property
    def is_ready(self) -> bool:
        return True

    def _get_conn(self) -> sqlite3.Connection:
        """每次操作获取新连接，WAL 模式支持并发读"""
        conn = sqlite3.connect(str(MEMORY_DB_PATH), timeout=15)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    def _init_db(self):
        conn = self._get_conn()
        try:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);

                CREATE TABLE IF NOT EXISTS chat_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                );
                CREATE INDEX IF NOT EXISTS idx_history_session ON chat_history(session_id);

                CREATE TABLE IF NOT EXISTS user_memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    mem_key TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(user_id, mem_key)
                );
                CREATE INDEX IF NOT EXISTS idx_memory_user ON user_memory(user_id);
            """)
            conn.commit()
        finally:
            conn.close()

    # ==================== 向量索引（懒加载）====================

    def _lazy_init_vectorstore(self):
        """懒加载向量索引：首次语义搜索时初始化"""
        if self._vec_init_attempted:
            return
        self._vec_init_attempted = True

        if not SILICONFLOW_API_KEY and not LLM_API_KEY:
            logger.warning("记忆向量: 未配置任何 API Key，语义搜索不可用")
            return

        try:
            from langchain_chroma import Chroma
            from app.rag_service import SiliconFlowEmbeddings, DashScopeEmbeddings

            if SILICONFLOW_API_KEY:
                self._embedding = SiliconFlowEmbeddings(
                    api_key=SILICONFLOW_API_KEY,
                    model="BAAI/bge-large-zh-v1.5",
                )
                logger.info("记忆向量: SiliconFlow (BAAI/bge-large-zh-v1.5)")
            else:
                self._embedding = DashScopeEmbeddings(
                    api_key=LLM_API_KEY,
                    model="text-embedding-v3",
                )
                logger.info("记忆向量: DashScope (text-embedding-v3)")

            self._vectorstore = Chroma(
                collection_name=MEMORY_VECTOR_COLLECTION,
                embedding_function=self._embedding,
                persist_directory=str(VECTOR_DB_DIR),
            )

            # 如果集合为空，从 SQLite 批量加载已有记忆
            self._sync_bulk_from_sqlite()

            self._vec_initialized = True
            logger.info("记忆向量索引初始化完成 (DashScope Embedding)")
        except Exception as e:
            logger.warning(f"记忆向量索引初始化失败: {e}")
            self._vec_initialized = False

    def _sync_bulk_from_sqlite(self):
        """如果向量集合为空，从 SQLite 批量加载所有记忆"""
        try:
            collection = self._vectorstore._collection
            if collection.count() > 0:
                return  # 已有数据，跳过
        except Exception:
            pass

        conn = self._get_conn()
        try:
            rows = conn.execute(
                "SELECT user_id, mem_key, content FROM user_memory"
            ).fetchall()
        finally:
            conn.close()

        if not rows:
            return

        # 批量向量化存入（DashScope 每批最多 25 条）
        texts = [f"[{r['mem_key']}] {r['content']}" for r in rows]
        metadatas = [{"user_id": r["user_id"], "mem_key": r["mem_key"]} for r in rows]
        ids = [f"mem_{r['user_id']}_{r['mem_key']}" for r in rows]

        self._vectorstore.add_texts(texts=texts, metadatas=metadatas, ids=ids)
        logger.info(f"记忆向量: 从 SQLite 批量同步 {len(rows)} 条记忆")

    def _vector_add(self, user_id: str, key: str, content: str):
        """保存单条记忆到向量索引"""
        if not self._vec_initialized:
            return
        try:
            text = f"[{key}] {content}"
            doc_id = f"mem_{user_id}_{key}"
            self._vectorstore.add_texts(
                texts=[text],
                metadatas=[{"user_id": user_id, "mem_key": key}],
                ids=[doc_id],
            )
        except Exception as e:
            logger.warning(f"记忆向量写入失败: {e}")

    def _vector_delete(self, user_id: str, key: str):
        """从向量索引删除单条记忆"""
        if not self._vec_initialized:
            return
        try:
            doc_id = f"mem_{user_id}_{key}"
            self._vectorstore._collection.delete(ids=[doc_id])
        except Exception as e:
            logger.warning(f"记忆向量删除失败: {e}")

    # ==================== 异步执行辅助 ====================

    async def _run_read(self, func, *args):
        """异步执行读操作（不锁，可并发）"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_executor, partial(func, *args))

    async def _run_write(self, func, *args):
        """异步执行写操作（加锁，串行化写入）"""
        async with self._write_lock:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(_executor, partial(func, *args))

    # ==================== 会话管理 ====================

    def _get_or_create_session_sync(self, user_id: str, session_id: Optional[str]) -> str:
        conn = self._get_conn()
        try:
            now = datetime.now().isoformat()
            if session_id:
                row = conn.execute(
                    "SELECT session_id FROM sessions WHERE session_id = ? AND user_id = ?",
                    (session_id, user_id),
                ).fetchone()
                if row:
                    conn.execute("UPDATE sessions SET updated_at = ? WHERE session_id = ?", (now, session_id))
                    conn.commit()
                    return session_id
            new_sid = session_id or str(uuid.uuid4())
            conn.execute(
                "INSERT OR IGNORE INTO sessions (session_id, user_id, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (new_sid, user_id, now, now),
            )
            conn.commit()
            return new_sid
        finally:
            conn.close()

    def _get_user_sessions_sync(self, user_id: str, limit: int = 10) -> list[dict]:
        conn = self._get_conn()
        try:
            rows = conn.execute(
                "SELECT session_id, created_at, updated_at FROM sessions WHERE user_id = ? ORDER BY updated_at DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    async def get_or_create_session(self, user_id: str, session_id: Optional[str] = None) -> str:
        return await self._run_write(self._get_or_create_session_sync, user_id, session_id)

    async def get_user_sessions(self, user_id: str, limit: int = 10) -> list[dict]:
        return await self._run_read(self._get_user_sessions_sync, user_id, limit)

    # ==================== 对话历史 ====================

    def _save_message_sync(self, session_id: str, role: str, content: str):
        conn = self._get_conn()
        try:
            now = datetime.now().isoformat()
            conn.execute(
                "INSERT INTO chat_history (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                (session_id, role, content, now),
            )
            conn.commit()
        finally:
            conn.close()

    def _get_history_sync(self, session_id: str, limit: int) -> list[dict]:
        conn = self._get_conn()
        try:
            rows = conn.execute(
                "SELECT role, content FROM chat_history WHERE session_id = ? ORDER BY id DESC LIMIT ?",
                (session_id, limit),
            ).fetchall()
            return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]
        finally:
            conn.close()

    def _get_full_history_sync(self, session_id: str) -> list[dict]:
        conn = self._get_conn()
        try:
            rows = conn.execute(
                "SELECT role, content, created_at FROM chat_history WHERE session_id = ? ORDER BY id ASC",
                (session_id,),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    async def save_message(self, session_id: str, role: str, content: str):
        await self._run_write(self._save_message_sync, session_id, role, content)

    async def get_history(self, session_id: str, limit: int = MAX_HISTORY_MESSAGES) -> list[dict]:
        return await self._run_read(self._get_history_sync, session_id, limit)

    async def get_full_history(self, session_id: str) -> list[dict]:
        return await self._run_read(self._get_full_history_sync, session_id)

    # ==================== 用户记忆 ====================

    def _save_memory_sync(self, user_id: str, key: str, content: str):
        conn = self._get_conn()
        try:
            now = datetime.now().isoformat()
            conn.execute(
                """INSERT INTO user_memory (user_id, mem_key, content, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(user_id, mem_key) DO UPDATE SET content = ?, updated_at = ?""",
                (user_id, key, content, now, now, content, now),
            )
            conn.commit()
        finally:
            conn.close()

    def _get_memories_sync(self, user_id: str) -> list[dict]:
        conn = self._get_conn()
        try:
            rows = conn.execute(
                "SELECT mem_key, content, updated_at FROM user_memory WHERE user_id = ? ORDER BY updated_at DESC",
                (user_id,),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def _search_memories_sync(self, user_id: str, keyword: str, limit: int = 5) -> list[str]:
        conn = self._get_conn()
        try:
            rows = conn.execute(
                """SELECT mem_key, content FROM user_memory
                   WHERE user_id = ? AND (mem_key LIKE ? OR content LIKE ?)
                   ORDER BY updated_at DESC LIMIT ?""",
                (user_id, f"%{keyword}%", f"%{keyword}%", limit),
            ).fetchall()
            return [f"[{r['mem_key']}] {r['content']}" for r in rows]
        finally:
            conn.close()

    def _delete_memory_sync(self, user_id: str, key: str):
        conn = self._get_conn()
        try:
            conn.execute(
                "DELETE FROM user_memory WHERE user_id = ? AND mem_key = ?",
                (user_id, key),
            )
            conn.commit()
        finally:
            conn.close()

    async def save_memory(self, user_id: str, key: str, content: str):
        await self._run_write(self._save_memory_sync, user_id, key, content)
        # 同步写入向量索引
        self._vector_add(user_id, key, content)

    async def get_memories(self, user_id: str) -> list[dict]:
        return await self._run_read(self._get_memories_sync, user_id)

    async def search_memories(self, user_id: str, keyword: str, limit: int = 5) -> list[str]:
        return await self._run_read(self._search_memories_sync, user_id, keyword, limit)

    async def search_memories_semantic(self, user_id: str, query: str, limit: int = MEMORY_VECTOR_TOP_K) -> list[str]:
        """语义搜索用户记忆（基于向量相似度），失败自动回退到关键词搜索"""
        # 懒加载
        if not self._vec_initialized and not self._vec_init_attempted:
            self._lazy_init_vectorstore()

        if self._vec_initialized:
            try:
                results = self._vectorstore.similarity_search(
                    query,
                    k=limit,
                    filter={"user_id": user_id},
                )
                return [doc.page_content for doc in results]
            except Exception as e:
                logger.warning(f"语义搜索失败，回退到关键词搜索: {e}")

        # 向量索引不可用，回退到 SQL LIKE
        return await self.search_memories(user_id, query, limit)

    async def delete_memory(self, user_id: str, key: str):
        """删除记忆（SQLite + 向量索引同步清理）"""
        await self._run_write(self._delete_memory_sync, user_id, key)
        self._vector_delete(user_id, key)

    async def build_memory_context(self, user_id: str, query: str = "") -> str:
        """
        构建记忆上下文。
        有 query 时返回语义相关的 top-K 记忆；无 query 时返回全部记忆。
        向量相关操作丢到线程池，避免阻塞事件循环。
        """
        if query:
            # 懒加载 + 检索都是同步阻塞调用，放线程池
            def _semantic_search():
                if not self._vec_initialized and not self._vec_init_attempted:
                    self._lazy_init_vectorstore()
                if self._vec_initialized:
                    try:
                        return self._vectorstore.similarity_search(
                            query,
                            k=MEMORY_VECTOR_TOP_K,
                            filter={"user_id": user_id},
                        )
                    except Exception as e:
                        logger.warning(f"记忆向量检索失败，回退到全量: {e}")
                return None

            try:
                loop = asyncio.get_event_loop()
                results = await loop.run_in_executor(_executor, _semantic_search)
                if results:
                    lines = ["\n## 用户记忆（相关）"]
                    for doc in results:
                        lines.append(f"- {doc.page_content}")
                    return "\n".join(lines)
            except Exception as e:
                logger.warning(f"记忆上下文构建失败（回退全量）: {e}")

        # 无 query 或向量搜索失败：返回全部记忆（向后兼容）
        memories = await self.get_memories(user_id)
        if not memories:
            return ""
        lines = ["\n## 用户记忆"]
        for m in memories:
            lines.append(f"- **{m['mem_key']}**: {m['content']}")
        return "\n".join(lines)

    def close(self):
        """清理资源"""
        if self._vec_initialized:
            logger.info("记忆向量索引已关闭")
        _executor.shutdown(wait=False)
