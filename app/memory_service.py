"""
用户记忆服务 - 异步 SQLite，并发安全
支持:
  - 用户偏好/习惯记忆
  - 对话历史记录
  - 会话管理
  - 多用户并发安全（asyncio.Lock + WAL 模式）
"""
import asyncio
import json
import logging
import sqlite3
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import partial
from typing import Optional

from app.config import MEMORY_DB_PATH, MAX_HISTORY_MESSAGES

logger = logging.getLogger(__name__)

# 线程池：用于将同步 SQLite 操作转为异步
_executor = ThreadPoolExecutor(max_workers=4)


class MemoryService:
    """异步记忆服务：线程安全 + 并发友好"""

    def __init__(self):
        self._write_lock = asyncio.Lock()  # 写操作全局锁
        self._init_db()
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

    async def save_memory(self, user_id: str, key: str, content: str):
        await self._run_write(self._save_memory_sync, user_id, key, content)

    async def get_memories(self, user_id: str) -> list[dict]:
        return await self._run_read(self._get_memories_sync, user_id)

    async def search_memories(self, user_id: str, keyword: str, limit: int = 5) -> list[str]:
        return await self._run_read(self._search_memories_sync, user_id, keyword, limit)

    async def build_memory_context(self, user_id: str) -> str:
        memories = await self.get_memories(user_id)
        if not memories:
            return ""
        lines = ["\n## 用户记忆"]
        for m in memories:
            lines.append(f"- **{m['mem_key']}**: {m['content']}")
        return "\n".join(lines)

    def close(self):
        """清理资源"""
        _executor.shutdown(wait=False)
