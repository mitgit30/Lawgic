import json
import os
import sqlite3
from pathlib import Path
from uuid import uuid4


class LocalChatDB:
    def __init__(self, db_path: str = "misc/chat_history.db") -> None:
        self.db_path = Path(os.getenv("CHAT_DB_PATH", db_path))
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    email TEXT,
                    name TEXT,
                    picture TEXT,
                    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chats (
                    id TEXT PRIMARY KEY,
                    user_id TEXT,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(user_id) REFERENCES users(id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    text TEXT NOT NULL,
                    retrieved_context_json TEXT,
                    citations_json TEXT,
                    created_at TEXT NOT NULL DEFAULT (datetime('now')),
                    FOREIGN KEY(chat_id) REFERENCES chats(id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_documents (
                    chat_id TEXT PRIMARY KEY,
                    upload_response_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
                    FOREIGN KEY(chat_id) REFERENCES chats(id)
                )
                """
            )

            # Lightweight migration for existing DBs that do not have user_id.
            columns = conn.execute("PRAGMA table_info(chats)").fetchall()
            column_names = {col["name"] for col in columns}
            if "user_id" not in column_names:
                conn.execute("ALTER TABLE chats ADD COLUMN user_id TEXT")
                conn.execute("UPDATE chats SET user_id = 'legacy' WHERE user_id IS NULL OR user_id = ''")

    def upsert_user(self, *, user_id: str, email: str | None, name: str | None, picture: str | None) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO users(id, email, name, picture, updated_at)
                VALUES (?, ?, ?, ?, datetime('now'))
                ON CONFLICT(id) DO UPDATE SET
                    email = excluded.email,
                    name = excluded.name,
                    picture = excluded.picture,
                    updated_at = datetime('now')
                """,
                (user_id, email, name, picture),
            )

    def list_chats(self, user_id: str) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT c.id, c.title, c.created_at, d.upload_response_json
                FROM chats c
                LEFT JOIN chat_documents d ON d.chat_id = c.id
                WHERE c.user_id = ?
                ORDER BY c.created_at ASC
                """,
                (user_id,),
            ).fetchall()
        result = []
        for row in rows:
            upload_response = None
            if row["upload_response_json"]:
                upload_response = json.loads(row["upload_response_json"])
            result.append(
                {
                    "id": row["id"],
                    "title": row["title"],
                    "created_at": row["created_at"],
                    "messages": [],
                    "upload_response": upload_response,
                }
            )
        return result

    def load_messages(self, chat_id: str) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT role, text, retrieved_context_json, citations_json
                FROM messages
                WHERE chat_id = ?
                ORDER BY id ASC
                """,
                (chat_id,),
            ).fetchall()

        messages = []
        for row in rows:
            msg = {"role": row["role"], "text": row["text"]}
            if row["retrieved_context_json"]:
                msg["retrieved_context"] = json.loads(row["retrieved_context_json"])
            if row["citations_json"]:
                msg["citations"] = json.loads(row["citations_json"])
            messages.append(msg)
        return messages

    def create_chat(self, title: str, *, user_id: str) -> dict:
        chat_id = str(uuid4())
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO chats(id, user_id, title, created_at) VALUES(?, ?, ?, datetime('now'))",
                (chat_id, user_id, title),
            )
            row = conn.execute(
                "SELECT id, title, created_at FROM chats WHERE id = ? AND user_id = ?",
                (chat_id, user_id),
            ).fetchone()
        return {
            "id": row["id"],
            "title": row["title"],
            "created_at": row["created_at"],
            "messages": [],
            "upload_response": None,
        }

    def save_message(
        self,
        *,
        chat_id: str,
        role: str,
        text: str,
        retrieved_context: list[dict] | None = None,
        citations: list[dict] | None = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO messages(chat_id, role, text, retrieved_context_json, citations_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    chat_id,
                    role,
                    text,
                    json.dumps(retrieved_context or []),
                    json.dumps(citations or []),
                ),
            )

    def save_upload_response(self, *, chat_id: str, upload_response: dict) -> None:
        payload = json.dumps(upload_response)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO chat_documents(chat_id, upload_response_json, updated_at)
                VALUES (?, ?, datetime('now'))
                ON CONFLICT(chat_id) DO UPDATE SET
                    upload_response_json = excluded.upload_response_json,
                    updated_at = datetime('now')
                """,
                (chat_id, payload),
            )

    def delete_chat(self, chat_id: str, *, user_id: str) -> None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id FROM chats WHERE id = ? AND user_id = ?",
                (chat_id, user_id),
            ).fetchone()
            if not row:
                return
            conn.execute("DELETE FROM messages WHERE chat_id = ?", (chat_id,))
            conn.execute("DELETE FROM chat_documents WHERE chat_id = ?", (chat_id,))
            conn.execute("DELETE FROM chats WHERE id = ? AND user_id = ?", (chat_id, user_id))
