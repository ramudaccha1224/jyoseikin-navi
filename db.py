"""
db.py  –  SQLite CRUD 全般
テーブル: users / conversations / messages
"""
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta

# .env または環境変数から取得。デフォルトは data/app.db
_BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATABASE_PATH = os.getenv("DATABASE_PATH", os.path.join(_BASE_DIR, "data", "app.db"))


@contextmanager
def get_conn():
    """SQLite接続のコンテキストマネージャー（自動コミット・ロールバック・クローズ）"""
    os.makedirs(os.path.dirname(DATABASE_PATH), exist_ok=True)
    conn = sqlite3.connect(DATABASE_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# =============================================================
# テーブル作成（アプリ起動時に1回呼び出す）
# =============================================================
def create_tables():
    with get_conn() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            username      TEXT    NOT NULL UNIQUE,
            display_name  TEXT    NOT NULL,
            password_hash TEXT    NOT NULL,
            is_admin      INTEGER NOT NULL DEFAULT 0,
            is_active     INTEGER NOT NULL DEFAULT 1,
            created_at    TEXT    NOT NULL DEFAULT (datetime('now', 'localtime')),
            last_login_at TEXT
        );

        CREATE TABLE IF NOT EXISTS conversations (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            title      TEXT    NOT NULL DEFAULT '無題の会話',
            domain_key TEXT    NOT NULL DEFAULT '',
            form_name  TEXT    NOT NULL DEFAULT '',
            created_at TEXT    NOT NULL DEFAULT (datetime('now', 'localtime')),
            updated_at TEXT    NOT NULL DEFAULT (datetime('now', 'localtime'))
        );

        CREATE TABLE IF NOT EXISTS messages (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
            role            TEXT    NOT NULL CHECK(role IN ('user', 'assistant')),
            content         TEXT    NOT NULL,
            created_at      TEXT    NOT NULL DEFAULT (datetime('now', 'localtime'))
        );

        CREATE INDEX IF NOT EXISTS idx_conv_user    ON conversations(user_id);
        CREATE INDEX IF NOT EXISTS idx_conv_updated ON conversations(updated_at);
        CREATE INDEX IF NOT EXISTS idx_msg_conv     ON messages(conversation_id);
        """)


# =============================================================
# ユーザー関連
# =============================================================
def get_user_by_username(username: str) -> dict | None:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE username = ?", (username,)
        ).fetchone()
    return dict(row) if row else None


def get_user_by_id(user_id: int) -> dict | None:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE id = ?", (user_id,)
        ).fetchone()
    return dict(row) if row else None


def get_all_users() -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM users ORDER BY created_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def create_user(username: str, display_name: str, password_hash: str, is_admin: bool = False) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO users (username, display_name, password_hash, is_admin) VALUES (?,?,?,?)",
            (username, display_name, password_hash, int(is_admin)),
        )
    return cur.lastrowid


def update_password(user_id: int, new_hash: str) -> None:
    with get_conn() as conn:
        conn.execute(
            "UPDATE users SET password_hash = ? WHERE id = ?",
            (new_hash, user_id),
        )


def set_user_active(user_id: int, is_active: bool) -> None:
    with get_conn() as conn:
        conn.execute(
            "UPDATE users SET is_active = ? WHERE id = ?",
            (int(is_active), user_id),
        )


def delete_user(user_id: int) -> None:
    with get_conn() as conn:
        conn.execute("DELETE FROM users WHERE id = ?", (user_id,))


def update_last_login(user_id: int) -> None:
    with get_conn() as conn:
        conn.execute(
            "UPDATE users SET last_login_at = datetime('now','localtime') WHERE id = ?",
            (user_id,),
        )


def get_all_user_stats() -> list[dict]:
    """全ユーザーの利用統計（管理画面用）"""
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT
                u.id,
                u.username,
                u.display_name,
                u.is_active,
                u.last_login_at,
                COUNT(DISTINCT c.id)  AS total_conversations,
                COUNT(m.id)           AS total_messages
            FROM users u
            LEFT JOIN conversations c ON c.user_id = u.id
            LEFT JOIN messages m      ON m.conversation_id = c.id
            GROUP BY u.id
            ORDER BY u.created_at DESC
        """).fetchall()
    return [dict(r) for r in rows]


# =============================================================
# 会話スレッド関連
# =============================================================
def create_conversation(user_id: int, domain_key: str, form_name: str, title: str = "無題の会話") -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO conversations (user_id, domain_key, form_name, title) VALUES (?,?,?,?)",
            (user_id, domain_key, form_name, title),
        )
    return cur.lastrowid


def get_conversations_by_user(user_id: int, limit: int = 20, offset: int = 0) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT * FROM conversations
               WHERE user_id = ?
               ORDER BY updated_at DESC
               LIMIT ? OFFSET ?""",
            (user_id, limit, offset),
        ).fetchall()
    return [dict(r) for r in rows]


def get_conversation(conv_id: int) -> dict | None:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM conversations WHERE id = ?", (conv_id,)
        ).fetchone()
    return dict(row) if row else None


def update_conversation_title(conv_id: int, title: str) -> None:
    with get_conn() as conn:
        conn.execute(
            "UPDATE conversations SET title = ? WHERE id = ?",
            (title, conv_id),
        )


def touch_conversation(conv_id: int) -> None:
    """updated_at を現在時刻に更新（スレッド一覧のソート用）"""
    with get_conn() as conn:
        conn.execute(
            "UPDATE conversations SET updated_at = datetime('now','localtime') WHERE id = ?",
            (conv_id,),
        )


def delete_old_conversations(days: int = 90) -> int:
    """updated_at が days 日以上前の会話を削除（messages は CASCADE で連鎖削除）"""
    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
    with get_conn() as conn:
        cur = conn.execute(
            "DELETE FROM conversations WHERE updated_at < ?", (cutoff,)
        )
    return cur.rowcount


# =============================================================
# メッセージ関連
# =============================================================
def add_message(conv_id: int, role: str, content: str) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO messages (conversation_id, role, content) VALUES (?,?,?)",
            (conv_id, role, content),
        )
    return cur.lastrowid


def get_messages_by_conversation(conv_id: int) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM messages WHERE conversation_id = ? ORDER BY id ASC",
            (conv_id,),
        ).fetchall()
    return [dict(r) for r in rows]
