
import sqlite3
from pathlib import Path
from datetime import datetime

# Database location
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "memory.db"


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def initialize_database():

    with get_connection() as conn:

        conn.execute("PRAGMA journal_mode = WAL")

        conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                started_at TEXT NOT NULL
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TEXT NOT NULL,

                FOREIGN KEY (conversation_id)
                REFERENCES conversations(id)
            )
        """)

    print("FRIDAY Memory Database Ready!")


def start_conversation():

    with get_connection() as conn:

        cursor = conn.execute("""
            INSERT INTO conversations (started_at)
            VALUES (?)
        """, (datetime.now().isoformat(),))

        return cursor.lastrowid


def save_message(conversation_id, role, content):

    if not content or not content.strip():
        return

    with get_connection() as conn:

        conn.execute("""
            INSERT INTO messages
            (conversation_id, role, content, timestamp)
            VALUES (?, ?, ?, ?)
        """, (
            conversation_id,
            role,
            content.strip(),
            datetime.now().isoformat()
        ))

    print(f"[MEMORY] Saved {role} message")
