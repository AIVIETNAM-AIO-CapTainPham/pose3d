"""
Blake2b deduplication backed by a lightweight SQLite store.

Usage:
    store = HashStore("hashes.db")
    if store.is_new(raw_bytes):
        store.add(raw_bytes)
        # ... process image
    store.close()
"""

import hashlib
import sqlite3
from pathlib import Path


def _hash(data: bytes) -> str:
    return hashlib.blake2b(data, digest_size=20).hexdigest()


class HashStore:
    def __init__(self, db_path: str):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS seen (hash TEXT PRIMARY KEY)"
        )
        self.conn.commit()

    def is_new(self, data: bytes) -> bool:
        h = _hash(data)
        row = self.conn.execute(
            "SELECT 1 FROM seen WHERE hash = ?", (h,)
        ).fetchone()
        return row is None

    def add(self, data: bytes) -> str:
        h = _hash(data)
        self.conn.execute(
            "INSERT OR IGNORE INTO seen (hash) VALUES (?)", (h,)
        )
        self.conn.commit()
        return h

    def count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM seen").fetchone()[0]

    def close(self):
        self.conn.close()
