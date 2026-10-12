"""小规模学习版使用 sqlite3；每次操作独立连接，事务提交后自动关闭。"""
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import json
import sqlite3


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class Database:
    def __init__(self, path: Path):
        self.path = path

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.path, timeout=15)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def initialize(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("""CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY, subject_id TEXT NOT NULL,
                collection_id TEXT NOT NULL, collected_at TEXT NOT NULL,
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                status TEXT NOT NULL, request_json TEXT NOT NULL,
                files_json TEXT NOT NULL, result_json TEXT, error TEXT,
                UNIQUE(subject_id, collection_id))""")
            db.execute("CREATE INDEX IF NOT EXISTS ix_subject_time ON tasks(subject_id, collected_at)")
            # 简化恢复策略：保留失败原因，让使用者显式重试，不悄悄丢掉任务。
            db.execute("""UPDATE tasks SET status='failed', error=?, updated_at=?
                          WHERE status IN ('queued','running')""",
                       ("上次服务退出时任务未完成，请重试。", utcnow()))

    @staticmethod
    def unpack(row):
        if row is None:
            return None
        result = dict(row)
        for name in ("request", "files", "result"):
            value = result.pop(name + "_json")
            result[name] = json.loads(value) if value else None
        return result

    def get(self, task_id: str):
        with self.connect() as db:
            return self.unpack(db.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone())

    def find(self, subject_id: str, collection_id: str):
        with self.connect() as db:
            return self.unpack(db.execute("SELECT * FROM tasks WHERE subject_id=? AND collection_id=?",
                                         (subject_id, collection_id)).fetchone())

    def create(self, task_id: str, payload: dict, files: dict):
        now = utcnow()
        with self.connect() as db:
            db.execute("""INSERT INTO tasks
                (id,subject_id,collection_id,collected_at,created_at,updated_at,status,request_json,files_json)
                VALUES (?,?,?,?,?,?,'queued',?,?)""",
                       (task_id, payload["subject_id"], payload["collection_id"], payload["collected_at"],
                        now, now, json.dumps(payload, ensure_ascii=False), json.dumps(files)))

    def update(self, task_id: str, status: str, result=None, error=None):
        with self.connect() as db:
            db.execute("UPDATE tasks SET status=?,updated_at=?,result_json=?,error=? WHERE id=?",
                       (status, utcnow(), json.dumps(result, ensure_ascii=False, allow_nan=False)
                        if result is not None else None, error, task_id))

    def list(self, subject_id=None, start=None, end=None, limit=200):
        clauses, args = [], []
        for condition, value in (("subject_id=?", subject_id), ("collected_at>=?", start),
                                 ("collected_at<?", end)):
            if value is not None:
                clauses.append(condition)
                args.append(value)
        query = "SELECT * FROM tasks" + (" WHERE " + " AND ".join(clauses) if clauses else "")
        query += " ORDER BY collected_at DESC, created_at DESC LIMIT ?"
        with self.connect() as db:
            return [self.unpack(row) for row in db.execute(query, [*args, limit]).fetchall()]

    def history(self, subject_id: str, before: str):
        with self.connect() as db:
            rows = db.execute("""SELECT * FROM tasks WHERE subject_id=? AND collected_at<?
                                 AND status='succeeded' ORDER BY collected_at ASC LIMIT 1000""",
                              (subject_id, before)).fetchall()
        return [{"collected_at": row["collected_at"], **json.loads(row["result_json"])} for row in rows]
