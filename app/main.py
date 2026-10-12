from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
import csv
import io
import shutil
from uuid import uuid4
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import Response
from pydantic import ValidationError
from app.config import load_settings
from app.contracts import Submission
from app.db import Database
from app.pipeline import Pipeline

EXTENSIONS = {"eeg": {".edf"}, "audio": {".wav", ".ogg", ".flac"},
              "face": {".png", ".jpg", ".jpeg"}}


def iso_utc(value):
    if value is None:
        return None
    if value.tzinfo is None or value.utcoffset() is None:
        raise HTTPException(422, "时间必须带时区")
    return value.astimezone(timezone.utc).isoformat()


def public_task(task, detail=True):
    # 文件路径、原始文本不出现在列表或导出里；学习版仍仅供本机使用。
    result = {k: v for k, v in task.items() if k not in ("request", "files")}
    result["source"] = task["request"]["source"]
    if not detail and result["result"]:
        result["result"] = {k: v for k, v in result["result"].items() if k != "features"}
    return result


def create_app(settings=None):
    settings = settings or load_settings()
    db = Database(settings.data_dir / "mindtrack.sqlite3")

    @asynccontextmanager
    async def lifespan(app):
        db.initialize()
        app.state.pipeline = Pipeline(settings, db)
        yield
        app.state.pipeline.close()

    app = FastAPI(title="MindTrack 学习原型", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    def health():
        return {"status": "ok", "version": "0.1.0"}

    @app.get("/api/config")
    def config():
        return {"scenarios": settings.options["prediction"]["scenarios"],
                "text_mode": settings.options["features"]["text_mode"],
                "extractor": settings.options["plugins"]["extractor"],
                "predictor": settings.options["plugins"]["predictor"],
                "risk": settings.options["risk"]}

    def create_task(payload, prepare_files):
        pipeline = app.state.pipeline
        # 单进程内统一串行提交，配合数据库唯一约束防止重复入库。
        with pipeline.submit_lock:
            existing = db.find(payload.subject_id, payload.collection_id)
            if existing:
                return {"task_id": existing["id"], "status": existing["status"], "reused": True}
            if not pipeline.capacity.acquire(blocking=False):
                raise HTTPException(429, "后台任务已满，请稍后提交")
            task_id = uuid4().hex
            directory = settings.data_dir / "uploads" / task_id
            try:
                directory.mkdir(parents=True)
                files = prepare_files(directory)
                if not files and not payload.text.strip():
                    raise HTTPException(422, "至少填写文本或上传一种文件")
                data = payload.model_dump(mode="json")
                data["collected_at"] = iso_utc(payload.collected_at)
                db.create(task_id, data, files)
                pipeline.schedule(task_id)
            except Exception:
                pipeline.capacity.release()
                if db.get(task_id):
                    db.update(task_id, "failed", error="任务提交失败，请重试")
                shutil.rmtree(directory, ignore_errors=True)
                raise
            return {"task_id": task_id, "status": "queued", "reused": False}

    @app.post("/api/demo", status_code=202)
    def demo(payload: Submission):
        payload = payload.model_copy(update={"source": "demo"})
        if not payload.text.strip():
            payload = payload.model_copy(update={"text": "今天学习了一会儿，晚上打算早点休息。"})
        def prepare(directory):
            samples = {"eeg": "eeg/S001R01.edf", "face": "face/business-person.png",
                       "audio": "audio/5703-47212-0000.ogg"}
            files = {}
            for kind, relative in samples.items():
                source = settings.root / "resources/samples" / relative
                if not source.is_file():
                    raise HTTPException(503, "样例资源不完整，请执行 python scripts/download_resources.py")
                destination = directory / (kind + source.suffix)
                shutil.copyfile(source, destination)
                files[kind] = str(destination)
            return files
        return create_task(payload, prepare)

    @app.post("/api/tasks", status_code=202)
    def upload(metadata: str = Form(...), eeg: UploadFile | None = File(None),
               face: UploadFile | None = File(None), audio: UploadFile | None = File(None)):
        uploads = {"eeg": eeg, "face": face, "audio": audio}
        try:
            try:
                payload = Submission.model_validate_json(metadata)
            except ValidationError as exc:
                raise HTTPException(422, str(exc)) from exc
            payload = payload.model_copy(update={"source": "uploaded"})
            def prepare(directory):
                files = {}
                for kind, item in uploads.items():
                    if item is None:
                        continue
                    suffix = Path(item.filename or "").suffix.lower()
                    if suffix not in EXTENSIONS[kind]:
                        raise HTTPException(422, f"{kind} 文件类型不支持")
                    target = directory / (kind + suffix)
                    size = 0
                    with target.open("wb") as output:
                        while chunk := item.file.read(1024 * 1024):
                            size += len(chunk)
                            if size > settings.options["app"]["max_file_mb"] * 1024 * 1024:
                                raise HTTPException(413, "单个文件超过大小限制")
                            output.write(chunk)
                    if size == 0:
                        raise HTTPException(422, "不能上传空文件")
                    files[kind] = str(target)
                return files
            return create_task(payload, prepare)
        finally:
            for item in uploads.values():
                if item is not None:
                    item.file.close()

    @app.get("/api/tasks/{task_id}")
    def task(task_id: str):
        found = db.get(task_id)
        if found is None:
            raise HTTPException(404, "任务不存在")
        return public_task(found)

    @app.post("/api/tasks/{task_id}/retry", status_code=202)
    def retry(task_id: str):
        pipeline = app.state.pipeline
        with pipeline.submit_lock:
            found = db.get(task_id)
            if found is None:
                raise HTTPException(404, "任务不存在")
            if found["status"] != "failed":
                raise HTTPException(409, "只有失败任务可以重试")
            if not pipeline.capacity.acquire(blocking=False):
                raise HTTPException(429, "后台任务已满")
            db.update(task_id, "queued")
            pipeline.schedule(task_id)
        return {"task_id": task_id, "status": "queued"}

    def query_tasks(subject_id, start, end, limit):
        start, end = iso_utc(start), iso_utc(end)
        if start and end and start >= end:
            raise HTTPException(422, "开始时间必须早于结束时间")
        return db.list(subject_id, start, end, limit)

    @app.get("/api/records")
    def records(subject_id: str | None = None, start: datetime | None = None,
                end: datetime | None = None, limit: int = Query(200, ge=1, le=10000)):
        return [public_task(t, detail=False) for t in query_tasks(subject_id, start, end, limit)]

    @app.get("/api/export")
    def export(subject_id: str | None = None, start: datetime | None = None,
               end: datetime | None = None):
        rows = query_tasks(subject_id, start, end, 10000)
        output = io.StringIO(newline="")
        writer = csv.writer(output)
        writer.writerow(["task_id", "subject_id", "collection_id", "collected_at", "status",
                         "score", "risk_label", "is_simulated", "model_version", "source"])
        for row in rows:
            result = row["result"] or {}
            values = [row["id"], row["subject_id"], row["collection_id"], row["collected_at"],
                      row["status"], result.get("score", ""), result.get("risk_label", ""),
                      result.get("is_simulated", ""), result.get("model_version", ""), row["request"]["source"]]
            # 允许使用者自定义的字符串也不能变成电子表格公式。
            writer.writerow(["'" + str(v) if str(v).startswith(("=", "+", "-", "@")) else v for v in values])
        return Response(content=output.getvalue().encode("utf-8-sig"), media_type="text/csv",
                        headers={"Content-Disposition": 'attachment; filename="records.csv"'})

    return app


app = create_app()
