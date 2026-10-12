from datetime import datetime, timezone
import json
import time
from threading import Event
import pytest
from fastapi.testclient import TestClient
from app.config import load_settings
from app.contracts import FeatureBundle, ModalityFeature, Prediction
from app.db import Database
from app.main import create_app

GATE = Event()


class TestExtractor:
    __test__ = False

    def __init__(self, settings):
        pass

    def extract(self, sample):
        if sample.text == "block":
            GATE.wait(timeout=5)
        if sample.text == "fail":
            raise ValueError("故意触发特征处理失败")
        return FeatureBundle(extractor_version="test-v1", modalities={
            "text": ModalityFeature(values=[1, 2, 3, 4, 5, 6, 7])})


class AlternatePredictor:
    def __init__(self, settings):
        pass

    def predict(self, features, sample, history, scenario=None):
        return Prediction(score=20.1, model_version="alternate-v1", is_simulated=True,
                          explanation=str(len(history)))


@pytest.fixture
def settings(tmp_path):
    result = load_settings()
    result.data_dir = tmp_path
    result.options["plugins"]["extractor"] = "tests.test_api:TestExtractor"
    return result


@pytest.fixture
def client(settings):
    with TestClient(create_app(settings)) as value:
        yield value


def payload(collection="one", subject="demo_01", text="测试", collected="2026-10-06T10:00:00+08:00"):
    return {"subject_id": subject, "collection_id": collection, "collected_at": collected,
            "text": text, "scenario": "高风险演示"}


def submit(client, data):
    return client.post("/api/tasks", data={"metadata": json.dumps(data)})


def finish(client, task_id):
    for _ in range(200):
        row = client.get(f"/api/tasks/{task_id}").json()
        if row["status"] in ("succeeded", "failed"):
            return row
        time.sleep(0.01)
    raise AssertionError("任务未及时完成")


def test_pipeline_persistence_and_csv(client, settings):
    response = submit(client, payload())
    assert response.status_code == 202
    row = finish(client, response.json()["task_id"])
    assert row["status"] == "succeeded"
    assert row["result"]["score"] == 70
    assert row["result"]["risk_label"] == "高风险"
    assert row["result"]["is_simulated"] is True
    assert len(row["result"]["features"]["modalities"]["text"]["values"]) == 7
    assert "files" not in row and "request" not in row
    assert Database(settings.data_dir / "mindtrack.sqlite3").get(row["id"])["status"] == "succeeded"
    exported = client.get("/api/export?subject_id=demo_01")
    assert exported.content.startswith(b"\xef\xbb\xbf")
    assert "mock-v1" in exported.text and "True" in exported.text


def test_idempotent_collection(client):
    a = submit(client, payload()).json()
    b = submit(client, payload()).json()
    assert a["task_id"] == b["task_id"]
    assert b["reused"] is True
    assert len(client.get("/api/records").json()) == 1


def test_filter_by_subject_and_time(client):
    for item in (payload("a"), payload("b", subject="demo_02"),
                 payload("c", collected="2026-10-05T10:00:00+08:00")):
        finish(client, submit(client, item).json()["task_id"])
    response = client.get("/api/records", params={"subject_id": "demo_01",
        "start": "2026-10-06T00:00:00+08:00", "end": "2026-10-07T00:00:00+08:00"})
    assert [r["collection_id"] for r in response.json()] == ["a"]


def test_failures_are_visible_and_retryable(client):
    row = finish(client, submit(client, payload(text="fail")).json()["task_id"])
    assert row["status"] == "failed" and "故意" in row["error"]
    assert client.post(f"/api/tasks/{row['id']}/retry").status_code == 202
    assert finish(client, row["id"])["status"] == "failed"


def test_restart_marks_interrupted_tasks(settings):
    db = Database(settings.data_dir / "mindtrack.sqlite3")
    db.initialize()
    data = payload()
    data["source"] = "uploaded"
    db.create("interrupted", data, {})
    with TestClient(create_app(settings)) as client:
        row = client.get("/api/tasks/interrupted").json()
        assert row["status"] == "failed" and "上次服务" in row["error"]


def test_upload_validation(client, settings):
    assert submit(client, payload(text="")).status_code == 422
    assert submit(client, payload(collected="2026-10-06T10:00:00")).status_code == 422
    assert submit(client, payload(subject="../../outside")).status_code == 422
    assert client.post("/api/tasks", data={"metadata": "not-json"}).status_code == 422
    assert client.post("/api/tasks", data={"metadata": json.dumps(payload())},
                       files={"face": ("bad.exe", b"123")}).status_code == 422
    settings.options["app"]["max_file_mb"] = 0
    assert client.post("/api/tasks", data={"metadata": json.dumps(payload())},
                       files={"eeg": ("x.edf", b"123")}).status_code == 413


def test_missing_task_and_invalid_filter(client):
    assert client.get("/api/tasks/absent").status_code == 404
    assert client.get("/api/records", params={"start": "2026-01-01T00:00:00"}).status_code == 422


def test_busy_worker_does_not_block_queries(settings):
    settings.options["app"]["max_pending_tasks"] = 1
    GATE.clear()
    try:
        with TestClient(create_app(settings)) as client:
            task = submit(client, payload(text="block")).json()
            assert client.get("/health").status_code == 200
            assert submit(client, payload("second")).status_code == 429
            GATE.set()
            assert finish(client, task["task_id"])["status"] == "succeeded"
    finally:
        GATE.set()


def test_plugin_replacement_and_history(settings):
    settings.options["plugins"]["predictor"] = "tests.test_api:AlternatePredictor"
    with TestClient(create_app(settings)) as client:
        for item in (payload("future", collected="2026-10-08T10:00:00+08:00"),
                     payload("other", subject="demo_02", collected="2026-10-01T10:00:00+08:00"),
                     payload("past", collected="2026-10-01T10:00:00+08:00")):
            finish(client, submit(client, item).json()["task_id"])
        row = finish(client, submit(client, payload()).json()["task_id"])
        assert row["result"]["model_version"] == "alternate-v1"
        assert row["result"]["risk_label"] == "低风险"
        assert row["result"]["explanation"] == "1"
