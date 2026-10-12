"""手动启用的真实样例联调，首次语音计算会较慢。"""
from datetime import datetime, timezone
import os
import time
import pytest
from fastapi.testclient import TestClient
from app.config import load_settings
from app.main import create_app


@pytest.mark.skipif(os.environ.get("MINDTRACK_TEST_RESOURCES") != "1",
                    reason="设置MINDTRACK_TEST_RESOURCES=1后验证真实公开样例")
def test_public_samples_end_to_end(tmp_path):
    settings = load_settings()
    settings.data_dir = tmp_path
    with TestClient(create_app(settings)) as client:
        response = client.post("/api/demo", json={
            "subject_id": "demo_01", "collection_id": "resource-check",
            "collected_at": datetime.now(timezone.utc).isoformat(),
            "text": "今天有点疲惫，想早点休息。", "scenario": "高风险演示",
        })
        assert response.status_code == 202, response.text
        task_id = response.json()["task_id"]
        deadline = time.monotonic() + 240
        while time.monotonic() < deadline:
            task = client.get(f"/api/tasks/{task_id}").json()
            if task["status"] in ("succeeded", "failed"):
                break
            time.sleep(0.2)
        assert task["status"] == "succeeded", task
        result = task["result"]
        assert result["is_simulated"] and result["score"] == 70
        modalities = result["features"]["modalities"]
        assert set(modalities) == {"eeg", "audio", "face", "text"}
        assert modalities["eeg"]["metadata"]["channel_count"] == 64
        assert len(modalities["face"]["values"]) == 52
        assert len(modalities["audio"]["values"]) == 15
