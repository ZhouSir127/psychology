"""模拟主线一的上传客户端；重复 collection_id 会复用已有任务。"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import httpx

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", default="demo_01")
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--interval", type=float, default=3, help="上传间隔秒数")
    args = parser.parse_args()
    if args.count < 1 or args.interval < 0:
        parser.error("count必须为正，interval不能为负")
    with httpx.Client(base_url="http://127.0.0.1:8000", timeout=60) as client:
        scenarios = client.get("/api/config").json()["scenarios"]
        for index in range(args.count):
            payload = {"subject_id": args.subject,
                       "collection_id": "replay_" + str(time.time_ns()),
                       "collected_at": datetime.now(timezone.utc).isoformat(),
                       "text": "这是一条模拟采集客户端发送的文本。",
                       "scenario": scenarios[index % len(scenarios)]}
            # 真正走multipart上传接口，便于以后对接采集程序。
            with (ROOT / "resources/samples/eeg/S001R01.edf").open("rb") as eeg:
                response = client.post("/api/tasks", data={"metadata": json.dumps(payload)},
                                       files={"eeg": ("sample.edf", eeg, "application/octet-stream")})
            response.raise_for_status()
            print(response.json(), flush=True)
            if index + 1 < args.count:
                time.sleep(args.interval)


if __name__ == "__main__":
    main()
