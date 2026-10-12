"""从官方来源下载小型公开样例；--text 可额外下载约500MB文本模型。"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
ASSETS = {
    "samples/eeg/S001R01.edf": "https://physionet.org/files/eegmmidb/1.0.0/S001/S001R01.edf",
    "samples/eeg/S001R02.edf": "https://physionet.org/files/eegmmidb/1.0.0/S001/S001R02.edf",
    "samples/audio/5703-47212-0000.ogg": "https://librosa.org/data/audio/5703-47212-0000.ogg",
    "samples/face/business-person.png": "https://storage.googleapis.com/mediapipe-assets/business-person.png",
    "models/face/face_landmarker.task": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
}


def download(url, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and target.stat().st_size > 0:
        print(f"已存在：{target.name}", flush=True)
        return
    temporary = target.with_suffix(target.suffix + ".part")
    for attempt in range(3):
        try:
            print(f"下载：{target.name}", flush=True)
            request = urllib.request.Request(url, headers={"User-Agent": "MindTrack-Learning/0.1"})
            with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
            if temporary.stat().st_size == 0:
                raise ValueError("下载文件为空")
            temporary.replace(target)
            return
        except Exception:
            temporary.unlink(missing_ok=True)
            if attempt == 2:
                raise
            time.sleep(2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", action="store_true", help="额外下载多语言MiniLM文本模型")
    args = parser.parse_args()
    resource_dir = ROOT / "resources"
    manifest = []
    for name, url in ASSETS.items():
        target = resource_dir / name
        download(url, target)
        sha = hashlib.sha256(target.read_bytes()).hexdigest()
        if name.endswith(".ogg") and sha != "a284612b46af0535f7e1873758c4387bb8369f6dbbe192ffdec1f171108f98dd":
            raise ValueError("语音样例与librosa官方登记的SHA256不一致")
        manifest.append({"path": name, "url": url, "size_bytes": target.stat().st_size, "sha256": sha})
    (resource_dir / "manifest.json").write_text(json.dumps({
        "verified_at": datetime.now(timezone.utc).isoformat(), "assets": manifest,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.text:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            local_dir=resource_dir / "models/text/minilm",
            allow_patterns=["*.json", "*.safetensors", "*.model", "1_Pooling/*", "README.md"],
        )
        print("文本模型下载完成：把config.toml的text_mode改为minilm后重启后端")
    print("基础资源准备完成。运行 python run.py 启动项目。")


if __name__ == "__main__":
    main()
