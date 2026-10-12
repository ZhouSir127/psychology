"""学习版入口：两个本机服务，Ctrl+C同时关闭。"""
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parent

def main():
    for port in (8000, 8501):
        with socket.socket() as test:
            try:
                test.bind(("127.0.0.1", port))
            except OSError:
                raise SystemExit(f"端口{port}已占用，请先关闭之前启动的服务。")
    runtime = ROOT / "runtime"
    runtime.mkdir(exist_ok=True)
    children = []
    with (runtime / "backend.log").open("a", encoding="utf-8") as api_log, \
            (runtime / "frontend.log").open("a", encoding="utf-8") as ui_log:
        try:
            children.append(subprocess.Popen(
                [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8000"],
                cwd=ROOT, stdout=api_log, stderr=subprocess.STDOUT) )
            
            for _ in range(60):
                if children[0].poll() is not None:
                    raise RuntimeError("后端启动失败，请查看 runtime/backend.log")
                try:
                    with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=1):
                        break
                except OSError:
                    time.sleep(0.5)
            else:
                raise RuntimeError("后端启动超时，请查看 runtime/backend.log")
            
            children.append(subprocess.Popen(
                [sys.executable, "-m", "streamlit", "run", "dashboard.py", "--server.address", "127.0.0.1",
                 "--server.port", "8501", "--server.headless", "true", "--browser.gatherUsageStats", "false"],
                cwd=ROOT, stdout=ui_log, stderr=subprocess.STDOUT))
            print("页面：http://127.0.0.1:8501\n接口文档：http://127.0.0.1:8000/docs\n按 Ctrl+C 关闭。", flush=True)
            while all(p.poll() is None for p in children):
                time.sleep(1)
            raise RuntimeError("服务退出，请查看 runtime 下的日志")
        except KeyboardInterrupt:
            print("正在关闭服务……")
        finally:
            for child in reversed(children):
                if child.poll() is None:
                    child.terminate()
            for child in children:
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()


if __name__ == "__main__":
    main()
