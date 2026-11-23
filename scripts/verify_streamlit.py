"""Check that Streamlit starts headlessly, then stop it."""
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PORT = 8599
URL = f"http://127.0.0.1:{PORT}/_stcore/health"


def main() -> None:
    proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "app.py",
         "--server.headless", "true", "--server.port", str(PORT),
         "--browser.gatherUsageStats", "false"],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace",
    )
    try:
        deadline = time.time() + 60
        health = ""
        while time.time() < deadline:
            if proc.poll() is not None:
                out = proc.stdout.read() if proc.stdout else ""
                raise AssertionError(f"Streamlit exited early:\n{out}")
            try:
                with urllib.request.urlopen(URL, timeout=2) as resp:
                    health = resp.read().decode(errors="replace").strip()
                    if resp.status == 200:
                        break
            except (urllib.error.URLError, OSError):
                time.sleep(0.5)
        else:
            raise AssertionError("Streamlit health endpoint did not respond in 60s")
        assert health.lower() == "ok", health
        print(f"streamlit health: {health} at {URL}")
        print("STREAMLIT_STARTUP_OK")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    main()
