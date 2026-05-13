import sys
from datetime import datetime
from pathlib import Path


LOG_FILE = Path(__file__).parent.parent / "transcripts.log"


class TerminalOutput:
    def __init__(self):
        self._last_partial = ""
        self._session_start = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(LOG_FILE, "a") as f:
            f.write(f"\n=== Session {self._session_start} ===\n")

    def show_partial(self, text: str):
        sys.stdout.write(f"\r\033[K[...] {text}")
        sys.stdout.flush()
        self._last_partial = text

    def show_final(self, text: str):
        sys.stdout.write(f"\r\033[K[done] {text}\n")
        sys.stdout.flush()
        self._last_partial = ""
        ts = datetime.now().strftime("%H:%M:%S")
        with open(LOG_FILE, "a") as f:
            f.write(f"[{ts}] {text}\n")

    def show_status(self, msg: str):
        sys.stdout.write(f"\r\033[K{msg}")
        sys.stdout.flush()

    def clear_line(self):
        sys.stdout.write("\r\033[K")
        sys.stdout.flush()
