import sys
from datetime import datetime
from pathlib import Path


LOG_FILE = Path(__file__).parent.parent / "transcripts.log"


class TerminalOutput:
    def __init__(self):
        self._log = open(LOG_FILE, "a")
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._log.write(f"\n=== Session {ts} ===\n")
        self._log.flush()

    def show_partial(self, text: str):
        sys.stdout.write(f"\r\033[K[...] {text}")
        sys.stdout.flush()

    def show_final(self, text: str):
        sys.stdout.write(f"\r\033[K[done] {text}\n")
        sys.stdout.flush()
        ts = datetime.now().strftime("%H:%M:%S")
        self._log.write(f"[{ts}] {text}\n")
        self._log.flush()
