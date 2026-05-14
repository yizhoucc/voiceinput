import subprocess
import threading
import queue
import time


class SystemTextInserter:
    """Replace entire text field content with latest display text.

    Uses Cmd+A (select all) + Cmd+V (paste) as a single atomic operation.
    No backspace counting, no state tracking, no sync issues.
    Works in empty fields and chat boxes where our text is the only content.

    Serialized via queue to prevent overlapping AppleScript calls.
    """

    def __init__(self):
        self._q: queue.Queue[str | None] = queue.Queue()
        self._worker = threading.Thread(target=self._work, daemon=True)
        self._worker.start()

    def update(self, text: str):
        self._q.put(text)

    def commit(self, text: str):
        pass  # commit is handled by the full update

    def reset(self):
        while not self._q.empty():
            try:
                self._q.get_nowait()
            except queue.Empty:
                break

    def _work(self):
        while True:
            text = self._q.get()
            if text is None:
                break

            # Skip to latest
            latest = text
            while not self._q.empty():
                try:
                    latest = self._q.get_nowait()
                except queue.Empty:
                    break

            try:
                subprocess.run(["pbcopy"], input=latest.encode("utf-8"), check=True)
                subprocess.run([
                    "osascript", "-e",
                    '''tell application "System Events"
                        keystroke "a" using command down
                        keystroke "v" using command down
                    end tell'''
                ], check=True, capture_output=True, timeout=5)
            except Exception as e:
                print(f"\n[insert] Error: {e}")
