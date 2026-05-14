import subprocess
import threading
import queue


class SystemTextInserter:
    """Insert/replace text at cursor position in any macOS app.

    Uses a single worker thread to serialize all insert operations,
    preventing race conditions between undo+paste calls.
    """

    def __init__(self):
        self._q: queue.Queue[str | None] = queue.Queue()
        self._has_inserted = False
        self._worker = threading.Thread(target=self._work, daemon=True)
        self._worker.start()

    def replace_last(self, new_text: str):
        self._q.put(new_text)

    def reset(self):
        # Drain any pending operations
        while not self._q.empty():
            try:
                self._q.get_nowait()
            except queue.Empty:
                break
        self._has_inserted = False

    def _work(self):
        while True:
            text = self._q.get()
            if text is None:
                break

            # Skip stale entries, only process the latest
            latest = text
            while not self._q.empty():
                try:
                    latest = self._q.get_nowait()
                except queue.Empty:
                    break

            try:
                if self._has_inserted:
                    self._undo_and_paste(latest)
                else:
                    self._paste(latest)
                self._has_inserted = True
            except Exception as e:
                print(f"\n[insert] Error: {e}")

    def _paste(self, text: str):
        subprocess.run(["pbcopy"], input=text.encode("utf-8"), check=True)
        subprocess.run([
            "osascript", "-e",
            'tell application "System Events" to keystroke "v" using command down'
        ], check=True, capture_output=True, timeout=5)

    def _undo_and_paste(self, text: str):
        subprocess.run(["pbcopy"], input=text.encode("utf-8"), check=True)
        subprocess.run([
            "osascript", "-e",
            '''tell application "System Events"
                keystroke "z" using command down
                delay 0.05
                keystroke "v" using command down
            end tell'''
        ], check=True, capture_output=True, timeout=5)
