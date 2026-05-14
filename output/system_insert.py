import subprocess
import threading
import queue


class SystemTextInserter:
    """Two modes: real-time Cmd+A replace + final append-only.

    During recording: Cmd+A + Cmd+V replaces entire field with latest text.
    After stop: the final committed text is what stays.
    """

    def __init__(self):
        self._q: queue.Queue[str | None] = queue.Queue()
        self._worker = threading.Thread(target=self._work, daemon=True)
        self._worker.start()

    def update(self, text: str):
        """Replace entire text field with latest text (during recording)."""
        self._q.put(("update", text))

    def append(self, text: str):
        """Append text (for final commit after recording stops)."""
        self._q.put(("append", text))

    def reset(self):
        while not self._q.empty():
            try:
                self._q.get_nowait()
            except queue.Empty:
                break

    def _work(self):
        while True:
            item = self._q.get()
            if item is None:
                break

            # Skip to latest update (keep append operations)
            mode, text = item
            if mode == "update":
                latest = text
                while not self._q.empty():
                    try:
                        peek = self._q.get_nowait()
                        if peek[0] == "update":
                            latest = peek[1]
                        else:
                            # Put non-update back and process current update first
                            self._do_update(latest)
                            mode, text = peek
                            break
                    except queue.Empty:
                        break
                else:
                    text = latest

            try:
                if mode == "update":
                    self._do_update(text)
                elif mode == "append":
                    self._do_paste(text)
            except Exception as e:
                print(f"\n[insert] Error: {e}")

    def _do_update(self, text: str):
        """Cmd+A + Cmd+V to replace entire field."""
        subprocess.run(["pbcopy"], input=text.encode("utf-8"), check=True)
        subprocess.run([
            "osascript", "-e",
            '''tell application "System Events"
                keystroke "a" using command down
                keystroke "v" using command down
            end tell'''
        ], check=True, capture_output=True, timeout=5)

    def _do_paste(self, text: str):
        """Simple Cmd+V paste (append at cursor)."""
        subprocess.run(["pbcopy"], input=text.encode("utf-8"), check=True)
        subprocess.run([
            "osascript", "-e",
            'tell application "System Events" to keystroke "v" using command down'
        ], check=True, capture_output=True, timeout=5)
