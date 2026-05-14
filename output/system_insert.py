import subprocess
import threading


class SystemTextInserter:
    """Insert final text at cursor position in any macOS app.

    Only inserts on finalize (not during partials) to avoid
    undo/replace timing issues. Single Cmd+V paste = reliable.
    """

    def __init__(self):
        self._lock = threading.Lock()

    def insert(self, text: str):
        """Paste text at current cursor position."""
        with self._lock:
            subprocess.run(["pbcopy"], input=text.encode("utf-8"), check=True)
            subprocess.run([
                "osascript", "-e",
                'tell application "System Events" to keystroke "v" using command down'
            ], check=True, capture_output=True)

    def reset(self):
        pass
