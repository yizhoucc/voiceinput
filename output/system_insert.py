import subprocess
import time
import threading


class SystemTextInserter:
    """Insert text at cursor position in any macOS app.

    Uses clipboard + Cmd+V for insert.
    For replacement: Cmd+Z (undo last paste) + paste new text.
    Two keystrokes regardless of text length = instant.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._has_inserted = False

    def insert(self, text: str):
        with self._lock:
            self._paste(text)
            self._has_inserted = True

    def replace_last(self, new_text: str):
        with self._lock:
            if self._has_inserted:
                self._undo_and_paste(new_text)
            else:
                self._paste(new_text)
            self._has_inserted = True

    def _paste(self, text: str):
        subprocess.run(["pbcopy"], input=text.encode("utf-8"), check=True)
        subprocess.run([
            "osascript", "-e",
            'tell application "System Events" to keystroke "v" using command down'
        ], check=True, capture_output=True)

    def _undo_and_paste(self, text: str):
        """Undo last paste + paste new text in one AppleScript call."""
        subprocess.run(["pbcopy"], input=text.encode("utf-8"), check=True)
        subprocess.run([
            "osascript", "-e",
            '''tell application "System Events"
                keystroke "z" using command down
                delay 0.03
                keystroke "v" using command down
            end tell'''
        ], check=True, capture_output=True)

    def reset(self):
        self._has_inserted = False
