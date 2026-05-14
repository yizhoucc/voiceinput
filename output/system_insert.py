import subprocess
import time
import threading


class SystemTextInserter:
    """Insert text at cursor position in any macOS app via clipboard + Cmd+V."""

    def __init__(self):
        self._lock = threading.Lock()
        self._last_inserted_len = 0

    def insert(self, text: str):
        """Insert text at current cursor position."""
        with self._lock:
            self._clipboard_paste(text)
            self._last_inserted_len = len(text)

    def replace_last(self, new_text: str):
        """Delete previously inserted text and insert new text."""
        with self._lock:
            if self._last_inserted_len > 0:
                self._delete_chars(self._last_inserted_len)
            self._clipboard_paste(new_text)
            self._last_inserted_len = len(new_text)

    def _clipboard_paste(self, text: str):
        """Copy text to clipboard and simulate Cmd+V."""
        # Save current clipboard
        # Copy our text
        proc = subprocess.run(["pbcopy"], input=text.encode("utf-8"), check=True)
        time.sleep(0.05)
        # Simulate Cmd+V
        self._press_cmd_v()

    def _press_cmd_v(self):
        """Simulate Cmd+V using osascript."""
        subprocess.run([
            "osascript", "-e",
            'tell application "System Events" to keystroke "v" using command down'
        ], check=True, capture_output=True)

    def _delete_chars(self, count: int):
        """Simulate pressing backspace N times using osascript."""
        # Select the text by pressing Shift+Left arrow N times, then delete
        # This is more reliable than N individual backspaces
        script = f'''
        tell application "System Events"
            repeat {count} times
                key code 51
            end repeat
        end tell
        '''
        subprocess.run(["osascript", "-e", script], check=True, capture_output=True)

    def reset(self):
        self._last_inserted_len = 0
