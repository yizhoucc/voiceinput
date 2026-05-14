import subprocess
import threading
import time
import Quartz


class SystemTextInserter:
    """Two-level editor insertion: permanent commits + temporary partial display.

    Committed text: appended via Cmd+V, never modified. One paste per segment.
    Partial text: displayed via diff (backspace old + paste new), updates frequently.

    Editor always shows: [committed_text][partial_text]
    The partial portion is short (one sentence) so diff is fast and reliable.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._committed_len = 0  # chars of permanent text in editor
        self._partial_len = 0    # chars of temporary partial in editor
        self._committed_str = ""  # what's permanently in editor

    def commit(self, text: str):
        """Append committed text permanently. Remove current partial first."""
        with self._lock:
            if self._partial_len > 0:
                self._backspace(self._partial_len)
                self._partial_len = 0
            paste_text = text + " "
            self._paste(paste_text)
            self._committed_len += len(paste_text)
            self._committed_str += paste_text

    def update_partial(self, full_display: str):
        """Update the temporary partial portion of the display."""
        with self._lock:
            # Extract the non-committed portion
            new_partial = full_display[len(self._committed_str):]
            if not new_partial and self._partial_len == 0:
                return

            # Remove old partial, paste new partial
            if self._partial_len > 0:
                self._backspace(self._partial_len)
            if new_partial:
                self._paste(new_partial)
            self._partial_len = len(new_partial)

    def reset(self):
        with self._lock:
            self._committed_len = 0
            self._partial_len = 0
            self._committed_str = ""

    def _backspace(self, count: int):
        for _ in range(count):
            ev = Quartz.CGEventCreateKeyboardEvent(None, 51, True)
            Quartz.CGEventPost(Quartz.kCGHIDEventTap, ev)
            ev = Quartz.CGEventCreateKeyboardEvent(None, 51, False)
            Quartz.CGEventPost(Quartz.kCGHIDEventTap, ev)
        time.sleep(max(0.02, count * 0.001))

    def _paste(self, text: str):
        subprocess.run(["pbcopy"], input=text.encode("utf-8"), check=True)
        ev = Quartz.CGEventCreateKeyboardEvent(None, 9, True)
        Quartz.CGEventSetFlags(ev, Quartz.kCGEventFlagMaskCommand)
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, ev)
        ev = Quartz.CGEventCreateKeyboardEvent(None, 9, False)
        Quartz.CGEventSetFlags(ev, Quartz.kCGEventFlagMaskCommand)
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, ev)
        time.sleep(0.03)
