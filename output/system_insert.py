import subprocess
import Quartz


class SystemTextInserter:
    """Append-only text insertion at cursor position.

    Only inserts committed (stable) text. Never modifies/deletes.
    Each committed segment is pasted once via Cmd+V.
    Partials are shown in terminal only.

    This is 100% reliable because:
    - Single Cmd+V per segment (no undo, no selection, no backspace)
    - Once pasted, text is never touched again
    - No _last_text tracking needed, no sync issues
    """

    def __init__(self):
        self._inserted_count = 0

    def append(self, text: str):
        """Append text at current cursor position. Called once per committed segment."""
        if not text:
            return
        subprocess.run(["pbcopy"], input=text.encode("utf-8"), check=True)
        ev = Quartz.CGEventCreateKeyboardEvent(None, 9, True)  # v key
        Quartz.CGEventSetFlags(ev, Quartz.kCGEventFlagMaskCommand)
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, ev)
        ev = Quartz.CGEventCreateKeyboardEvent(None, 9, False)
        Quartz.CGEventSetFlags(ev, Quartz.kCGEventFlagMaskCommand)
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, ev)
        self._inserted_count += 1

    def reset(self):
        self._inserted_count = 0
