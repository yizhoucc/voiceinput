import subprocess
import Quartz


class SystemTextInserter:
    """Append-only text insertion at cursor. Only pastes committed text."""

    def commit(self, text: str):
        """Paste committed text at cursor. Called once per segment, never modified."""
        if not text:
            return
        subprocess.run(["pbcopy"], input=(text + " ").encode("utf-8"), check=True)
        ev = Quartz.CGEventCreateKeyboardEvent(None, 9, True)
        Quartz.CGEventSetFlags(ev, Quartz.kCGEventFlagMaskCommand)
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, ev)
        ev = Quartz.CGEventCreateKeyboardEvent(None, 9, False)
        Quartz.CGEventSetFlags(ev, Quartz.kCGEventFlagMaskCommand)
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, ev)

    def update_partial(self, text: str):
        """No-op for editor. Partial shown in overlay instead."""
        pass

    def reset(self):
        pass
