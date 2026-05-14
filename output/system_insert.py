import subprocess
import threading
import queue
import Quartz
import time


class SystemTextInserter:
    """Insert/update text at cursor using incremental diff.

    Instead of undo+paste (unreliable) or select+paste (slow),
    we compute the diff between old and new text:
    1. Find common prefix
    2. Backspace to delete old suffix (few chars, fast via CGEvent)
    3. Paste new suffix via clipboard (handles Chinese/Unicode)

    Most updates only change a few characters at the tail → fast & invisible.
    """

    def __init__(self):
        self._q: queue.Queue[str | None] = queue.Queue()
        self._last_text = ""
        self._worker = threading.Thread(target=self._work, daemon=True)
        self._worker.start()

    def replace_last(self, new_text: str):
        self._q.put(new_text)

    def reset(self):
        while not self._q.empty():
            try:
                self._q.get_nowait()
            except queue.Empty:
                break
        self._last_text = ""

    def _work(self):
        while True:
            text = self._q.get()
            if text is None:
                break

            # Only process the latest
            latest = text
            while not self._q.empty():
                try:
                    latest = self._q.get_nowait()
                except queue.Empty:
                    break

            try:
                self._apply_diff(latest)
            except Exception as e:
                print(f"\n[insert] Error: {e}")

    def _apply_diff(self, new_text: str):
        old = self._last_text
        # Find common prefix
        prefix_len = 0
        for i in range(min(len(old), len(new_text))):
            if old[i] == new_text[i]:
                prefix_len += 1
            else:
                break

        # How many chars to delete from old (after common prefix)
        delete_count = len(old) - prefix_len
        # What to append (after common prefix)
        append_text = new_text[prefix_len:]

        if delete_count > 0:
            self._backspace(delete_count)

        if append_text:
            self._paste(append_text)

        self._last_text = new_text

    def _backspace(self, count: int):
        """Send N backspace events via CGEvent (fast, ~0.7ms each)."""
        for _ in range(count):
            ev = Quartz.CGEventCreateKeyboardEvent(None, 51, True)  # 51 = backspace
            Quartz.CGEventPost(Quartz.kCGHIDEventTap, ev)
            ev = Quartz.CGEventCreateKeyboardEvent(None, 51, False)
            Quartz.CGEventPost(Quartz.kCGHIDEventTap, ev)
        if count > 0:
            time.sleep(0.02)  # small delay to let app process

    def _paste(self, text: str):
        """Paste text via clipboard + Cmd+V."""
        subprocess.run(["pbcopy"], input=text.encode("utf-8"), check=True)
        ev = Quartz.CGEventCreateKeyboardEvent(None, 9, True)  # v key
        Quartz.CGEventSetFlags(ev, Quartz.kCGEventFlagMaskCommand)
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, ev)
        ev = Quartz.CGEventCreateKeyboardEvent(None, 9, False)
        Quartz.CGEventSetFlags(ev, Quartz.kCGEventFlagMaskCommand)
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, ev)
        time.sleep(0.03)  # let paste complete
