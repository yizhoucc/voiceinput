import subprocess
import threading
import queue
import Quartz


class SystemTextInserter:
    """Insert/replace text at cursor via character-count selection + paste.

    No Cmd+Z undo (unreliable across apps). Instead:
    1. Track how many characters we last inserted
    2. Select that many chars backwards (Shift+Left via CGEvent, ~70ms/100 chars)
    3. Paste new text (replaces selection)
    """

    def __init__(self):
        self._q: queue.Queue[str | None] = queue.Queue()
        self._has_inserted = False
        self._last_char_count = 0
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
        self._has_inserted = False
        self._last_char_count = 0

    def _work(self):
        while True:
            text = self._q.get()
            if text is None:
                break

            latest = text
            while not self._q.empty():
                try:
                    latest = self._q.get_nowait()
                except queue.Empty:
                    break

            try:
                if self._has_inserted and self._last_char_count > 0:
                    print(f"\n[insert] select back {self._last_char_count} chars, paste {len(latest)} chars")
                    self._select_back(self._last_char_count)
                else:
                    print(f"\n[insert] first paste {len(latest)} chars")
                self._paste(latest)
                self._has_inserted = True
                self._last_char_count = len(latest)
            except Exception as e:
                print(f"\n[insert] Error: {e}")

    def _select_back(self, count: int):
        """Select `count` characters backwards using Shift+Left via CGEvent."""
        for _ in range(count):
            # Key down: Left arrow (keycode 123) with Shift
            ev_down = Quartz.CGEventCreateKeyboardEvent(None, 123, True)
            Quartz.CGEventSetFlags(ev_down, Quartz.kCGEventFlagMaskShift)
            Quartz.CGEventPost(Quartz.kCGHIDEventTap, ev_down)
            # Key up
            ev_up = Quartz.CGEventCreateKeyboardEvent(None, 123, False)
            Quartz.CGEventPost(Quartz.kCGHIDEventTap, ev_up)

    def _paste(self, text: str):
        """Copy to clipboard and Cmd+V."""
        subprocess.run(["pbcopy"], input=text.encode("utf-8"), check=True)
        # Cmd+V via CGEvent
        ev_down = Quartz.CGEventCreateKeyboardEvent(None, 9, True)  # 'v' keycode
        Quartz.CGEventSetFlags(ev_down, Quartz.kCGEventFlagMaskCommand)
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, ev_down)
        ev_up = Quartz.CGEventCreateKeyboardEvent(None, 9, False)
        Quartz.CGEventSetFlags(ev_up, Quartz.kCGEventFlagMaskCommand)
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, ev_up)
