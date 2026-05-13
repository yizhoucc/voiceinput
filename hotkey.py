import threading
import sys


class HotkeyListener:
    """Simple Enter-key toggle. Press Enter to start/stop recording."""

    def __init__(self, on_activate, on_deactivate):
        self._on_activate = on_activate
        self._on_deactivate = on_deactivate
        self._active = False

    def start(self):
        t = threading.Thread(target=self._loop, daemon=True)
        t.start()

    def _loop(self):
        while True:
            try:
                input()
            except EOFError:
                break
            if self._active:
                self._active = False
                self._on_deactivate()
            else:
                self._active = True
                self._on_activate()

    def stop(self):
        pass
