import threading
from pynput import keyboard


class HotkeyListener:
    """Listen for Ctrl+Option+1 to toggle recording."""

    HOTKEY = {keyboard.Key.ctrl, keyboard.Key.alt}
    TRIGGER_CHAR = "1"

    def __init__(self, on_activate, on_deactivate):
        self._on_activate = on_activate
        self._on_deactivate = on_deactivate
        self._active = False
        self._pressed = set()
        self._listener: keyboard.Listener | None = None

    def start(self):
        self._listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
        )
        self._listener.daemon = True
        self._listener.start()

    def _on_press(self, key):
        if key in (keyboard.Key.ctrl, keyboard.Key.ctrl_l, keyboard.Key.ctrl_r):
            self._pressed.add(keyboard.Key.ctrl)
        elif key in (keyboard.Key.alt, keyboard.Key.alt_l, keyboard.Key.alt_r):
            self._pressed.add(keyboard.Key.alt)
        elif hasattr(key, 'char') and key.char == self.TRIGGER_CHAR:
            if self.HOTKEY.issubset(self._pressed):
                if self._active:
                    self._active = False
                    self._on_deactivate()
                else:
                    self._active = True
                    self._on_activate()

    def _on_release(self, key):
        if key in (keyboard.Key.ctrl, keyboard.Key.ctrl_l, keyboard.Key.ctrl_r):
            self._pressed.discard(keyboard.Key.ctrl)
        elif key in (keyboard.Key.alt, keyboard.Key.alt_l, keyboard.Key.alt_r):
            self._pressed.discard(keyboard.Key.alt)

    def stop(self):
        if self._listener:
            self._listener.stop()
