import threading
from pynput import keyboard


class HotkeyListener:
    def __init__(self, on_activate, on_deactivate):
        self._on_activate = on_activate
        self._on_deactivate = on_deactivate
        self._listener: keyboard.Listener | None = None
        self._option_pressed = False
        self._space_pressed = False
        self._active = False

    def start(self):
        self._listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
        )
        self._listener.daemon = True
        self._listener.start()

    def stop(self):
        if self._listener:
            self._listener.stop()

    def _on_press(self, key):
        if key == keyboard.Key.alt or key == keyboard.Key.alt_l or key == keyboard.Key.alt_r:
            self._option_pressed = True
        elif key == keyboard.Key.space and self._option_pressed:
            if not self._active:
                self._active = True
                self._on_activate()

    def _on_release(self, key):
        if key == keyboard.Key.alt or key == keyboard.Key.alt_l or key == keyboard.Key.alt_r:
            self._option_pressed = False
            if self._active:
                self._active = False
                self._on_deactivate()
        elif key == keyboard.Key.space:
            if self._active and not self._option_pressed:
                self._active = False
                self._on_deactivate()
