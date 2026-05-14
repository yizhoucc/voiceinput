import ctypes
import ctypes.util
import subprocess
import sys
import threading


def check_accessibility() -> bool:
    """Check if this process has macOS Accessibility permission."""
    path = ctypes.util.find_library("ApplicationServices")
    if not path:
        return False
    appserv = ctypes.cdll.LoadLibrary(path)
    return bool(appserv.AXIsProcessTrusted())


def request_accessibility():
    """Open System Settings to the Accessibility pane and prompt user."""
    print("[hotkey] Accessibility permission required for global hotkey.")
    print("[hotkey] Opening System Settings → Privacy → Accessibility...")
    print("[hotkey] Add your terminal app, then restart this program.")
    print()
    subprocess.Popen([
        "open", "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility"
    ])


class HotkeyListener:
    """Global hotkey listener. Ctrl+Option+1 to toggle recording.

    Requires Accessibility permission. Falls back to Enter key if not granted.
    """

    def __init__(self, on_activate, on_deactivate):
        self._on_activate = on_activate
        self._on_deactivate = on_deactivate
        self._active = False
        self._use_pynput = False

    def start(self):
        if check_accessibility():
            self._use_pynput = True
            self._start_pynput()
        else:
            request_accessibility()
            print("[hotkey] Falling back to Enter key toggle for now.\n")
            self._start_terminal()

    def _start_pynput(self):
        from pynput import keyboard

        self._pressed = set()
        self._listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
        )
        self._listener.daemon = True
        self._listener.start()
        print("[hotkey] Listening for Ctrl+Option+1 (global)")

    def _on_press(self, key):
        from pynput import keyboard

        if key in (keyboard.Key.ctrl, keyboard.Key.ctrl_l, keyboard.Key.ctrl_r):
            self._pressed.add("ctrl")
        elif key in (keyboard.Key.alt, keyboard.Key.alt_l, keyboard.Key.alt_r):
            self._pressed.add("alt")
        elif hasattr(key, "char") and key.char == "1":
            if "ctrl" in self._pressed and "alt" in self._pressed:
                self._toggle()

    def _on_release(self, key):
        from pynput import keyboard

        if key in (keyboard.Key.ctrl, keyboard.Key.ctrl_l, keyboard.Key.ctrl_r):
            self._pressed.discard("ctrl")
        elif key in (keyboard.Key.alt, keyboard.Key.alt_l, keyboard.Key.alt_r):
            self._pressed.discard("alt")

    def _start_terminal(self):
        t = threading.Thread(target=self._terminal_loop, daemon=True)
        t.start()

    def _terminal_loop(self):
        while True:
            try:
                input()
                self._toggle()
            except EOFError:
                break

    def _toggle(self):
        if self._active:
            self._active = False
            self._on_deactivate()
        else:
            self._active = True
            self._on_activate()

    def stop(self):
        if self._use_pynput and hasattr(self, "_listener"):
            self._listener.stop()
