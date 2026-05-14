import ctypes
import ctypes.util
import subprocess
import threading


def check_accessibility() -> bool:
    path = ctypes.util.find_library("ApplicationServices")
    if not path:
        return False
    appserv = ctypes.cdll.LoadLibrary(path)
    return bool(appserv.AXIsProcessTrusted())


def request_accessibility():
    print("[hotkey] Accessibility permission required for global hotkey.")
    print("[hotkey] Opening System Settings → Privacy → Accessibility...")
    subprocess.Popen([
        "open", "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility"
    ])


class HotkeyListener:
    """Ctrl+Shift+R = smart mode, Ctrl+Shift+E = manual mode.
    Falls back to Enter/E in terminal if no Accessibility.
    """

    def __init__(self, on_start, on_stop):
        self._on_start = on_start  # on_start(mode: str)
        self._on_stop = on_stop    # on_stop()
        self._active = False
        self._mode = None

    def start(self):
        if check_accessibility():
            self._start_pynput()
        else:
            request_accessibility()
            print("[hotkey] Falling back to terminal keys.\n")
            self._start_terminal()

    def _start_pynput(self):
        from pynput import keyboard
        self._pressed = set()
        self._listener = keyboard.Listener(
            on_press=self._on_press, on_release=self._on_release)
        self._listener.daemon = True
        self._listener.start()
        print("[hotkey] Ctrl+Shift+R = smart | Ctrl+Shift+E = manual")

    def _on_press(self, key):
        from pynput import keyboard
        if key in (keyboard.Key.ctrl, keyboard.Key.ctrl_l, keyboard.Key.ctrl_r):
            self._pressed.add("ctrl")
        elif key in (keyboard.Key.shift, keyboard.Key.shift_l, keyboard.Key.shift_r):
            self._pressed.add("shift")
        elif "ctrl" in self._pressed and "shift" in self._pressed:
            mode = None
            if hasattr(key, "char") and key.char == "\x12":
                mode = "smart"
            elif hasattr(key, "char") and key.char == "\x05":
                mode = "manual"
            elif hasattr(key, "vk") and key.vk == 15:
                mode = "smart"
            elif hasattr(key, "vk") and key.vk == 14:
                mode = "manual"
            if mode:
                threading.Thread(target=self._toggle, args=(mode,), daemon=True).start()

    def _on_release(self, key):
        from pynput import keyboard
        if key in (keyboard.Key.ctrl, keyboard.Key.ctrl_l, keyboard.Key.ctrl_r):
            self._pressed.discard("ctrl")
        elif key in (keyboard.Key.shift, keyboard.Key.shift_l, keyboard.Key.shift_r):
            self._pressed.discard("shift")

    def _start_terminal(self):
        threading.Thread(target=self._terminal_loop, daemon=True).start()

    def _terminal_loop(self):
        while True:
            try:
                line = input()
                mode = "manual" if line.strip().lower() == "e" else "smart"
                self._toggle(mode)
            except EOFError:
                break

    def _toggle(self, mode):
        if self._active:
            self._on_stop()
            self._active = False
            self._mode = None
        else:
            self._active = True
            self._mode = mode
            self._on_start(mode)

    def stop(self):
        if hasattr(self, "_listener"):
            self._listener.stop()
