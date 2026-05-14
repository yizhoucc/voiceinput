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
    print("[hotkey] Add your terminal app, then restart.")
    print()
    subprocess.Popen([
        "open", "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility"
    ])


class HotkeyListener:
    """F5 key to toggle recording. Falls back to Enter if no Accessibility."""

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
            print("[hotkey] Falling back to Enter key toggle.\n")
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
        print("[hotkey] Listening for Ctrl+Shift+R (global toggle)")

    def _on_press(self, key):
        from pynput import keyboard
        if key in (keyboard.Key.ctrl, keyboard.Key.ctrl_l, keyboard.Key.ctrl_r):
            self._pressed.add("ctrl")
        elif key in (keyboard.Key.shift, keyboard.Key.shift_l, keyboard.Key.shift_r):
            self._pressed.add("shift")
        elif hasattr(key, "char") and key.char == "\x12":  # Ctrl+Shift+R sends 0x12
            if "ctrl" in self._pressed and "shift" in self._pressed:
                threading.Thread(target=self._toggle, daemon=True).start()
        elif hasattr(key, "vk") and key.vk == 15:  # 'r' key vk code
            if "ctrl" in self._pressed and "shift" in self._pressed:
                threading.Thread(target=self._toggle, daemon=True).start()

    def _on_release(self, key):
        from pynput import keyboard
        if key in (keyboard.Key.ctrl, keyboard.Key.ctrl_l, keyboard.Key.ctrl_r):
            self._pressed.discard("ctrl")
        elif key in (keyboard.Key.shift, keyboard.Key.shift_l, keyboard.Key.shift_r):
            self._pressed.discard("shift")

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
