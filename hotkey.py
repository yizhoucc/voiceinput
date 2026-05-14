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

        self._listener = keyboard.Listener(on_press=self._on_press)
        self._listener.daemon = True
        self._listener.start()
        print("[hotkey] Listening for F5 (global toggle)")

    def _on_press(self, key):
        from pynput import keyboard
        if key == keyboard.Key.f5:
            self._toggle()

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
