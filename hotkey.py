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
    """Two hotkeys:
    Ctrl+Shift+R = smart mode (speaker-change auto-commit)
    Ctrl+Shift+E = manual mode (only commit on stop)

    Falls back to Enter/E in terminal if no Accessibility.
    """

    def __init__(self, on_activate, on_deactivate, on_activate_manual, on_deactivate_manual):
        self._on_activate = on_activate
        self._on_deactivate = on_deactivate
        self._on_activate_manual = on_activate_manual
        self._on_deactivate_manual = on_deactivate_manual
        self._active = False
        self._mode = None  # "smart" or "manual"
        self._use_pynput = False

    def start(self):
        if check_accessibility():
            self._use_pynput = True
            self._start_pynput()
        else:
            request_accessibility()
            print("[hotkey] Falling back to terminal keys.\n")
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
        print("[hotkey] Ctrl+Shift+R = smart mode (speaker-change auto-commit)")
        print("[hotkey] Ctrl+Shift+E = manual mode (commit only on stop)")

    def _check_hotkey(self, key):
        from pynput import keyboard
        if not ("ctrl" in self._pressed and "shift" in self._pressed):
            return None
        # Ctrl+Shift+R = 0x12, Ctrl+Shift+E = 0x05
        if hasattr(key, "char"):
            if key.char == "\x12":
                return "smart"
            elif key.char == "\x05":
                return "manual"
        if hasattr(key, "vk"):
            if key.vk == 15:  # r
                return "smart"
            elif key.vk == 14:  # e
                return "manual"
        return None

    def _on_press(self, key):
        from pynput import keyboard
        if key in (keyboard.Key.ctrl, keyboard.Key.ctrl_l, keyboard.Key.ctrl_r):
            self._pressed.add("ctrl")
        elif key in (keyboard.Key.shift, keyboard.Key.shift_l, keyboard.Key.shift_r):
            self._pressed.add("shift")
        else:
            mode = self._check_hotkey(key)
            if mode:
                threading.Thread(target=self._toggle, args=(mode,), daemon=True).start()

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
                line = input()
                if line.strip().lower() == "e":
                    self._toggle("manual")
                else:
                    self._toggle("smart")
            except EOFError:
                break

    def _toggle(self, mode):
        if self._active:
            if self._mode == "manual":
                self._on_deactivate_manual()
            else:
                self._on_deactivate()
            self._active = False
            self._mode = None
        else:
            self._active = True
            self._mode = mode
            if mode == "manual":
                self._on_activate_manual()
            else:
                self._on_activate()

    def stop(self):
        if self._use_pynput and hasattr(self, "_listener"):
            self._listener.stop()
