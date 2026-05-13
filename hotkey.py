import time
import sys
import threading


class HotkeyListener:
    """Detect double-tap Fn key to toggle recording (like Typeless).

    Uses macOS NSEvent global monitor. Requires:
    1. Accessibility permission for the terminal app (System Settings > Privacy > Accessibility)
    2. Disable system dictation shortcut (System Settings > Keyboard > Dictation > Shortcut > Off)
       Otherwise double-Fn triggers macOS dictation instead.

    Fallback: if running in a non-GUI environment or without permissions,
    use enter key in terminal to toggle.
    """

    DOUBLE_TAP_INTERVAL = 0.4

    def __init__(self, on_activate, on_deactivate):
        self._on_activate = on_activate
        self._on_deactivate = on_deactivate
        self._active = False
        self._last_fn_up_time = 0.0
        self._fn_is_down = False
        self._monitors = []

    def start(self):
        if self._try_native():
            return
        # Fallback to terminal input
        print("[hotkey] No GUI/accessibility access. Using Enter key to toggle.")
        t = threading.Thread(target=self._terminal_fallback, daemon=True)
        t.start()

    def _try_native(self) -> bool:
        try:
            from AppKit import (
                NSEvent,
                NSEventMaskFlagsChanged,
                NSRunLoop,
                NSDate,
            )

            monitor = NSEvent.addGlobalMonitorForEventsMatchingMask_handler_(
                NSEventMaskFlagsChanged,
                self._handle_flags_changed,
            )
            if monitor is None:
                return False
            self._monitors.append(monitor)

            local = NSEvent.addLocalMonitorForEventsMatchingMask_handler_(
                NSEventMaskFlagsChanged,
                lambda e: (self._handle_flags_changed(e), e)[1],
            )
            if local:
                self._monitors.append(local)

            def run_loop():
                loop = NSRunLoop.currentRunLoop()
                while True:
                    loop.runMode_beforeDate_(
                        "kCFRunLoopDefaultMode",
                        NSDate.dateWithTimeIntervalSinceNow_(0.2),
                    )

            t = threading.Thread(target=run_loop, daemon=True)
            t.start()
            print("[hotkey] Listening for double-tap Fn key.")
            return True
        except Exception as e:
            print(f"[hotkey] Native monitor failed: {e}")
            return False

    def _handle_flags_changed(self, event):
        from AppKit import NSEventModifierFlagFunction

        fn_down = bool(event.modifierFlags() & NSEventModifierFlagFunction)

        # Ignore if other modifiers are held
        other = event.modifierFlags() & (0x10000 | 0x20000 | 0x40000 | 0x80000 | 0x100000)
        if other:
            return

        if fn_down and not self._fn_is_down:
            self._fn_is_down = True
        elif not fn_down and self._fn_is_down:
            self._fn_is_down = False
            now = time.monotonic()
            elapsed = now - self._last_fn_up_time
            self._last_fn_up_time = now

            if elapsed < self.DOUBLE_TAP_INTERVAL:
                self._toggle()

    def _terminal_fallback(self):
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
        try:
            from AppKit import NSEvent
            for m in self._monitors:
                NSEvent.removeMonitor_(m)
        except Exception:
            pass
        self._monitors.clear()
