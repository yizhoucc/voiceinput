"""Floating overlay window for showing partial STT text in real-time."""
import threading
import tkinter as tk
import queue


class Overlay:
    """Small borderless always-on-top window at the bottom of screen.

    Shows the current partial (uncommitted) text.
    Committed text disappears from overlay (it's in the editor now).
    """

    def __init__(self):
        self._q: queue.Queue[str | None] = queue.Queue()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def show(self, text: str):
        self._q.put(text)

    def hide(self):
        self._q.put("")

    def _run(self):
        self._root = tk.Tk()
        self._root.overrideredirect(True)
        self._root.attributes("-topmost", True)
        self._root.attributes("-alpha", 0.85)
        self._root.configure(bg="#1e1e1e")

        screen_w = self._root.winfo_screenwidth()
        screen_h = self._root.winfo_screenheight()
        win_w = min(800, screen_w - 100)
        win_h = 60
        x = (screen_w - win_w) // 2
        y = screen_h - 120
        self._root.geometry(f"{win_w}x{win_h}+{x}+{y}")

        self._label = tk.Label(
            self._root,
            text="",
            font=("SF Pro", 14),
            fg="#e0e0e0",
            bg="#1e1e1e",
            wraplength=win_w - 20,
            justify="left",
            anchor="w",
        )
        self._label.pack(fill="both", expand=True, padx=10, pady=5)

        self._root.withdraw()  # start hidden
        self._poll()
        self._root.mainloop()

    def _poll(self):
        try:
            while True:
                text = self._q.get_nowait()
                if text:
                    # Show only the last ~100 chars to keep it readable
                    display = text[-100:] if len(text) > 100 else text
                    self._label.config(text=display)
                    self._root.deiconify()
                else:
                    self._root.withdraw()
        except queue.Empty:
            pass
        self._root.after(100, self._poll)
