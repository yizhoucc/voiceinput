import sys


class TerminalOutput:
    def __init__(self):
        self._last_partial = ""

    def show_partial(self, text: str):
        sys.stdout.write(f"\r\033[K[...] {text}")
        sys.stdout.flush()
        self._last_partial = text

    def show_final(self, text: str):
        sys.stdout.write(f"\r\033[K[done] {text}\n")
        sys.stdout.flush()
        self._last_partial = ""

    def show_status(self, msg: str):
        sys.stdout.write(f"\r\033[K{msg}")
        sys.stdout.flush()

    def clear_line(self):
        sys.stdout.write("\r\033[K")
        sys.stdout.flush()
