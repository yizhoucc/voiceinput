import subprocess


class SystemTextInserter:
    """Append-only. Single Cmd+V paste. Never modifies existing text."""

    def paste(self, text: str):
        if not text.strip():
            return
        subprocess.run(["pbcopy"], input=text.encode("utf-8"), check=True)
        subprocess.run([
            "osascript", "-e",
            'tell application "System Events" to keystroke "v" using command down'
        ], check=True, capture_output=True, timeout=5)

    def reset(self):
        pass
