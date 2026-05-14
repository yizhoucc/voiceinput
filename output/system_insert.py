import subprocess


class SystemTextInserter:
    """Append-only insertion. One Cmd+V per committed segment. Never modifies."""

    def append(self, text: str):
        if not text:
            return
        subprocess.run(["pbcopy"], input=text.encode("utf-8"), check=True)
        subprocess.run([
            "osascript", "-e",
            'tell application "System Events" to keystroke "v" using command down'
        ], check=True, capture_output=True, timeout=5)

    def reset(self):
        pass
