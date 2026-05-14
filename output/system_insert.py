import subprocess
import threading
import queue


class SystemTextInserter:
    """Incremental text insertion using diff + AppleScript (synchronous).

    Tracks exactly what text is in the editor. On each update:
    1. Find common prefix between old and new text
    2. Backspace to remove old suffix (via AppleScript, synchronous)
    3. Paste new suffix (via AppleScript, same call)

    AppleScript's tell block executes commands sequentially,
    eliminating the CGEvent async timing problem.
    """

    def __init__(self):
        self._q: queue.Queue[str | None] = queue.Queue()
        self._text_in_editor = ""
        self._worker = threading.Thread(target=self._work, daemon=True)
        self._worker.start()

    def update(self, text: str):
        self._q.put(text)

    def commit(self, text: str):
        pass

    def reset(self):
        while not self._q.empty():
            try:
                self._q.get_nowait()
            except queue.Empty:
                break
        self._text_in_editor = ""

    def _work(self):
        while True:
            text = self._q.get()
            if text is None:
                break

            latest = text
            while not self._q.empty():
                try:
                    latest = self._q.get_nowait()
                except queue.Empty:
                    break

            try:
                self._apply_diff(latest)
            except Exception as e:
                print(f"\n[insert] Error: {e}")

    def _apply_diff(self, new_text: str):
        old = self._text_in_editor

        # Find common prefix
        prefix_len = 0
        for i in range(min(len(old), len(new_text))):
            if old[i] == new_text[i]:
                prefix_len += 1
            else:
                break

        delete_count = len(old) - prefix_len
        append_text = new_text[prefix_len:]

        if delete_count == 0 and not append_text:
            return

        # Copy new suffix to clipboard first
        if append_text:
            subprocess.run(["pbcopy"], input=append_text.encode("utf-8"), check=True)

        # Build AppleScript: backspace + paste in one synchronous call
        script_parts = ['tell application "System Events"']

        if delete_count > 0:
            if delete_count <= 20:
                for _ in range(delete_count):
                    script_parts.append('    key code 51')
            else:
                script_parts.append(f'    repeat {delete_count} times')
                script_parts.append('        key code 51')
                script_parts.append('    end repeat')

        if append_text:
            if delete_count > 0:
                script_parts.append('    delay 0.02')
            script_parts.append('    keystroke "v" using command down')

        script_parts.append('end tell')
        script = "\n".join(script_parts)

        subprocess.run(
            ["osascript", "-e", script],
            check=True, capture_output=True, timeout=10
        )

        self._text_in_editor = new_text
