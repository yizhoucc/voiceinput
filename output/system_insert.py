import subprocess
import threading
import queue


class SystemTextInserter:
    """Insert/replace text at cursor via Cmd+Z undo + Cmd+V paste.

    Serialized queue ensures no concurrent osascript calls.
    Cmd+Z is instant (no cursor movement visible), one undo + one paste.
    """

    def __init__(self):
        self._q: queue.Queue[str | None] = queue.Queue()
        self._has_inserted = False
        self._worker = threading.Thread(target=self._work, daemon=True)
        self._worker.start()

    def replace_last(self, new_text: str):
        self._q.put(new_text)

    def reset(self):
        while not self._q.empty():
            try:
                self._q.get_nowait()
            except queue.Empty:
                break
        self._has_inserted = False

    def _work(self):
        while True:
            text = self._q.get()
            if text is None:
                break

            # Only process the latest entry
            latest = text
            while not self._q.empty():
                try:
                    latest = self._q.get_nowait()
                except queue.Empty:
                    break

            try:
                subprocess.run(["pbcopy"], input=latest.encode("utf-8"), check=True)

                if self._has_inserted:
                    # Undo previous paste + paste new (single AppleScript call)
                    subprocess.run([
                        "osascript", "-e",
                        '''tell application "System Events"
                            keystroke "z" using command down
                            delay 0.05
                            keystroke "v" using command down
                        end tell'''
                    ], check=True, capture_output=True, timeout=5)
                else:
                    subprocess.run([
                        "osascript", "-e",
                        'tell application "System Events" to keystroke "v" using command down'
                    ], check=True, capture_output=True, timeout=5)

                self._has_inserted = True
            except Exception as e:
                print(f"\n[insert] Error: {e}")
