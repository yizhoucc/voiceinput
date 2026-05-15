"""Apple Speech STT using SFSpeechRecognizer + AVAudioEngine.

Uses AVAudioEngine's input tap for audio capture + real-time recognition.
This bypasses the manual AVAudioPCMBuffer creation issues with PyObjC.
"""
import threading
import numpy as np
import objc
from Foundation import NSLocale, NSRunLoop, NSDate
from stt.base import STTProvider
from config import config

Speech = objc.loadBundle("Speech",
    bundle_path="/System/Library/Frameworks/Speech.framework",
    module_globals=globals())
AVFoundation = objc.loadBundle("AVFoundation",
    bundle_path="/System/Library/Frameworks/AVFoundation.framework",
    module_globals=globals())

# Register block signatures
objc.registerMetaDataForSelector(
    b"SFSpeechRecognizer",
    b"recognitionTaskWithRequest:resultHandler:",
    {"arguments": {3: {"callable": {
        "retval": {"type": b"v"},
        "arguments": {0: {"type": b"^v"}, 1: {"type": b"@"}, 2: {"type": b"@"}},
    }}}},
)

objc.registerMetaDataForSelector(
    b"AVAudioNode",
    b"installTapOnBus:bufferSize:format:block:",
    {"arguments": {5: {"callable": {
        "retval": {"type": b"v"},
        "arguments": {0: {"type": b"^v"}, 1: {"type": b"@"}, 2: {"type": b"@"}},
    }}}},
)


class AppleSpeechSTT(STTProvider):
    """Word-by-word streaming STT using macOS native speech recognition.

    Uses AVAudioEngine for audio capture (bypasses our sounddevice capture)
    and SFSpeechRecognizer for real-time recognition.
    """

    def __init__(self, on_partial, on_final, on_commit=None):
        super().__init__(on_partial, on_final, on_commit)
        self._recognizer = None
        self._engine = None
        self._request = None
        self._task = None
        self._full_text = ""
        self._is_running = False
        self._runloop_thread = None

    def warmup(self):
        locale_id = "zh-CN"
        if config.primary_language == "en":
            locale_id = "en-US"
        locale = NSLocale.alloc().initWithLocaleIdentifier_(locale_id)
        self._recognizer = SFSpeechRecognizer.alloc().initWithLocale_(locale)
        self._engine = AVAudioEngine.alloc().init()

        if not self._recognizer.isAvailable():
            print("[apple-speech] Not available!")
            return

        # Start NSRunLoop for callbacks
        self._runloop_thread = threading.Thread(target=self._run_loop, daemon=True)
        self._runloop_thread.start()

        print(f"[apple-speech] Ready (locale={locale_id}, on-device={self._recognizer.supportsOnDeviceRecognition()})")

    def _run_loop(self):
        loop = NSRunLoop.currentRunLoop()
        while True:
            loop.runMode_beforeDate_("kCFRunLoopDefaultMode", NSDate.dateWithTimeIntervalSinceNow_(0.1))

    def feed_audio(self, audio: np.ndarray):
        """Start recognition if not already running.
        Note: Apple Speech uses its own audio capture via AVAudioEngine,
        so this method just triggers the start. The audio parameter from
        sounddevice is ignored — AVAudioEngine captures directly from mic.
        """
        if not self._is_running:
            self._start()

    def _start(self):
        if self._is_running:
            return
        self._is_running = True

        self._request = SFSpeechAudioBufferRecognitionRequest.alloc().init()
        self._request.setShouldReportPartialResults_(True)
        if self._recognizer.supportsOnDeviceRecognition():
            self._request.setRequiresOnDeviceRecognition_(True)

        input_node = self._engine.inputNode()
        record_format = input_node.outputFormatForBus_(0)

        def tap_handler(buffer, when):
            self._request.appendAudioPCMBuffer_(buffer)

        input_node.installTapOnBus_bufferSize_format_block_(0, 1024, record_format, tap_handler)

        def result_handler(result, error):
            if error:
                return
            if result:
                text = result.bestTranscription().formattedString()
                self._full_text = text
                if result.isFinal():
                    self.on_final(text)
                    self.on_commit(text)
                else:
                    self.on_partial(text)

        self._task = self._recognizer.recognitionTaskWithRequest_resultHandler_(
            self._request, result_handler)

        self._engine.prepare()
        self._engine.startAndReturnError_(None)

    def finalize(self):
        if not self._is_running:
            return
        self._engine.stop()
        self._engine.inputNode().removeTapOnBus_(0)
        self._request.endAudio()
        self._is_running = False

    def reset(self):
        if self._is_running:
            self.finalize()
        if self._task:
            self._task.cancel()
            self._task = None
        self._request = None
        self._full_text = ""
        self._is_running = False

    def prepare_finalize(self):
        pass
