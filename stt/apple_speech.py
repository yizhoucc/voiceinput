"""Apple Speech STT provider using macOS SFSpeechRecognizer.

True word-by-word streaming via on-device recognition.
Each recognized word triggers on_partial immediately.
"""
import threading
import numpy as np
import objc
from Foundation import NSLocale
from stt.base import STTProvider
from config import config

Speech = objc.loadBundle("Speech",
    bundle_path="/System/Library/Frameworks/Speech.framework",
    module_globals=globals())
AVFoundation = objc.loadBundle("AVFoundation",
    bundle_path="/System/Library/Frameworks/AVFoundation.framework",
    module_globals=globals())


class AppleSpeechSTT(STTProvider):
    """Word-by-word streaming STT using macOS SFSpeechRecognizer."""

    def __init__(self, on_partial, on_final, on_commit=None):
        super().__init__(on_partial, on_final, on_commit)
        self._recognizer = None
        self._request = None
        self._task = None
        self._lock = threading.Lock()
        self._full_text = ""
        self._is_running = False

    def warmup(self):
        locale_id = "zh-CN"
        if config.primary_language == "en":
            locale_id = "en-US"
        locale = NSLocale.alloc().initWithLocaleIdentifier_(locale_id)
        self._recognizer = SFSpeechRecognizer.alloc().initWithLocale_(locale)

        if not self._recognizer.isAvailable():
            print("[apple-speech] Recognizer not available!")
            return

        self._recognizer.setDefaultTaskHint_(0)  # dictation hint
        print(f"[apple-speech] Ready (locale={locale_id}, on-device={self._recognizer.supportsOnDeviceRecognition()})")

    def feed_audio(self, audio: np.ndarray):
        with self._lock:
            if not self._is_running:
                self._start_recognition()
                self._is_running = True

            # Convert float32 [-1,1] to int16 PCM
            flat = audio.flatten()
            pcm = (flat * 32767).clip(-32768, 32767).astype(np.int16)

            # Create AVAudioPCMBuffer and append
            from AppKit import NSData
            data = NSData.dataWithBytes_length_(pcm.tobytes(), len(pcm.tobytes()))
            self._request.appendAudioPCMBuffer_(self._create_buffer(pcm))

    def _create_buffer(self, pcm_int16):
        """Create AVAudioPCMBuffer from int16 PCM data."""
        format_ = AVAudioFormat.alloc().initWithCommonFormat_sampleRate_channels_interleaved_(
            1,  # PCMFormatInt16
            float(config.sample_rate),
            1,
            True
        )
        frame_count = len(pcm_int16)
        buffer = AVAudioPCMBuffer.alloc().initWithPCMFormat_frameCapacity_(format_, frame_count)
        buffer.setFrameLength_(frame_count)

        # Copy data into buffer
        import ctypes
        dst = buffer.int16ChannelData()
        src = pcm_int16.ctypes.data_as(ctypes.POINTER(ctypes.c_int16))
        ctypes.memmove(dst[0], src, frame_count * 2)

        return buffer

    def _start_recognition(self):
        self._request = SFSpeechAudioBufferRecognitionRequest.alloc().init()
        self._request.setShouldReportPartialResults_(True)

        if self._recognizer.supportsOnDeviceRecognition():
            self._request.setRequiresOnDeviceRecognition_(True)

        def handle_result(result, error):
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
            self._request, handle_result
        )

    def finalize(self):
        with self._lock:
            if self._request:
                self._request.endAudio()
            if self._full_text:
                self.on_commit(self._full_text)
                self.on_final(self._full_text)
            self._is_running = False

    def reset(self):
        with self._lock:
            if self._task:
                self._task.cancel()
                self._task = None
            self._request = None
            self._full_text = ""
            self._is_running = False

    def prepare_finalize(self):
        pass
