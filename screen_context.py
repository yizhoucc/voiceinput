"""Extract keywords from current screen via macOS Vision OCR."""
import subprocess
import re
import objc
from Quartz import CGImageSourceCreateWithURL, CGImageSourceCreateImageAtIndex
from Foundation import NSURL

UI_STOPWORDS = frozenset({
    "file", "edit", "view", "window", "help", "session", "shell", "scripts",
    "profiles", "alert", "running", "error", "bash", "import", "def", "class",
    "return", "true", "false", "none", "self", "print", "for", "while",
    "else", "try", "except", "from", "with", "the", "and", "that", "this",
    "are", "was", "were", "been", "have", "has", "had", "will", "would",
    "could", "should", "not", "but", "can", "all", "may", "its", "than",
    "into", "also", "out", "about", "ctrl", "shift", "esc", "tab", "enter",
    "click", "press", "open", "close", "new", "save", "copy", "paste",
    "undo", "redo", "find", "replace", "select", "delete", "insert",
    "home", "end", "page", "zoom", "scroll", "menu", "toolbar", "status",
    "input", "output", "source", "code", "line", "size", "type", "mode",
    "your", "you", "our", "his", "her", "its", "they", "them", "what",
    "when", "where", "which", "who", "how", "just", "more", "most", "some",
    "other", "each", "every", "both", "few", "many", "much", "such",
})


def _is_plausible(word: str) -> bool:
    if re.match(r"^\d+\.?\d*$", word):
        return True
    if re.match(r"^[A-Z][A-Z0-9]{1,5}$", word):
        return len(word) <= 4 or any(c in "AEIOU" for c in word)
    lower = word.lower()
    if not any(c in "aeiou" for c in lower):
        return False
    if len(word) <= 2:
        return False
    if re.search(r"[a-z][A-Z]", word) and not word[0].isupper():
        return False
    return True


def get_screen_keywords() -> list[str]:
    """Screenshot → OCR → filtered keywords. Returns list of unique terms."""
    try:
        subprocess.run(["screencapture", "-x", "/tmp/_voiceinput_screen.png"],
                       capture_output=True, timeout=3)

        url = NSURL.fileURLWithPath_("/tmp/_voiceinput_screen.png")
        source = CGImageSourceCreateWithURL(url, None)
        if source is None:
            return []
        image = CGImageSourceCreateImageAtIndex(source, 0, None)

        Vision = objc.loadBundle("Vision",
            bundle_path="/System/Library/Frameworks/Vision.framework",
            module_globals=globals())

        request = VNRecognizeTextRequest.alloc().init()
        request.setRecognitionLevel_(1)
        request.setRecognitionLanguages_(["zh-Hans", "en"])

        handler = VNImageRequestHandler.alloc().initWithCGImage_options_(image, None)
        handler.performRequests_error_([request], None)

        results = request.results()
        if not results:
            return []

        all_text = " ".join(r.text() for r in results)

        # Extract candidates
        en_words = re.findall(r"\b[A-Za-z]{2,}\b", all_text)
        acronyms = re.findall(r"\b[A-Z][A-Z0-9]{1,}\b", all_text)
        zh_words = re.findall(r"[一-鿿]{2,}", all_text)
        candidates = set(en_words + acronyms + zh_words)

        # Filter
        keywords = []
        for w in candidates:
            if w.lower() in UI_STOPWORDS:
                continue
            if w.islower() and len(w) < 4:
                continue
            if not all(ord(c) < 128 or "一" <= c <= "鿿" for c in w):
                continue
            if re.match(r"^[A-Za-z]+$", w) and not _is_plausible(w):
                continue
            keywords.append(w)

        return sorted(set(keywords))

    except Exception:
        return []
