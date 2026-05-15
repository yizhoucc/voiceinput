"""Load and manage custom dictionary for recognition and polishing."""
from pathlib import Path

DEFAULT_DICT = Path(__file__).parent / "dictionary.txt"


def load_dictionary(path: Path = DEFAULT_DICT) -> tuple[list[str], dict[str, str]]:
    """Load dictionary file. Returns (terms, corrections).

    terms: list of words/phrases for whisper prompt
    corrections: dict of wrong→right for auto-correction
    """
    terms = []
    corrections = {}

    if not path.exists():
        return terms, corrections

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if " -> " in line:
            wrong, right = line.split(" -> ", 1)
            corrections[wrong.strip()] = right.strip()
            terms.append(right.strip())
        else:
            terms.append(line)

    return terms, corrections
