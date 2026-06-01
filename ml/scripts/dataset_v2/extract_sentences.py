"""Step 1: Extract sentences from books (txt and epub).

Splits each book into paragraphs, splits paragraphs into sentences on
Chinese terminators. Filters out sentences containing non-CJK/non-punctuation
characters, or with fewer than 4 CJK characters.

Output schema: [{"source": str, "paragraphs": [[sent, ...], ...]}]
"""
import re
import warnings
from pathlib import Path

from filters import ALLOWED_CHARS, CJK_RE, MAX_SENTENCE_CHARS, brackets_valid, passes_filter

warnings.filterwarnings("ignore", category=UserWarning)

_TERMINATORS = re.compile(r'([。！？…]+[\u201d\u2019」』）\)]*|\n+)')


def extract_book(path: Path) -> dict:
    """Extract paragraphs from a .txt or .epub book file.

    Returns {"source": "<title>", "paragraphs": [[sent, ...], ...]}.
    """
    suffix = path.suffix.lower()
    if suffix == ".epub":
        paragraphs = _extract_epub(path)
    elif suffix == ".txt":
        paragraphs = _extract_txt(path)
    else:
        raise ValueError(f"Unsupported format: {suffix}")
    return {"source": path.stem[:60], "paragraphs": paragraphs}


def split_sentences(text: str) -> list[str]:
    """Split Chinese text on terminal punctuation (。！？…) and newlines.

    Keeps punctuation attached to the preceding sentence and handles
    closing quotes after terminators (e.g. 。"). Newlines are treated
    as sentence boundaries (the delimiter is discarded). Text is
    otherwise left alone — no whitespace normalization.
    """
    text = text.strip()
    if not text:
        return []
    parts = _TERMINATORS.split(text)
    sentences: list[str] = []
    i = 0
    while i < len(parts):
        sent = parts[i]
        if i + 1 < len(parts):
            delim = parts[i + 1]
            if '\n' not in delim:
                sent += delim
            i += 2
        else:
            i += 1
        sent = sent.strip()
        if not sent:
            continue
        if len(sent) > MAX_SENTENCE_CHARS:
            sent = sent[:MAX_SENTENCE_CHARS]
        if not passes_filter(sent):
            continue
        if not brackets_valid(sent):
            continue
        sentences.append(sent)
    return sentences


def _extract_epub(path: Path) -> list[list[str]]:
    import ebooklib
    from ebooklib import epub
    from bs4 import BeautifulSoup

    book = epub.read_epub(str(path), options={"ignore_ncx": True})
    paragraphs: list[list[str]] = []
    seen: set[str] = set()

    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        soup = BeautifulSoup(item.get_content(), "lxml")
        for tag in soup(["script", "style"]):
            tag.decompose()
        for p in soup.find_all(["p", "div", "h1", "h2", "h3", "h4"]):
            text = p.get_text().strip()
            if not text:
                continue
            sents = split_sentences(text)
            key = "\n".join(sents)
            if sents and key not in seen:
                seen.add(key)
                paragraphs.append(sents)

    return paragraphs


def _extract_txt(path: Path) -> list[list[str]]:
    for enc in ("utf-8", "gb18030"):
        try:
            text = path.read_text(encoding=enc)
            break
        except (UnicodeDecodeError, ValueError):
            continue
    else:
        print(f"  WARNING: could not decode {path.name}, skipping")
        return []

    paragraphs: list[list[str]] = []
    seen: set[str] = set()
    for para in re.split(r'\n\s*\n|\n', text):
        para = para.strip()
        if not para:
            continue
        sents = split_sentences(para)
        key = "\n".join(sents)
        if sents and key not in seen:
            seen.add(key)
            paragraphs.append(sents)

    return paragraphs
