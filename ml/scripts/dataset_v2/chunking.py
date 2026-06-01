"""Shared chunking logic for CWS and WSD task building.

Merges short consecutive sentences within a paragraph until hitting max_chars,
and truncates any single sentence that exceeds max_chars at the nearest
punctuation boundary.
"""
import re

_PUNCT = re.compile(r'[。，！？；：、…\u201c\u201d\u2018\u2019「」『』（）]')


def chunk_paragraphs(
    paragraphs: list[list[str]], max_chars: int = 64
) -> list[str]:
    """Merge short sentences, truncate long ones. Returns flat list of chunks.

    Rules:
    - Walk sentences within each paragraph, greedily concatenate until
      adding the next sentence would exceed max_chars.
    - Never merge across paragraph boundaries.
    - If a single sentence exceeds max_chars, truncate at the last
      punctuation mark within the limit (or hard-cut if none).
    """
    chunks: list[str] = []
    for para in paragraphs:
        buf = ""
        for sent in para:
            if len(sent) > max_chars:
                # Flush buffer first
                if buf:
                    chunks.append(buf)
                    buf = ""
                chunks.append(_truncate(sent, max_chars))
            elif len(buf) + len(sent) > max_chars:
                # Adding this sentence would overflow — flush and start new
                if buf:
                    chunks.append(buf)
                buf = sent
            else:
                buf += sent
        if buf:
            chunks.append(buf)
    return chunks


def _truncate(text: str, max_chars: int) -> str:
    """Truncate at last punctuation within max_chars, or hard-cut."""
    if len(text) <= max_chars:
        return text
    # Find last punctuation within limit
    last_punct = -1
    for m in _PUNCT.finditer(text[:max_chars]):
        last_punct = m.end()
    if last_punct > max_chars // 2:
        return text[:last_punct]
    return text[:max_chars]
