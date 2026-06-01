"""Shared sentence-filtering logic for the dataset_v2 pipeline.

Used by extract_sentences.py, cws/icwb2.py, and analyze_icwb2.py.
"""
import re

MAX_SENTENCE_CHARS = 500
_MIN_CJK_CHARS = 4

CJK_RE = re.compile(r'[\u4e00-\u9fff]')

# Allowed: CJK characters + explicit punctuation whitelist
ALLOWED_CHARS = re.compile(
    r'^[\u4e00-\u9fff'
    r'。，、；：！？'                # period, comma, enumeration comma, semicolon, colon, excl, question
    r'…'                            # ellipsis
    r'\u201c\u201d\u2018\u2019'     # smart quotes ""''
    r'「」『』'                      # corner brackets
    r'（）'                          # fullwidth parens
    r'——'                            # em dash
    r'《》'                          # book title marks
    r']+$'
)

_BRACKET_PAIRS = [
    ('\u201c', '\u201d'),  # ""
    ('\u2018', '\u2019'),  # ''
    ('「', '」'),
    ('『', '』'),
    ('（', '）'),
    ('《', '》'),
]


def brackets_valid(text: str) -> bool:
    """Check all bracket pairs are matched (single depth, no nesting)."""
    for opener, closer in _BRACKET_PAIRS:
        depth = 0
        for ch in text:
            if ch == opener:
                depth += 1
                if depth > 1:
                    return False
            elif ch == closer:
                depth -= 1
                if depth < 0:
                    return False
        if depth != 0:
            return False
    return True


def passes_filter(text: str) -> bool:
    """Return True if text contains only allowed chars and enough CJK."""
    if not ALLOWED_CHARS.match(text):
        return False
    if len(CJK_RE.findall(text)) < _MIN_CJK_CHARS:
        return False
    return True
