"""CC-CEDICT word lookup and segmentation utilities.

Provides:
  - load_cedict_words(): load multi-char simplified forms from cedict.sqlite
  - words_at(): find all cedict words starting at a position
  - greedy_segment(): forward longest-match segmentation
  - reconstruct_segments(): rebuild segmentation from LLM picks + prefilled
  - load_merged_senses(): load sense-merged definitions (compact, LLM-distilled)
  - format_def(): format a single word's definition string
  - lookup_defs_for_candidates(): get defs dict for all candidate words in a task
"""

import json
import re
import sqlite3
from collections import defaultdict
from pathlib import Path

_CEDICT_DB = (
    Path(__file__).parent.parent.parent.parent.parent
    / "WenReader" / "Resources" / "cedict.sqlite"
)
_MERGED_SENSES_PATH = (
    Path(__file__).parent.parent.parent.parent  # ml/
    / "data" / "entries_after_merging.json"
)
_CJK_ONLY = re.compile(r'^[\u4e00-\u9fff]+$')


def load_cedict_words(db_path: Path = _CEDICT_DB) -> tuple[set[str], int]:
    """Load all multi-char pure-CJK simplified forms from cedict.sqlite.

    Returns (word_set, max_word_len).
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT simplified FROM entries WHERE length(simplified) >= 2")
    words = {w for (w,) in cur.fetchall() if _CJK_ONLY.match(w)}
    conn.close()
    max_len = max(len(w) for w in words)
    return words, max_len


def words_at(text: str, pos: int, word_set: set[str], max_len: int) -> list[str]:
    """Return all multi-char CEDICT words starting at text[pos], longest first."""
    matches = []
    for end in range(pos + 2, min(pos + max_len + 1, len(text) + 1)):
        candidate = text[pos:end]
        if candidate in word_set:
            matches.append(candidate)
    return list(reversed(matches))  # longest first


def greedy_segment(text: str, word_set: set[str], max_len: int) -> list[str]:
    """Greedy forward longest-match segmentation using CEDICT."""
    segments = []
    i = 0
    while i < len(text):
        matches = words_at(text, i, word_set, max_len)
        if matches:
            segments.append(matches[0])
            i += len(matches[0])
        else:
            segments.append(text[i])
            i += 1
    return segments


def reconstruct_segments(
    text: str, picks: dict[str, str], prefilled: dict[str, str] | None = None
) -> list[str]:
    """Reconstruct full segmentation from LLM picks + optional prefilled picks.

    Characters not covered by any pick are emitted as single-char segments.
    """
    segments = []
    i = 0
    pick_map = {int(pos): word for pos, word in picks.items()}
    if prefilled:
        for pos, word in prefilled.items():
            pick_map.setdefault(int(pos), word)

    while i < len(text):
        if i in pick_map:
            word = pick_map[i]
            segments.append(word)
            i += len(word)
        else:
            segments.append(text[i])
            i += 1
    return segments


# ---------------------------------------------------------------------------
# Merged-sense definitions
# ---------------------------------------------------------------------------


def load_merged_senses(path: Path = _MERGED_SENSES_PATH) -> dict[str, list[dict]]:
    """Load WSD sense-merged entries, keyed by simplified word.

    Returns {word: [entry, ...]} where each entry has pinyin, clusters,
    and trivial_senses. Multiple entries per word = different pronunciations.
    """
    lookup: dict[str, list[dict]] = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        entries = json.load(f)
    for e in entries:
        lookup[e["word"]].append(e)
    return lookup


def format_def(word: str, merged: dict[str, list[dict]], db_path: Path = _CEDICT_DB) -> str:
    """Format a compact definition string for a word (without word prefix).

    Prefers merged senses (LLM-distilled cluster labels). Falls back to
    raw cedict.sqlite glosses if the word isn't in merged senses.

    Returns e.g. "[shù zì] number; numeral; figure / digital (electronics etc)"
    or "" if no definition found.
    """
    entries = merged.get(word, [])
    if entries:
        parts = []
        for e in entries:
            pinyin = e["pinyin"]
            cluster_strs = []
            for c in e.get("clusters", []):
                if "en" in c:
                    cluster_strs.append(c["en"])
                else:
                    joined = "; ".join(c["senses"])
                    if len(joined) > 60:
                        joined = joined[:57] + "..."
                    cluster_strs.append(joined)
            # Skip trivial_senses (classifier info) — not useful for segmentation
            if cluster_strs:
                senses_str = " / ".join(cluster_strs)
                parts.append(f"[{pinyin}] {senses_str}")
        return " | ".join(parts) if parts else ""

    # Fallback: raw glosses from cedict.sqlite
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT e.pinyin, group_concat(g.gloss_text, '; ')
        FROM entries e
        JOIN sense_clusters sc ON sc.entry_id = e.id
        JOIN senses s ON s.sense_cluster_id = sc.id
        JOIN glosses g ON g.sense_id = s.id
        WHERE e.simplified = ?
        GROUP BY e.id
        """,
        (word,),
    )
    rows = cur.fetchall()
    conn.close()
    if not rows:
        return ""
    parts = []
    for pinyin, glosses in rows:
        # Truncate long raw gloss strings
        if len(glosses) > 80:
            glosses = glosses[:77] + "..."
        parts.append(f"[{pinyin}] {glosses}")
    return " | ".join(parts)


def lookup_defs_for_candidates(
    candidates: dict[str, list[str]], merged: dict[str, list[dict]]
) -> dict[str, str]:
    """Build a defs dict for all unique candidate words.

    Returns {word: definition_string} for every multi-char word in candidates.
    Words with no definition found are omitted.
    """
    unique_words: set[str] = set()
    for words in candidates.values():
        unique_words.update(words)

    defs = {}
    for word in sorted(unique_words):
        d = format_def(word, merged)
        if d:
            defs[word] = d
    return defs
