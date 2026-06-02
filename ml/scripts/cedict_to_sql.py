"""Build cedict.sqlite from CC-CEDICT with WSD sense cluster embeddings.

Schema: entries ← sense_clusters ← senses ← glosses
"""

import argparse
import json
import logging
import re
import sqlite3
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

SCRIPT_DIR = Path(__file__).resolve().parent
ML_DIR = SCRIPT_DIR.parent
INPUT_CEDICT_PATH = ML_DIR.parent / "cedict" / "cedict_ts.u8"
OUTPUT_DB_PATH = ML_DIR.parent / "WenReader" / "Resources" / "cedict.sqlite"

ENTRIES_PATH = ML_DIR / "data" / "entries_after_merging.json"
TRANSLATION_CACHE_PATH = ML_DIR / "data" / "translation_cache.json"

logger = logging.getLogger(__name__)

LINE_RE = re.compile(
    r"^(\S+)\s+(\S+)\s+\[(.+?)]\s+/(.*)/\s*$"
)


# -- Parsing ------------------------------------------------------------------

def parse_cedict(path):
    """Yield (trad, simp, pinyin, senses_raw) for each valid CC-CEDICT line."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            m = LINE_RE.match(line)
            if m:
                yield m.groups()


def parse_senses(senses_raw):
    """Parse 'a/CL:b/c' into [{is_classifier, glosses, raw}]."""
    out = []
    for part in senses_raw.split("/"):
        part = part.strip()
        if not part:
            continue
        is_cl = part.startswith("CL:")
        if is_cl:
            part = part[3:]
        glosses = [g.strip() for g in part.split(";") if g.strip()]
        if glosses:
            out.append({"is_classifier": int(is_cl), "glosses": glosses, "raw": part})
    return out


# -- Cluster lookup from merging JSON -----------------------------------------

def build_cluster_lookup(entries_path):
    """Returns:
        gloss_to_cluster: (word, pinyin, english_gloss) → (cluster_idx, is_trivial)
        entry_clusters:   (word, pinyin) → [{senses: [str], is_trivial: bool}, ...]
    """
    with open(entries_path, encoding="utf-8") as f:
        entries = json.load(f)

    gloss_to_cluster = {}
    entry_clusters = {}

    for e in entries:
        w, p = e["word"], e["pinyin"]
        clusters = e.get("clusters", [])
        trivial = e.get("trivial_senses", [])

        clist = []
        for ci, c in enumerate(clusters):
            for s in c["senses"]:
                gloss_to_cluster[(w, p, s)] = (ci, False)
            clist.append({"senses": c["senses"], "is_trivial": False})

        if trivial:
            ti = len(clusters)
            for s in trivial:
                gloss_to_cluster[(w, p, s)] = (ti, True)
            clist.append({"senses": trivial, "is_trivial": True})

        entry_clusters[(w, p)] = clist

    return gloss_to_cluster, entry_clusters


# -- DB schema ----------------------------------------------------------------

def init_db(db_path):
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(str(db_path))
    c = conn.cursor()
    c.executescript("""
        CREATE TABLE entries (
            id INTEGER PRIMARY KEY, traditional TEXT NOT NULL,
            simplified TEXT NOT NULL, pinyin TEXT NOT NULL,
            UNIQUE(traditional, simplified, pinyin));
        CREATE INDEX idx_entries_simplified ON entries(simplified);
        CREATE INDEX idx_entries_traditional ON entries(traditional);

        CREATE TABLE sense_clusters (
            id INTEGER PRIMARY KEY, entry_id INTEGER NOT NULL REFERENCES entries(id),
            is_trivial INTEGER NOT NULL DEFAULT 0, embedding BLOB);
        CREATE INDEX idx_clusters_entry ON sense_clusters(entry_id);

        CREATE TABLE senses (
            id INTEGER PRIMARY KEY,
            sense_cluster_id INTEGER NOT NULL REFERENCES sense_clusters(id),
            is_classifier INTEGER NOT NULL DEFAULT 0);
        CREATE INDEX idx_senses_cluster ON senses(sense_cluster_id);

        CREATE TABLE glosses (
            id INTEGER PRIMARY KEY, sense_id INTEGER NOT NULL REFERENCES senses(id),
            gloss_text TEXT NOT NULL);
        CREATE INDEX idx_glosses_sense ON glosses(sense_id);
    """)
    conn.commit()
    return conn


# -- Insert entries + clusters + senses + glosses -----------------------------

def populate_db(conn, gloss_to_cluster, entry_clusters):
    cur = conn.cursor()
    cur.execute("BEGIN")

    for trad, simp, pinyin, senses_raw in parse_cedict(INPUT_CEDICT_PATH):
        cur.execute(
            "INSERT OR IGNORE INTO entries (traditional, simplified, pinyin) VALUES (?,?,?)",
            (trad, simp, pinyin))
        entry_id = cur.lastrowid
        if entry_id == 0:
            continue

        ec = entry_clusters.get((simp, pinyin), [])

        # Create one sense_cluster row per cluster
        cluster_db_ids = {}
        for ci, info in enumerate(ec):
            cur.execute(
                "INSERT INTO sense_clusters (entry_id, is_trivial) VALUES (?,?)",
                (entry_id, int(info["is_trivial"])))
            cluster_db_ids[ci] = cur.lastrowid

        # Fallback: entry has no merge info → one default cluster
        if not ec:
            cur.execute(
                "INSERT INTO sense_clusters (entry_id, is_trivial) VALUES (?,0)",
                (entry_id,))
            default_cid = cur.lastrowid

        # Insert senses and glosses
        for sense in parse_senses(senses_raw):
            # Match sense to cluster via gloss lookup
            cid = None
            for g in sense["glosses"]:
                hit = gloss_to_cluster.get((simp, pinyin, g))
                if hit is not None:
                    cid = cluster_db_ids.get(hit[0])
                    break
            if cid is None:
                hit = gloss_to_cluster.get((simp, pinyin, sense["raw"]))
                if hit is not None:
                    cid = cluster_db_ids.get(hit[0])
            if cid is None:
                cid = default_cid if not ec else next(iter(cluster_db_ids.values()))

            cur.execute(
                "INSERT INTO senses (sense_cluster_id, is_classifier) VALUES (?,?)",
                (cid, sense["is_classifier"]))
            sid = cur.lastrowid
            for g in sense["glosses"]:
                cur.execute("INSERT INTO glosses (sense_id, gloss_text) VALUES (?,?)", (sid, g))

    conn.commit()

    # Promote trivial senses: if a trivial cluster is the ONLY cluster for its
    # entry and the word has other entries, promote it so it gets an embedding
    # and can compete in cross-entry WSD ranking.
    cur.execute("""
        UPDATE sense_clusters SET is_trivial = 0
        WHERE is_trivial = 1
          AND (SELECT COUNT(*) FROM sense_clusters sc2
               WHERE sc2.entry_id = sense_clusters.entry_id) = 1
          AND entry_id IN (
              SELECT e1.id FROM entries e1
              WHERE (SELECT COUNT(*) FROM entries e2
                     WHERE e2.simplified = e1.simplified) > 1)
    """)
    promoted = cur.rowcount
    conn.commit()
    logger.info("Promoted %d trivial clusters for cross-entry WSD.", promoted)


# -- Embedding generation -----------------------------------------------------

def generate_embeddings(conn, entry_clusters, translation_cache, wsd_model_dir):
    """Encode cluster-level Chinese sense text and store as BLOBs."""
    cur = conn.cursor()

    # Find which words are polysemous (≥2 non-trivial clusters across all entries)
    cur.execute("""
        SELECT e.simplified, sc.id, sc.is_trivial, e.pinyin, sc.entry_id
        FROM sense_clusters sc
        JOIN entries e ON e.id = sc.entry_id
        ORDER BY sc.entry_id, sc.id
    """)
    rows = cur.fetchall()

    # Group by word to determine polysemy
    word_clusters = {}  # word → [(sc_id, is_trivial, pinyin, entry_id)]
    for simp, sc_id, is_trivial, pinyin, eid in rows:
        word_clusters.setdefault(simp, []).append((sc_id, is_trivial, pinyin, eid))

    # Precompute Chinese text for each (word, pinyin, cluster_idx)
    # This is the text we embed — same for all DB entries with that word+pinyin.
    cluster_texts = {}  # (word, pinyin, idx) → chinese_text
    for (word, pinyin), ec in entry_clusters.items():
        for idx, info in enumerate(ec):
            zh_parts = []
            for sense_en in info["senses"]:
                zh = translation_cache.get(f"{word}|{pinyin}|{sense_en}")
                if zh:
                    zh_parts.append(zh)
            if zh_parts:
                cluster_texts[(word, pinyin, idx)] = "；".join(zh_parts)

    # Collect DB clusters that need embeddings
    to_encode = []  # (sc_id, chinese_text)
    for word, clusters in word_clusters.items():
        non_trivial = [c for c in clusters if not c[1]]
        if len(non_trivial) < 2:
            continue

        # Group by entry_id to determine position within each entry
        by_entry = {}
        for sc_id, is_trivial, pinyin, eid in clusters:
            by_entry.setdefault(eid, []).append((sc_id, is_trivial, pinyin))

        for eid, entry_rows in by_entry.items():
            pinyin = entry_rows[0][2]
            for idx, (sc_id, is_trivial, _) in enumerate(entry_rows):
                if is_trivial:
                    continue
                text = cluster_texts.get((word, pinyin, idx))
                if text:
                    to_encode.append((sc_id, text))

    logger.info("Encoding %d cluster embeddings...", len(to_encode))
    if not to_encode:
        return

    model = SentenceTransformer(str(wsd_model_dir), device="cpu")
    texts = [t for _, t in to_encode]
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True, batch_size=256)

    for i, (sc_id, _) in enumerate(to_encode):
        emb = embeddings[i].astype(np.float32)
        # Quantize to int8: scale (float32, 4 bytes) + values (int8, dim bytes)
        scale = float(np.max(np.abs(emb)))
        if scale == 0:
            quantized = np.zeros(len(emb), dtype=np.int8)
        else:
            quantized = np.clip(np.round(emb / scale * 127), -127, 127).astype(np.int8)
        blob = np.array([scale], dtype=np.float32).tobytes() + quantized.tobytes()
        cur.execute("UPDATE sense_clusters SET embedding = ? WHERE id = ?",
                    (blob, sc_id))
    conn.commit()
    emb_dim = embeddings.shape[1] if len(to_encode) > 0 else 0
    blob_size = 4 + emb_dim
    logger.info("Embedded %d clusters (int8 quantized, %d bytes each).", len(to_encode), blob_size)


# -- Main ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wsd-model", help="Path to WSD model directory")
    args = parser.parse_args()
    wsd_model_dir = Path(args.wsd_model) if args.wsd_model else ML_DIR / "models" / "wsd_finetuned" / "final"

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    gloss_to_cluster, entry_clusters = build_cluster_lookup(ENTRIES_PATH)
    with open(TRANSLATION_CACHE_PATH, encoding="utf-8") as f:
        translation_cache = json.load(f)

    conn = init_db(OUTPUT_DB_PATH)
    populate_db(conn, gloss_to_cluster, entry_clusters)
    generate_embeddings(conn, entry_clusters, translation_cache, wsd_model_dir)

    conn.close()
    logger.info("Done: %s", OUTPUT_DB_PATH)


if __name__ == "__main__":
    main()
