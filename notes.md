# ML Pipeline & App Integration Notes

Work on integrating the trained CWS and WSD ML models into the WenReader iOS app is starting. This document will capture schema decisions, WSD embedding findings, and integration notes as the work progresses.

## CoreML Export — Compatibility Issues (2025-02-27)

### The Problem

Exporting our fine-tuned BERT models (CWS: `BertForTokenClassification`, WSD: `SentenceTransformer` wrapping `gte-base-zh`) from PyTorch to CoreML `.mlpackage` format is blocked by a compatibility gap between `transformers` v5 and `coremltools`.

### What Was Tried

1. **Direct PyTorch → CoreML via `torch.jit.trace` + `coremltools` 9.0**
   - Fails: `NotImplementedError: PyTorch convert function for op 'new_ones' not implemented`
   - The `new_ones` op comes from `transformers` v5's refactored `_create_attention_masks()` in `modeling_bert.py`
   - Even with `attn_implementation="eager"`, the mask creation still uses `new_ones`

2. **`torch.export` + `coremltools` 9.0**
   - Same `new_ones` issue — it's in the exported graph regardless of export method
   - Adding a custom decomposition for `new_ones` (rewriting to `torch.ones`) gets past that op, but then `__and__` (bitwise AND) is also unsupported

3. **PyTorch → ONNX → CoreML via `coremltools` 9.0**
   - ONNX export works fine (`torch.onnx.export` handles all ops)
   - But `coremltools` 9.0 dropped ONNX as a source format entirely
   - Valid sources are only: `"auto"`, `"tensorflow"`, `"pytorch"`, `"milinternal"`

4. **PyTorch → ONNX → CoreML via `coremltools` 8.3**
   - Also dropped ONNX support (same valid sources as 9.0)
   - Additionally has a numpy 2.x compatibility bug in its `slice` op converter

5. **`coremltools` 7.x (which had ONNX support)**
   - Can't install: `protobuf` version conflict between `coremltools` 7.x (`<=4.0.0`) and `onnx` 1.20 (`>=4.25.1`)

### Root Cause

- `transformers` v5 refactored attention mask creation to use `new_ones` and bitwise ops
- `coremltools` 9.0 doesn't support these ops in its PyTorch frontend
- Apple removed ONNX-to-CoreML conversion from `coremltools` starting in v8, pushing users toward direct PyTorch conversion
- But their PyTorch op coverage (~70% as of coremltools 8.0 docs) doesn't cover the new transformers v5 patterns

### Why Xcode Can Still Convert ONNX but coremltools Can't

Xcode's "Create ML" / model converter uses a compiled C++ pipeline that still supports ONNX → CoreML. The Python `coremltools` library is a separate codebase that dropped ONNX support to focus on PyTorch. So the capability exists in Apple's toolchain, just not in the Python package.

### Current State

- ONNX export works: `ml/output/onnx/cws_encoder.onnx` was successfully generated
- The ONNX files can be converted to CoreML via Xcode's GUI or `xcrun coremlcompiler`
- The Python-only pipeline (export + verify in one script) is blocked

### Possible Solutions (TODO)

1. **Use `xcrun coremlcompiler` CLI** to convert ONNX → CoreML from the command line (avoids Xcode GUI)
2. **Pin `transformers` to v4.x** just for the export step (v4 doesn't have the `new_ones` issue)
3. **Use HuggingFace `optimum`** which has its own CoreML exporter that handles these compatibility issues
4. **Wait for coremltools update** — Apple may add `new_ones` support in a future release
5. **Manually patch the exported graph** to replace `new_ones` with supported ops before conversion

### Resolution

Downpinned `transformers` to `>=4.41.0,<5.0.0` and Python to 3.12. None of the training or inference scripts used v5-specific features — the v5 pin was unnecessary. `sentence-transformers>=5.2.2` only requires `transformers>=4.41.0,<6.0.0` so it's compatible.

Also downpinned Python from 3.14 to 3.12 because `coremltools` 9.0 doesn't ship native bindings (`libcoremlpython`, `libmilstoragepython`) for Python 3.14 — conversion runs but can't serialize weights without them.

With transformers v4 + Python 3.12, the direct `torch.jit.trace` → `ct.convert()` path works. CWS and WSD both export and verify successfully. Numerical tolerance is ~1-2% for CWS (due to MIL float optimization) but argmax predictions are identical. WSD tolerance is ~1e-3.


## Legacy CEDICT SQL Schema

The original `cedict_to_sql.py` produced a flat `cedict_entries` table with columns: `id`, `traditional`, `simplified`, `pinyin`, and `senses_raw`. The `senses_raw` column stored the entire definition string as-is from the CEDICT file — senses delimited by `/`, glosses within each sense delimited by `;`.

Example `senses_raw` value for 打 (dǎ):
```
to hit/to strike/to break/to fight/...
```

Limitations:
- **Runtime parsing required.** Every dictionary lookup had to split `senses_raw` by `/` and `;` on the Swift side, wasting CPU on repeated string operations.
- **No per-sense metadata.** Classifier senses (`CL:` prefix) couldn't be flagged without parsing. No way to attach embeddings or scores to individual senses.
- **No structured queries.** Couldn't query "all glosses for sense 2 of entry X" without parsing the blob first.
- **Inline references mixed in.** Patterns like `個|个[ge4]` were embedded in the raw string, requiring regex parsing at display time (this part is unchanged — the app still parses inline refs at display time).

## New Normalized Schema Rationale

The rewrite splits the flat `senses_raw` column into three tables:

**`entries`** — one row per unique (traditional, simplified, pinyin) triple. UNIQUE constraint prevents duplicates. Indexed on `simplified` and `traditional` for fast headword lookup.

**`senses`** — one row per logical sense within an entry. Has `is_classifier` (0/1) to flag `CL:` senses without runtime prefix detection. Has nullable `embedding` BLOB for WSD sense embeddings (only populated for polysemous words).

**`glosses`** — one row per individual English translation within a sense. Stores pre-parsed text — no slashes or semicolons in `gloss_text`. Inline references preserved verbatim.

Why this structure:
- **No runtime parsing.** The app reads structured rows via JOIN queries and populates `Entry` → `Sense` → `Gloss` objects directly.
- **Per-sense embeddings.** The `senses.embedding` BLOB column enables WSD without a separate lookup table.
- **Ordering by rowid.** No explicit `sense_order` or `gloss_order` columns — insertion order via `rowid` preserves CEDICT's original ordering. WSD scoring reorders senses at runtime anyway.
- **Classifier detection at export time.** `is_classifier` is set during the CEDICT parse, so the app never needs to check for `CL:` prefixes.

## WSD Sense Embedding Strategy (Database Export)

### Approach

Polysemous words (simplified forms with >1 sense across all entries) get precomputed WSD embeddings stored in the `senses.embedding` column. This enables fast cosine similarity at runtime without re-encoding sense descriptions on-device.

### How It Works

1. After all CEDICT entries are inserted into the normalized schema, the script queries for simplified forms with more than one total sense.
2. The WSD model (`SentenceTransformer` from `ml/models/wsd_finetuned/checkpoint-32500`) is loaded.
3. For each sense of a polysemous word, the script looks up the Chinese translation from `ml/data/translation_cache.json` using the key format `word|pinyin|english` (e.g., `打|da3|to hit`).
4. The Chinese translation (not the English gloss) is encoded with the WSD model using `normalize_embeddings=True` to produce a unit-length 768-dim vector.
5. The embedding is stored as raw float32 little-endian bytes (3072 bytes per sense) in the `senses.embedding` BLOB column.

### Why Chinese Translations, Not English Glosses

The WSD model (`thenlper/gte-base-zh`) is a Chinese-language bi-encoder. It was fine-tuned on Chinese text pairs. Encoding the Chinese translation of each sense produces embeddings in the same semantic space as the runtime context encoding (which is also Chinese text). English glosses would be out-of-distribution for this model.

### Verification Against WSD Training Data

The WSD model was fine-tuned on Chinese text pairs where the training data used the same `translation_cache.json` to generate sense descriptions. By encoding the Chinese translation from the cache (rather than the English gloss), the stored sense embeddings live in the same semantic space as the runtime context embeddings. This means cosine similarity at runtime is comparing apples to apples — both vectors come from Chinese text encoded by the same model.

The `export_coreml.py` verification step confirms CoreML outputs match PyTorch outputs within tolerance (< 1e-4 per element). For WSD specifically, the tolerance is ~1e-3 due to MIL float optimizations, but the ranking order is preserved — the relative ordering of cosine similarities is identical between PyTorch and CoreML.

### Translation Cache Key Format and Coverage Gaps

The cache key format is `simplified|pinyin|english_gloss` (e.g., `打|da3|to hit`).

For multi-gloss senses, the export script tries two lookup strategies:
1. Full sense text: glosses joined by `"; "` (e.g., `打|da3|to hit; to strike`)
2. Fallback: first gloss only (e.g., `打|da3|to hit`)

Senses with no matching cache entry get a NULL embedding and a logged warning. At runtime, NULL-embedding senses receive a default low score (`-.infinity`) and sort after all scored senses within their entry.

Coverage gaps are expected for:
- Rare or archaic senses that weren't in the WSD training data
- Classifier senses (`CL:` entries) — these are structural, not semantic
- Senses where the English gloss was reformatted between CEDICT versions and the cache

The export script logs every cache miss with the attempted key, so coverage can be audited from the build output.


## Swift Integration Decisions

### Shared BERT Tokenizer

A single `BertTokenizer` class serves both CWS and WSD models. The tokenizer accepts different `vocab.txt` files at init:
- CWS uses `cws_vocab.txt` (from `ckiplab/bert-base-chinese-ws`)
- WSD uses `wsd_vocab.txt` (from `thenlper/gte-base-zh`)

Both vocabs use the same BERT WordPiece format. For Chinese text, each CJK character maps to a single token (no subword splitting). Non-CJK text (Latin, punctuation) goes through standard WordPiece with `##` continuation prefixes.

The tokenizer produces `offset_mapping` — critical for CWS, where per-token logits must be mapped back to per-character B-probabilities. For CJK-heavy text this is 1:1, but mixed-script text (e.g., `iPhone手机`) needs the offset mapping to handle WordPiece splits correctly.

### Caching Strategy

Both `CoreMLSegmentationService` and `WSDService` use simple FIFO caches:

- **CWS sentence cache**: keyed by sentence text → `[String]` segmentation result. FIFO eviction at `ReaderConstants.Segmentation.maxSentenceCacheSize`. This avoids re-running CoreML inference when the user scrolls back to previously segmented text.
- **WSD cache**: keyed by `"word\0sentence"` (null-separated to avoid collisions) → sorted `[Entry]` result. FIFO eviction at `ReaderConstants.WSD.maxCacheSize`. Avoids re-encoding context when the same word is looked up again in the same sentence.

FIFO was chosen over LRU for simplicity — the access pattern is mostly sequential (reading through text), so recently-inserted entries are the most likely to be re-accessed. The overhead of maintaining an LRU linked list isn't justified for the expected cache sizes.

### CoreML Model Loading and Fallback

`CoreMLSegmentationService` loads the CWS `.mlpackage` from the app bundle at init. If loading fails, the app falls back to `CedictSegmentationService` (dictionary-only segmentation). Similarly, if `WSDService` fails to load, `DictionaryService` returns results in default insertion order — no crash, just degraded quality.

### Non-Lookupable Segment Marking

After segmentation, `CoreMLSegmentationService.isLookupable()` checks whether a segment contains at least one CJK character. Segments of pure punctuation, whitespace, or Latin text are marked non-lookupable. The interaction layer (`LongPressGestureHandler`) skips these — no highlight, no dictionary popup. This prevents confusing UX when the user taps on a comma or space.
