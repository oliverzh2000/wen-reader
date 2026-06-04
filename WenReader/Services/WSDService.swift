// Copyright 2025 Oliver Zhang
// Licensed under the MIT License

import Foundation

/// Word Sense Disambiguation service using a CoreML bi-encoder model.
///
/// Encodes sentence context with `★` markers around the target word,
/// runs CoreML inference to get a 512-dim normalized embedding, then
/// computes cosine similarity (dot product, since embeddings are pre-normalized)
/// against stored sense embeddings to rank senses by contextual relevance.
final class WSDService {

    private let runner: BertCoreMLRunner

    /// Dimensionality of the WSD embedding vectors.
    static let embeddingDim = 512
    /// Expected byte count for a stored embedding BLOB: 4 bytes scale (float32) + 512 bytes (int8) = 516.
    static let embeddingByteCount = 4 + embeddingDim

    /// Fixed sequence length for the CoreML model.
    private static let sequenceLength = 16
    /// Maximum token length (fixed shape).
    private static let modelMaxTokens = 16
    /// Maximum characters of marked context that fit (accounts for [CLS] + [SEP]).
    /// The ★ markers are part of the text and counted within this budget.
    private static let maxInputChars = modelMaxTokens - BertCoreMLRunner.specialTokenCount  // 14

    // MARK: - WSD Cache (FIFO)

    /// Cache mapping (word, sentence) pairs → sorted [Entry] results.
    /// Key is "\(word)\0\(sentenceContext)" using null separator to avoid collisions.
    private var cache: [String: [Entry]] = [:]
    /// Insertion-order keys for FIFO eviction.
    private var cacheOrder: [String] = []
    /// Maximum number of cached results before FIFO eviction kicks in.
    let maxCacheSize: Int

    /// Initialize with a CoreML WSD model from the app bundle.
    ///
    /// - Parameters:
    ///   - modelName: The CoreML model resource name (without extension). Defaults to `"wsd_encoder"`.
    ///   - vocabResourceName: The vocab file resource name for the WSD tokenizer. Defaults to `"wsd_vocab"`.
    ///   - maxCacheSize: Maximum cached WSD results before FIFO eviction. Defaults to `ReaderConstants.WSD.maxCacheSize`.
    /// - Throws: If the model or vocab file cannot be loaded.
    init(
        modelName: String = "wsd_encoder",
        vocabResourceName: String = "wsd_vocab",
        maxCacheSize: Int = ReaderConstants.WSD.maxCacheSize
    ) throws {
        self.runner = try BertCoreMLRunner(
            modelName: modelName,
            vocabName: vocabResourceName,
            sequenceLength: Self.sequenceLength
        )
        self.maxCacheSize = maxCacheSize
    }

    /// Internal initializer for testing with pre-built components.
    init(runner: BertCoreMLRunner, maxCacheSize: Int = ReaderConstants.WSD.maxCacheSize) {
        self.runner = runner
        self.maxCacheSize = maxCacheSize
    }

    // MARK: - Public API

    /// Rank dictionary entries by contextual relevance using WSD.
    ///
    /// 1. Wraps the target word with `★` markers in the sentence context
    /// 2. Tokenizes and runs CoreML inference to get a context embedding
    /// 3. Computes cosine similarity (dot product) between context embedding
    ///    and each cluster embedding
    /// 4. Returns entries with senses reordered: winning cluster's senses first,
    ///    trivial senses last
    ///
    /// - Parameters:
    ///   - word: The target word being looked up.
    ///   - sentenceContext: The full sentence containing the word.
    ///   - entries: Dictionary entries for the word.
    ///   - clusterEmbeddings: Cluster IDs mapped to their embedding data and trivial flag.
    /// - Returns: Entries with senses reordered by contextual relevance.
    func rankSenses(
        word: String,
        sentenceContext: String,
        wordOffsetInSentence: Int = 0,
        entries: [Entry],
        clusterEmbeddings: [(clusterId: Int, isTrivial: Bool, embedding: Data?)],
        senseClusterIds: [Int]
    ) async -> [Entry] {
        // Check cache first — include offset to distinguish repeated words in the same sentence
        let cacheKey = "\(word)\0\(sentenceContext)\0\(wordOffsetInSentence)"
        if let cached = cache[cacheKey] {
            print("[WSD DEBUG] Cache hit for '\(word)' at offset \(wordOffsetInSentence) in '\(sentenceContext.prefix(30))...'")
            return cached
        }

        print("[WSD DEBUG] rankSenses: word='\(word)', offset=\(wordOffsetInSentence), sentence='\(sentenceContext)' (\(sentenceContext.count) chars)")
        print("[WSD DEBUG] rankSenses: \(entries.count) entries, \(clusterEmbeddings.count) clusters, \(senseClusterIds.count) senseClusterIds")

        // Encode the sentence context with ★ markers around the target word at the correct offset
        let markedContext = wrapWordWithMarkers(word: word, in: sentenceContext, at: wordOffsetInSentence)
        print("[WSD DEBUG] Word: \(word)")
        print("[WSD DEBUG] Context: \(markedContext)")

        // Tokenize and run CoreML inference
        let wsdStart = CFAbsoluteTimeGetCurrent()
        guard let contextEmbedding = encodeContext(markedContext) else {
            Log.error("WSDService: failed to encode context for word '\(word)'")
            print("[WSD DEBUG] FAILED to encode context, returning original entries")
            return entries
        }
        let wsdInferenceMs = (CFAbsoluteTimeGetCurrent() - wsdStart) * 1000
        print("[WSD DEBUG] Inference: \(String(format: "%.1f", wsdInferenceMs))ms")

        // Score each cluster by cosine similarity
        let clusterScores = scoreClusterEmbeddings(
            contextEmbedding: contextEmbedding,
            clusterEmbeddings: clusterEmbeddings
        )

        // Reorder entries: within each entry, sort senses by cluster score
        let result = reorderEntries(
            entries: entries,
            clusterScores: clusterScores,
            senseClusterIds: senseClusterIds
        )

        storeInCache(key: cacheKey, result: result)
        return result
    }

    // MARK: - Cache Helpers

    /// Build a cache key from a (word, sentence, offset) tuple.
    /// Uses null separator to avoid collisions between different word/sentence combinations.
    static func cacheKey(word: String, sentence: String, offset: Int = 0) -> String {
        "\(word)\0\(sentence)\0\(offset)"
    }

    /// Store a WSD result in the cache with FIFO eviction.
    private func storeInCache(key: String, result: [Entry]) {
        // If already cached, don't duplicate
        if cache[key] != nil {
            return
        }

        // Evict oldest entry if at capacity
        if cacheOrder.count >= maxCacheSize {
            let oldest = cacheOrder.removeFirst()
            cache.removeValue(forKey: oldest)
        }

        cache[key] = result
        cacheOrder.append(key)
    }

    // MARK: - Context Encoding

    /// Wrap the target word with `★` markers in the sentence context at the specified character offset.
    ///
    /// For example, if word is "打" and context is "他打了球，然后打了人",
    /// with offset=1, wraps the first "打": "他★打★了球，然后打了人".
    /// With offset=7, wraps the second: "他打了球，然后★打★了人".
    ///
    /// Falls back to first occurrence if the offset doesn't match.
    func wrapWordWithMarkers(word: String, in sentence: String, at offset: Int = 0) -> String {
        let chars = Array(sentence)
        let wordChars = Array(word)
        
        // Try to match at the specified offset first
        if offset >= 0 && offset + wordChars.count <= chars.count {
            let slice = chars[offset..<(offset + wordChars.count)]
            if Array(slice) == wordChars {
                var result = String(chars[0..<offset])
                result += "★\(word)★"
                result += String(chars[(offset + wordChars.count)...])
                return result
            }
        }
        
        // Fallback: find the first occurrence using range(of:)
        guard let range = sentence.range(of: word) else {
            print("[WSD DEBUG] wrapWordWithMarkers: '\(word)' NOT FOUND in '\(sentence)'")
            return sentence
        }
        var result = sentence
        result.replaceSubrange(range, with: "★\(word)★")
        return result
    }

    /// Tokenize the marker-wrapped context and run CoreML inference
    /// to produce a normalized 512-dim embedding.
    ///
    /// Truncates the context to fit within the model's maximum token budget (14 tokens
    /// for content = 16 - [CLS] - [SEP]), centered around the ★ markers.
    ///
    /// - Parameter text: The marker-wrapped sentence context.
    /// - Returns: The normalized embedding as a Float array, or nil on failure.
    func encodeContext(_ text: String) -> [Float]? {
        let truncated = truncateForWSD(text, maxChars: Self.maxInputChars)
        print("[WSD DEBUG] encodeContext: original='\(text)' (\(text.count) chars), truncated='\(truncated)' (\(truncated.count) chars), maxInputChars=\(Self.maxInputChars)")
        let encoded = runner.tokenizer.encode(truncated)
        print("[WSD DEBUG] encodeContext: tokenized to \(encoded.inputIds.count) tokens, ids=\(encoded.inputIds)")
        guard let result = runner.predictFloats(
            inputIds: encoded.inputIds,
            outputKey: "embedding"
        ) else {
            print("[WSD DEBUG] encodeContext: predictFloats returned nil!")
            return nil
        }
        print("[WSD DEBUG] encodeContext: got \(result.count) floats, first 5: \(Array(result.prefix(5)))")
        // Verify embedding dimension
        guard result.count == Self.embeddingDim else {
            print("[WSD DEBUG] encodeContext: WRONG DIM! expected \(Self.embeddingDim), got \(result.count)")
            return nil
        }
        return result
    }

    /// Truncate WSD context to maxChars, keeping ★target★ centered.
    private func truncateForWSD(_ text: String, maxChars: Int) -> String {
        let chars = Array(text)
        guard chars.count > maxChars else { return text }

        // Find the ★ markers to center around the target word
        if let starIdx = chars.firstIndex(of: "★") {
            let center = chars.distance(from: chars.startIndex, to: starIdx)
            let half = maxChars / 2
            var start = max(0, center - half)
            var end = min(chars.count, start + maxChars)
            if end == chars.count {
                start = max(0, end - maxChars)
            }
            return String(chars[start..<end])
        }

        // No marker found — take from start
        return String(chars.prefix(maxChars))
    }

    // MARK: - Cluster Scoring

    /// Compute dot product between two float arrays (cosine similarity for normalized vectors).
    func dotProduct(_ a: [Float], _ b: [Float]) -> Float {
        guard a.count == b.count else { return 0 }
        var sum: Float = 0
        for i in 0..<a.count {
            sum += a[i] * b[i]
        }
        return sum
    }

    /// Default low score for senses/clusters with NULL embeddings.
    static let nullEmbeddingScore: Float = -.infinity

    /// Score each cluster embedding against the context embedding.
    /// Dequantizes int8 embeddings (scale + int8 values) and computes dot product.
    /// Returns a dictionary mapping clusterId → cosine similarity score.
    private func scoreClusterEmbeddings(
        contextEmbedding: [Float],
        clusterEmbeddings: [(clusterId: Int, isTrivial: Bool, embedding: Data?)]
    ) -> [Int: Float] {
        var scores: [Int: Float] = [:]
        print("[WSD DEBUG] scoreClusterEmbeddings: \(clusterEmbeddings.count) clusters, contextEmb dim=\(contextEmbedding.count)")
        for (clusterId, isTrivial, data) in clusterEmbeddings {
            if isTrivial {
                scores[clusterId] = Self.nullEmbeddingScore
                print("[WSD DEBUG]   cluster \(clusterId): trivial → -inf")
                continue
            }
            guard let data = data, data.count == Self.embeddingByteCount else {
                scores[clusterId] = Self.nullEmbeddingScore
                let dataDesc = data.map { "\($0.count) bytes" } ?? "nil"
                print("[WSD DEBUG]   cluster \(clusterId): bad embedding data (\(dataDesc), expected \(Self.embeddingByteCount)) → -inf")
                continue
            }
            // Dequantize: first 4 bytes = scale (float32), remaining 512 bytes = int8 values
            let floats: [Float] = data.withUnsafeBytes { ptr in
                let scale = ptr.load(fromByteOffset: 0, as: Float.self)
                let int8Ptr = ptr.baseAddress!.advanced(by: 4).assumingMemoryBound(to: Int8.self)
                let dequantFactor = scale / 127.0
                var result = [Float](repeating: 0, count: Self.embeddingDim)
                for i in 0..<Self.embeddingDim {
                    result[i] = Float(int8Ptr[i]) * dequantFactor
                }
                return result
            }
            let sim = dotProduct(contextEmbedding, floats)
            scores[clusterId] = sim
            print("[WSD DEBUG]   cluster \(clusterId): cos=\(String(format: "%.4f", sim))")
        }
        return scores
    }

    /// Reorder entries based on cluster scores.
    ///
    /// Each sense has a corresponding cluster ID (via senseClusterIds, indexed
    /// by flat sense position across all entries). Senses inherit their cluster's
    /// score and are sorted descending within each entry. Entries are sorted by
    /// their best cluster score.
    private func reorderEntries(
        entries: [Entry],
        clusterScores: [Int: Float],
        senseClusterIds: [Int]
    ) -> [Entry] {
        var senseIdx = 0
        var scoredEntryPairs: [(entry: Entry, bestScore: Float)] = []

        for entry in entries {
            var sensesWithScores: [(sense: Sense, score: Float)] = []

            let pinyin = entry.accentedPinyin.joined(separator: " ")
            print("[WSD DEBUG] Entry: \(entry.simplified) [\(pinyin)]")

            for sense in entry.senses {
                let score: Float
                if senseIdx < senseClusterIds.count {
                    let clusterId = senseClusterIds[senseIdx]
                    score = clusterScores[clusterId] ?? Self.nullEmbeddingScore
                } else {
                    score = Self.nullEmbeddingScore
                }
                sensesWithScores.append((sense, score))

                // Debug: print each sense with its cluster score
                let glossText = sense.glosses.first?.fragments.map { f in
                    switch f {
                    case .text(let t): return t
                    case .accentedPinyin(let p): return "[\(p.joined(separator: " "))]"
                    case .link(let l): return l.simplified
                    }
                }.joined() ?? "?"
                let scoreStr = score == Self.nullEmbeddingScore ? "NULL" : String(format: "%.4f", score)
                let clusterStr = senseIdx < senseClusterIds.count ? "c\(senseClusterIds[senseIdx])" : "?"
                print("[WSD DEBUG]   cos=\(scoreStr) [\(clusterStr)] → \(glossText)")

                senseIdx += 1
            }

            sensesWithScores.sort { $0.score > $1.score }

            // Mark senses below the top cluster as secondary
            let topScore = sensesWithScores.first?.score ?? Self.nullEmbeddingScore
            let markedSenses: [Sense] = sensesWithScores.map { pair in
                var sense = pair.sense
                sense.isPrimary = pair.score >= topScore
                return sense
            }

            let bestScore = topScore
            let sortedEntry = Entry(
                traditional: entry.traditional,
                simplified: entry.simplified,
                accentedPinyin: entry.accentedPinyin,
                senses: markedSenses
            )
            scoredEntryPairs.append((sortedEntry, bestScore))
        }

        scoredEntryPairs.sort { $0.bestScore > $1.bestScore }
        return scoredEntryPairs.map(\.entry)
    }
}
