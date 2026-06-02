// Copyright 2025 Oliver Zhang
// Licensed under the MIT License

import Accelerate
import Foundation

/// ML-based Chinese word segmentation using a span scoring architecture.
///
/// Cache strategy: encoder + MLP scores are cached per sentence. The cache
/// stores (position, word) → score. DP is the only thing that re-runs —
/// it's sub-microsecond and the only thing that changes during sub-splitting
/// (different mask, same scores).
final class SpanScorerSegmentationService: SegmentationService {

    private let runner: BertCoreMLRunner
    private let trie: CedictTrie

    private let widthEmbedding: [[Float]]   // [19, 64]
    private let mlpWeight1: [[Float]]       // [256, 576]
    private let mlpBias1: [Float]           // [256]
    private let mlpWeight2: [[Float]]       // [1, 256]
    private let mlpBias2: [Float]           // [1]

    static let maxWordLen = 19
    static let hiddenDim = 256
    static let widthEmbedDim = 64
    static let mlpHidden = 256

    /// Fixed sequence length for the CoreML model.
    private static let sequenceLength = 32
    /// Maximum characters of text that fit in the model input.
    static let maxInputChars = 32 - BertCoreMLRunner.specialTokenCount  // 30

    // MARK: - Score Cache

    /// All precomputed span scores for a sentence.
    private struct SentenceScores {
        let chars: [Character]
        /// scores[position] = [(word, mlpScore), ...] for all candidates at that position
        var scores: [Int: [(word: String, score: Float)]]
    }

    /// Single-slot cache: last sentence's precomputed scores.
    private var scoreCache: (sentence: String, data: SentenceScores)?

    // MARK: - Init

    init(
        trie: CedictTrie,
        modelName: String = "span_scorer_encoder",
        vocabResourceName: String = "cws_vocab",
        headWeightsResourceName: String = "span_head_weights"
    ) throws {
        self.runner = try BertCoreMLRunner(
            modelName: modelName,
            vocabName: vocabResourceName,
            sequenceLength: Self.sequenceLength
        )
        self.trie = trie

        guard let headURL = Bundle.main.url(forResource: headWeightsResourceName, withExtension: "bin") else {
            throw SpanScorerError.headWeightsNotFound(headWeightsResourceName)
        }
        let headData = try Data(contentsOf: headURL)
        let allFloats: [Float] = headData.withUnsafeBytes { ptr in
            Array(ptr.bindMemory(to: Float.self))
        }
        print("[SPAN DEBUG] Head weights: \(headData.count) bytes = \(allFloats.count) floats")

        var offset = 0
        let wDim = Self.widthEmbedDim
        let hDim = Self.hiddenDim
        let mlpH = Self.mlpHidden
        let inputDim = hDim * 2 + wDim
        
        let expectedFloats = Self.maxWordLen * wDim + mlpH * inputDim + mlpH + mlpH + 1
        print("[SPAN DEBUG] Expected \(expectedFloats) floats: widthEmb=\(Self.maxWordLen * wDim), w1=\(mlpH * inputDim), b1=\(mlpH), w2=\(mlpH), b2=1")
        guard allFloats.count >= expectedFloats else {
            print("[SPAN DEBUG] FATAL: head weights file too small! Has \(allFloats.count), need \(expectedFloats)")
            throw SpanScorerError.headWeightsNotFound(headWeightsResourceName)
        }

        var widthEmb = [[Float]]()
        for row in 0..<Self.maxWordLen {
            let s = offset + row * wDim
            widthEmb.append(Array(allFloats[s..<(s + wDim)]))
        }
        self.widthEmbedding = widthEmb
        offset += Self.maxWordLen * wDim

        var w1 = [[Float]]()
        for row in 0..<mlpH {
            let s = offset + row * inputDim
            w1.append(Array(allFloats[s..<(s + inputDim)]))
        }
        self.mlpWeight1 = w1
        offset += mlpH * inputDim

        self.mlpBias1 = Array(allFloats[offset..<(offset + mlpH)])
        offset += mlpH
        self.mlpWeight2 = [Array(allFloats[offset..<(offset + mlpH)])]
        offset += mlpH
        self.mlpBias2 = [allFloats[offset]]
        offset += 1

        // Pre-flatten weight matrices for Accelerate (cblas_sgemv)
        self.mlpWeight1Flat = w1.flatMap { $0 }
        self.mlpWeight2Flat = self.mlpWeight2[0]

        Log.info("SpanScorer: loaded head weights (\(offset) floats, \(headData.count) bytes)")
    }

    // MARK: - Score computation (encoder + MLP, cached)

    /// Get precomputed scores for a sentence, running encoder + MLP if not cached.
    private func getScores(for sentence: String) -> SentenceScores? {
        if let cached = scoreCache, cached.sentence == sentence {
            return cached.data
        }

        let chars = Array(sentence)
        print("[SPAN DEBUG] getScores: sentence='\(sentence)' (\(chars.count) chars), maxInputChars=\(Self.maxInputChars)")

        // Enumerated model shapes cap at modelMaxTokens. For CJK, 1 char = 1 token.
        guard chars.count <= Self.maxInputChars else {
            Log.error("SpanScorer: sentence too long (\(chars.count) chars), max \(Self.maxInputChars)")
            return nil
        }

        let encoderStart = CFAbsoluteTimeGetCurrent()
        guard let hiddenStates = runEncoder(text: sentence) else {
            print("[SPAN DEBUG] Encoder returned nil for: \(sentence)")
            return nil
        }
        // Derive seqLen from actual encoder output, not chars.count + 2.
        // The 1-char-per-token assumption breaks for whitespace (dropped by tokenizer)
        // or non-CJK text (WordPiece splitting). Using the real size prevents OOB crashes.
        let seqLen = hiddenStates.count / Self.hiddenDim
        let encoderMs = (CFAbsoluteTimeGetCurrent() - encoderStart) * 1000
        print("[SPAN DEBUG] getScores: hiddenStates has \(hiddenStates.count) floats → seqLen=\(seqLen) (chars.count+2 would be \(chars.count + 2))")

        // Score every candidate span via MLP
        let mlpStart = CFAbsoluteTimeGetCurrent()
        var scores: [Int: [(word: String, score: Float)]] = [:]
        var totalSpansScored = 0
        var skippedOOB = 0

        for i in 0..<chars.count {
            var candidates = trie.getWordsAt(chars, position: i)
            let singleChar = String(chars[i])
            if !candidates.contains(where: { $0.count == 1 }) {
                candidates.append(singleChar)
            }

            var posScores: [(word: String, score: Float)] = []
            for word in candidates {
                let wLen = word.count
                guard wLen <= Self.maxWordLen, i + wLen <= chars.count else { continue }
                let score = scoreSpan(
                    hiddenStates: hiddenStates,
                    start: i, endInclusive: i + wLen - 1,
                    width: wLen, seqLen: seqLen
                )
                if score == -100.0 {
                    skippedOOB += 1
                } else {
                    totalSpansScored += 1
                }
                posScores.append((word: word, score: score))
            }
            scores[i] = posScores
        }
        let mlpMs = (CFAbsoluteTimeGetCurrent() - mlpStart) * 1000

        print("[SPAN DEBUG] Encoder: \(String(format: "%.1f", encoderMs))ms, MLP: \(String(format: "%.2f", mlpMs))ms (\(chars.count) chars, \(totalSpansScored) spans scored, \(skippedOOB) skipped OOB)")

        let result = SentenceScores(chars: chars, scores: scores)
        scoreCache = (sentence: sentence, data: result)
        return result
    }

    // MARK: - SegmentationService

    func segment(run: String, sentence: String) async -> [String] {
        guard !run.isEmpty else { return [] }

        print("[SPAN DEBUG] segment called: run='\(run)' (\(run.count) chars), sentence='\(sentence)' (\(sentence.count) chars)")

        // Truncate context to fit fixed model input.
        // Keep the run fully included, centered in the window.
        let context = truncateContext(sentence: sentence, run: run, maxChars: Self.maxInputChars)
        print("[SPAN DEBUG] truncated context: '\(context)' (\(context.count) chars)")

        guard let sentenceScores = getScores(for: context) else {
            print("[SPAN DEBUG] getScores returned nil, falling back to char-by-char")
            return Array(run).map { String($0) }
        }

        let runChars = Array(run)
        let runStart = findRunOffset(runChars: runChars, in: sentenceScores.chars)
        print("[SPAN DEBUG] runStart=\(runStart) in context of \(sentenceScores.chars.count) chars")

        // Debug: print per-span scores for the run
        for i in 0..<runChars.count {
            let absPos = runStart + i
            if let candidates = sentenceScores.scores[absPos] {
                for (word, score) in candidates {
                    print("[SPAN DEBUG]   pos \(i): '\(word)' (len=\(word.count)) score=\(String(format: "%.4f", score))")
                }
            } else {
                print("[SPAN DEBUG]   pos \(i): NO CANDIDATES (char='\(runChars[i])')")
            }
        }

        let result = dpDecode(
            sentenceScores: sentenceScores,
            rangeStart: runStart,
            rangeLen: runChars.count,
            maskedWords: Set()
        )

        print("[SPAN DEBUG] Input: \(run) (in: \(sentence))")
        print("[SPAN DEBUG] Result: \(result.joined(separator: " | "))")
        return result
    }

    // MARK: - Sub-splitting

    /// Re-run DP on a word's range with that word (and longer) masked out.
    /// Reuses cached scores — no encoder or MLP recomputation.
    func resegment(
        originalRun: String,
        sentence: String,
        wordToSplit: String,
        wordStartInRun: Int
    ) -> [String]? {
        guard wordToSplit.count > 1 else { return [wordToSplit] }

        let context = truncateContext(sentence: sentence, run: originalRun, maxChars: Self.maxInputChars)

        guard let sentenceScores = getScores(for: context) else {
            print("[SPAN DEBUG] Resegment: no scores available")
            return nil
        }

        let runChars = Array(originalRun)
        let runStart = findRunOffset(runChars: runChars, in: sentenceScores.chars)
        let absWordStart = runStart + wordStartInRun
        let n = wordToSplit.count

        // Mask the word and anything longer at the same position
        var maskedWords = Set<String>()
        maskedWords.insert(wordToSplit)
        if let candidates = sentenceScores.scores[absWordStart] {
            for (w, _) in candidates where w.count >= n {
                maskedWords.insert(w)
            }
        }

        print("[SPAN DEBUG] Resegment: '\(wordToSplit)' at \(absWordStart), masked: \(maskedWords.sorted())")

        let result = dpDecode(
            sentenceScores: sentenceScores,
            rangeStart: absWordStart,
            rangeLen: n,
            maskedWords: maskedWords
        )

        print("[SPAN DEBUG] Resegment: '\(wordToSplit)' → \(result.joined(separator: " | "))")
        return result
    }

    // MARK: - DP Decode

    /// Pure DP over precomputed scores. Sub-microsecond.
    private func dpDecode(
        sentenceScores: SentenceScores,
        rangeStart: Int,
        rangeLen: Int,
        maskedWords: Set<String>
    ) -> [String] {
        let n = rangeLen
        var bestScore = [Float](repeating: -.infinity, count: n + 1)
        var backtrack = [Int](repeating: 0, count: n + 1)
        bestScore[0] = 0

        for i in 0..<n {
            guard bestScore[i] > -.infinity else { continue }
            let absPos = rangeStart + i

            guard let candidates = sentenceScores.scores[absPos] else {
                print("[SPAN DEBUG] dpDecode: no candidates at absPos=\(absPos) (relPos=\(i))")
                continue
            }
            for (word, score) in candidates {
                let wLen = word.count
                let relEnd = i + wLen
                guard relEnd <= n else { continue }
                if maskedWords.contains(word) { continue }

                let total = bestScore[i] + score
                if total > bestScore[relEnd] {
                    bestScore[relEnd] = total
                    backtrack[relEnd] = i
                }
            }
        }

        // Check if DP reached the end
        if bestScore[n] == -.infinity {
            print("[SPAN DEBUG] dpDecode: UNREACHABLE end! rangeStart=\(rangeStart) rangeLen=\(n)")
            print("[SPAN DEBUG] dpDecode: bestScore=\(bestScore)")
            // Fallback: return individual characters
            let chars = sentenceScores.chars
            return (0..<n).map { String(chars[rangeStart + $0]) }
        }

        let chars = sentenceScores.chars
        var segments: [String] = []
        var pos = n
        while pos > 0 {
            let start = backtrack[pos]
            let segStart = rangeStart + start
            let segEnd = rangeStart + pos
            guard segStart >= 0, segEnd <= chars.count else {
                print("[SPAN DEBUG] dpDecode: OOB in backtrack! segStart=\(segStart) segEnd=\(segEnd) chars.count=\(chars.count)")
                break
            }
            segments.append(String(chars[segStart..<segEnd]))
            pos = start
        }
        segments.reverse()
        return segments
    }

    // MARK: - Helpers

    private func findRunOffset(runChars: [Character], in sentenceChars: [Character]) -> Int {
        let n = sentenceChars.count
        let m = runChars.count
        guard m <= n else { return 0 }
        for i in 0...(n - m) {
            if Array(sentenceChars[i..<(i + m)]) == runChars {
                return i
            }
        }
        return 0
    }

    /// Truncate context to maxChars, keeping the run fully included and centered.
    private func truncateContext(sentence: String, run: String, maxChars: Int) -> String {
        let sentenceChars = Array(sentence)
        guard sentenceChars.count > maxChars else { return sentence }

        let runChars = Array(run)
        let runStart = findRunOffset(runChars: runChars, in: sentenceChars)
        let runEnd = runStart + runChars.count

        // If the run itself exceeds maxChars, just return the run (will be truncated by encoder)
        guard runChars.count <= maxChars else {
            return run
        }

        // Center the window around the run
        let padding = maxChars - runChars.count
        let leftPad = padding / 2
        let rightPad = padding - leftPad

        var windowStart = max(0, runStart - leftPad)
        var windowEnd = min(sentenceChars.count, runEnd + rightPad)

        // Adjust if we hit a boundary
        if windowStart == 0 {
            windowEnd = min(sentenceChars.count, maxChars)
        } else if windowEnd == sentenceChars.count {
            windowStart = max(0, sentenceChars.count - maxChars)
        }

        return String(sentenceChars[windowStart..<windowEnd])
    }

    // MARK: - CoreML Encoder

    private func runEncoder(text: String) -> [Float]? {
        let encoded = runner.tokenizer.encode(text)
        let seqLen = encoded.inputIds.count
        print("[SPAN DEBUG] runEncoder: text='\(text)' (\(Array(text).count) chars), tokenized to \(seqLen) tokens")
        print("[SPAN DEBUG] runEncoder: inputIds=\(encoded.inputIds)")

        // Only read seqLen * hiddenDim floats (ignore padding positions)
        let expectedFloats = seqLen * Self.hiddenDim
        print("[SPAN DEBUG] runEncoder: requesting \(expectedFloats) floats (\(seqLen) tokens × \(Self.hiddenDim) dim)")
        return runner.predictFloats(
            inputIds: encoded.inputIds,
            outputKey: "hidden_states",
            maxFloats: expectedFloats
        )
    }

    // MARK: - MLP Scoring (Accelerate)

    /// Contiguous weight storage for vDSP matrix-vector operations.
    private let mlpWeight1Flat: [Float]   // [mlpHidden × inputDim] row-major
    private let mlpWeight2Flat: [Float]   // [1 × mlpHidden] row-major

    private func scoreSpan(
        hiddenStates: [Float], start: Int, endInclusive: Int, width: Int, seqLen: Int
    ) -> Float {
        let hDim = Self.hiddenDim
        let wDim = Self.widthEmbedDim
        let mlpH = Self.mlpHidden
        let inputDim = hDim * 2 + wDim
        let startToken = start + 1
        let endToken = endInclusive + 1

        guard startToken < seqLen, endToken < seqLen, width >= 1, width <= Self.maxWordLen else {
            print("[SPAN DEBUG] scoreSpan SKIP (guard): start=\(start) end=\(endInclusive) width=\(width) seqLen=\(seqLen) startToken=\(startToken) endToken=\(endToken)")
            return -100.0
        }

        let startOff = startToken * hDim
        let endOff = endToken * hDim

        guard endOff + hDim <= hiddenStates.count else {
            print("[SPAN DEBUG] scoreSpan OOB: endOff=\(endOff) + hDim=\(hDim) = \(endOff + hDim) > hiddenStates.count=\(hiddenStates.count)")
            return -100.0
        }

        // Build input vector: [H[start] ; H[end] ; width_emb]
        var input = [Float](repeating: 0, count: inputDim)
        input.withUnsafeMutableBufferPointer { buf in
            hiddenStates.withUnsafeBufferPointer { src in
                buf.baseAddress!.update(from: src.baseAddress! + startOff, count: hDim)
                (buf.baseAddress! + hDim).update(from: src.baseAddress! + endOff, count: hDim)
            }
            widthEmbedding[width - 1].withUnsafeBufferPointer { wEmb in
                (buf.baseAddress! + hDim * 2).update(from: wEmb.baseAddress!, count: wDim)
            }
        }

        // Layer 1: hidden = ReLU(W1 @ input + b1)
        // cblas_sgemv: y = alpha * A * x + beta * y
        var hidden = [Float](mlpBias1)  // start with bias
        mlpWeight1Flat.withUnsafeBufferPointer { w1 in
            input.withUnsafeBufferPointer { inp in
                hidden.withUnsafeMutableBufferPointer { h in
                    cblas_sgemv(
                        CblasRowMajor, CblasNoTrans,
                        Int32(mlpH), Int32(inputDim),
                        1.0,                          // alpha
                        w1.baseAddress!, Int32(inputDim),  // A, lda
                        inp.baseAddress!, 1,          // x, incx
                        1.0,                          // beta (add to bias)
                        h.baseAddress!, 1             // y, incy
                    )
                }
            }
        }

        // ReLU in-place
        vDSP.threshold(hidden, to: 0, with: .zeroFill, result: &hidden)

        // Layer 2: score = W2 @ hidden + b2
        var score: Float = 0
        mlpWeight2Flat.withUnsafeBufferPointer { w2 in
            hidden.withUnsafeBufferPointer { h in
                vDSP_dotpr(w2.baseAddress!, 1, h.baseAddress!, 1, &score, vDSP_Length(mlpH))
            }
        }
        score += mlpBias2[0]

        return score
    }

    static func isLookupable(_ segment: String) -> Bool {
        for scalar in segment.unicodeScalars {
            if BertTokenizer.isCJKCharacter(scalar) { return true }
        }
        return false
    }
}

// MARK: - Errors

enum SpanScorerError: Error, LocalizedError {
    case headWeightsNotFound(String)

    var errorDescription: String? {
        switch self {
        case .headWeightsNotFound(let name):
            return "Span head weights '\(name).bin' not found in app bundle"
        }
    }
}
