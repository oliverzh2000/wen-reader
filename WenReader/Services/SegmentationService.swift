// Copyright 2025 Oliver Zhang
// Licensed under the MIT License

import Foundation

/// Protocol-based so you can swap impls or unit test easily.
protocol SegmentationService {
    /// Segment a run of Chinese text, using the full sentence for encoder context.
    ///
    /// The encoder sees the full sentence (richer hidden states from surrounding
    /// context), but only the run portion is segmented and returned.
    func segment(run: String, sentence: String) async -> [String]
}

/// Segments runs of Chinese text using CEDICT dictionary via DictionaryService.
///
/// Uses a forward DP approach (Viterbi-style) with hand-tuned scoring that
/// favors longer dictionary words. This is the fallback when the span scorer
/// CoreML model is unavailable.
final class CedictSegmentationService: SegmentationService {

    private let dict: DictionaryService
    private let maxWordLength: Int
    private var containsCache: [String: Bool] = [:]
    private let maxCacheSize: Int

    init(dict: DictionaryService, maxWordLength: Int = ReaderConstants.Segmentation.maxWordLength) {
        self.dict = dict
        self.maxWordLength = maxWordLength
        self.maxCacheSize = ReaderConstants.Segmentation.maxCacheSize
    }

    func segment(run: String, sentence: String) async -> [String] {
        // Fallback ignores sentence context — just segments the run.
        guard !run.isEmpty else { return [] }

        let characters = Array(run)
        let n = characters.count

        var stringIndices: [String.Index] = []
        stringIndices.reserveCapacity(n + 1)
        var idx = run.startIndex
        stringIndices.append(idx)
        for _ in 0..<n {
            idx = run.index(after: idx)
            stringIndices.append(idx)
        }

        var bestScore = Array(repeating: -Double.infinity, count: n + 1)
        var prevIndex = Array(repeating: -1, count: n + 1)
        bestScore[0] = 0.0

        for i in 0..<n {
            let base = bestScore[i]
            if base == -Double.infinity { continue }

            let maxJ = min(n, i + maxWordLength)
            for j in (i + 1)...maxJ {
                let len = j - i
                let range = stringIndices[i]..<stringIndices[j]
                let candidate = String(run[range])

                let isDictWord = await contains(candidate)
                if len > 1 && !isDictWord { continue }

                let edgeScore = scoreToken(length: len, isDictWord: isDictWord)
                let total = base + edgeScore

                if total > bestScore[j] {
                    bestScore[j] = total
                    prevIndex[j] = i
                }
            }
        }

        var tokens: [String] = []
        var pos = n
        while pos > 0 {
            let j = pos
            let i = prevIndex[j]
            let start = i >= 0 ? i : j - 1
            let end = j
            let tokenRange = stringIndices[start]..<stringIndices[end]
            tokens.append(String(run[tokenRange]))
            pos = start
        }

        return tokens.reversed()
    }

    private func scoreToken(length: Int, isDictWord: Bool) -> Double {
        if length == 1 {
            return isDictWord ? -0.5 : -2.0
        } else {
            return Double(length) * 1.5
        }
    }

    private func contains(_ word: String) async -> Bool {
        if let cached = containsCache[word] { return cached }
        let result = await dict.contains(word)
        if containsCache.count >= maxCacheSize {
            let keysToRemove = Array(containsCache.keys.prefix(maxCacheSize / 2))
            keysToRemove.forEach { containsCache.removeValue(forKey: $0) }
        }
        containsCache[word] = result
        return result
    }
}
