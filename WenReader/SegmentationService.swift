//
//  SegmentationService.swift
//  ChineseReader
//
//  Created by Oliver Zhang on 2025-11-29.
//

import Foundation

/// Protocol-based so you can swap impls or unit test easily.
protocol SegmentationService {
    /// Segment a single run of Chinese text (no punctuation).
    func segment(_ text: String) async -> [String]
}

/// Segments runs of Chinese using CEDICT via DictionaryService.
/// Uses a simple DP (Viterbi-like) algorithm preferring longer dictionary words.
final class CedictSegmentationService: SegmentationService {

    private let dict: DictionaryService
    private let maxWordLength: Int

    /// Simple in-memory cache so we don't hammer SQL on repeated substrings.
    private var containsCache: [String: Bool] = [:]

    init(dict: DictionaryService, maxWordLength: Int = 6) {
        self.dict = dict
        self.maxWordLength = maxWordLength
    }

    func segment(_ text: String) async -> [String] {
        // Treat empty as trivial.
        guard !text.isEmpty else { return [] }

        // Work on characters, but keep a mapping back to String.Index.
        let characters = Array(text)
        let n = characters.count

        // Map 0...n char indices to String.Index so we can slice later.
        var stringIndices: [String.Index] = []
        stringIndices.reserveCapacity(n + 1)

        var idx = text.startIndex
        stringIndices.append(idx)
        for _ in 0..<n {
            idx = text.index(after: idx)
            stringIndices.append(idx)
        }

        // DP arrays: bestScore[i] = best score for prefix ending at i
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
                let candidate = String(text[range])

                let isDictWord = await contains(candidate)

                // Skip non-dict, multi-char words.
                if len > 1 && !isDictWord { continue }

                let edgeScore = scoreToken(length: len, isDictWord: isDictWord)
                let total = base + edgeScore

                if total > bestScore[j] {
                    bestScore[j] = total
                    prevIndex[j] = i
                }
            }
        }

        // Backtrack best path.
        var tokens: [String] = []
        var pos = n

        while pos > 0 {
            let j = pos
            let i = prevIndex[j]

            let start: Int
            let end: Int

            if i >= 0 {
                start = i
                end = j
            } else {
                // Safety fallback: treat last char as its own token.
                start = j - 1
                end = j
            }

            let tokenRange = stringIndices[start]..<stringIndices[end]
            tokens.append(String(text[tokenRange]))

            pos = start
        }

        return tokens.reversed()
    }

    // MARK: - Scoring

    /// Higher = better. Multi-char dict words are preferred; single-char fallbacks penalized.
    private func scoreToken(length: Int, isDictWord: Bool) -> Double {
        if length == 1 {
            // Single characters:
            // - If it's actually in CEDICT (我, 他, 在), don't punish too much.
            // - If it's just a fallback (OOV), punish more so real words win.
            return isDictWord ? -0.5 : -2.0
        } else {
            // Multi-char, and (by construction) isDictWord == true here.
            // Reward longer words a bit more.
            // TODO: this needs to be more advanced.
            return Double(length) * 1.5
        }
    }

    // MARK: - Dictionary lookup with cache

    private func contains(_ word: String) async -> Bool {
        if let cached = containsCache[word] {
            return cached
        }

        let result = await dict.contains(word)
        containsCache[word] = result
        return result
    }
}
