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

/// Segments runs of Chinese text using CEDICT dictionary via DictionaryService.
///
/// # Algorithm: Dynamic Programming (Viterbi-style)
///
/// This implementation uses a forward DP approach similar to the Viterbi algorithm:
/// 1. Build DP table: `bestScore[i]` = best cumulative score for text prefix ending at position i
/// 2. Try all word lengths: For each position, consider all possible word endings (1 to maxWordLength chars)
/// 3. Score candidates: Rate potential words based on dictionary presence and length
/// 4. Backtrack: Reconstruct the optimal segmentation from the DP table
///
/// # Scoring System
///
/// The scoring favors longer dictionary words and penalizes unknown single characters:
/// - Multi-char dictionary words: +1.5 per character (strongly preferred)
///   - Example: "喜欢" (2 chars) = +3.0 points
/// - Single-char dictionary words: -0.5 (acceptable fallback)
///   - Example: "我" (common char in dict) = -0.5 points
/// - Single-char non-dictionary: -2.0 (heavily penalized, last resort)
///   - Example: rare/unknown character = -2.0 points
///
/// This scoring ensures we prefer "喜欢" as one word over "喜" + "欢" as two words.
///
/// # Example Segmentation
///
/// Input: "我喜欢看书" (I like reading books)
///
/// Process:
/// 1. Position 0→1: "我" (dict word, single char) → score -0.5
/// 2. Position 1→3: "喜欢" (dict word, 2 chars) → score +3.0
/// 3. Position 3→4: "看" (dict word, single char) → score -0.5
/// 4. Position 4→5: "书" (dict word, single char) → score -0.5
///
/// Result: ["我", "喜欢", "看", "书"]
/// Total score: -0.5 + 3.0 + (-0.5) + (-0.5) = +1.5
///
/// # Performance Optimizations
///
/// - Dictionary caching: Results cached to avoid repeated SQL queries
/// - Cache eviction: FIFO-style eviction when cache reaches maxCacheSize
/// - Max word length: Limits search space to reasonable word lengths (typically 4-6 chars)
///
/// # Future Improvements
///
/// - Add word frequency from CEDICT (prefer common words)
/// - Implement proper LRU cache for better hit rates
/// - Consider character-level language model scores
/// - Support ML-based segmentation models (transformer-based)
///
final class CedictSegmentationService: SegmentationService {

    private let dict: DictionaryService
    private let maxWordLength: Int

    /// In-memory cache for dictionary lookups to reduce SQL queries.
    /// Maps word → Bool (exists in dictionary).
    /// Limited to maxCacheSize to prevent unbounded memory growth during long reading sessions.
    private var containsCache: [String: Bool] = [:]
    private let maxCacheSize: Int

    init(dict: DictionaryService, maxWordLength: Int = ReaderConstants.Segmentation.maxWordLength) {
        self.dict = dict
        self.maxWordLength = maxWordLength
        self.maxCacheSize = ReaderConstants.Segmentation.maxCacheSize
    }

    /// Segments Chinese text into words using dynamic programming.
    ///
    /// - Parameter text: Chinese text to segment (should not contain punctuation)
    /// - Returns: Array of word strings in original order
    ///
    /// - Note: Empty strings return empty array. Single characters always succeed.
    func segment(_ text: String) async -> [String] {
        // Treat empty as trivial.
        guard !text.isEmpty else { return [] }

        // Convert String to Character array for easier indexing
        // Keep mapping to String.Index for final slicing (Swift String indexing is complex)
        let characters = Array(text)
        let n = characters.count

        // Build index map: position i → String.Index
        // This allows us to work with simple integer indices while preserving Unicode correctness
        var stringIndices: [String.Index] = []
        stringIndices.reserveCapacity(n + 1)

        var idx = text.startIndex
        stringIndices.append(idx)
        for _ in 0..<n {
            idx = text.index(after: idx)
            stringIndices.append(idx)
        }

        // DP table initialization
        // bestScore[i] = best cumulative score for segmenting text[0..<i]
        // prevIndex[i] = previous position in optimal path (for backtracking)
        var bestScore = Array(repeating: -Double.infinity, count: n + 1)
        var prevIndex = Array(repeating: -1, count: n + 1)

        // Base case: empty prefix has score 0
        bestScore[0] = 0.0

        // Forward pass: build DP table
        // For each position i, try all possible word endings within maxWordLength
        for i in 0..<n {
            let base = bestScore[i]
            // Skip unreachable positions
            if base == -Double.infinity { continue }

            // Try all word lengths from current position (1 to maxWordLength chars)
            let maxJ = min(n, i + maxWordLength)
            for j in (i + 1)...maxJ {
                let len = j - i
                let range = stringIndices[i]..<stringIndices[j]
                let candidate = String(text[range])

                // Check if candidate word exists in dictionary (with caching)
                let isDictWord = await contains(candidate)

                // Optimization: Skip multi-char words that aren't in dictionary
                // Single chars always allowed as fallback for unknown characters
                if len > 1 && !isDictWord { continue }

                // Calculate score for this word choice
                let edgeScore = scoreToken(length: len, isDictWord: isDictWord)
                let total = base + edgeScore

                // Update DP table if this path is better
                if total > bestScore[j] {
                    bestScore[j] = total
                    prevIndex[j] = i
                }
            }
        }

        // Backward pass: reconstruct optimal segmentation by backtracking
        var tokens: [String] = []
        var pos = n

        // Walk backwards from end, following prevIndex pointers
        while pos > 0 {
            let j = pos
            let i = prevIndex[j]

            let start: Int
            let end: Int

            if i >= 0 {
                // Normal case: use stored previous position
                start = i
                end = j
            } else {
                // Safety fallback: if DP failed, treat last char as single token
                // This should rarely happen unless text is completely unknown
                start = j - 1
                end = j
            }

            // Extract word using String.Index for Unicode safety
            let tokenRange = stringIndices[start]..<stringIndices[end]
            tokens.append(String(text[tokenRange]))

            pos = start
        }

        // Tokens were collected backwards, so reverse to get correct order
        return tokens.reversed()
    }

    // MARK: - Scoring Functions

    /// Calculates score for a token based on its properties.
    ///
    /// Scoring Philosophy:
    /// - Favor longer dictionary words (they convey meaning)
    /// - Allow single-char dictionary words (common particles/chars)
    /// - Heavily penalize unknown single chars (likely segmentation errors)
    ///
    /// - Parameters:
    ///   - length: Number of characters in the token
    ///   - isDictWord: Whether token exists in CEDICT dictionary
    /// - Returns: Score contribution (higher is better)
    private func scoreToken(length: Int, isDictWord: Bool) -> Double {
        if length == 1 {
            // Single-character tokens
            if isDictWord {
                // Known single-char word (e.g., 我, 他, 在, 的)
                // Small penalty to prefer multi-char words when possible
                return -0.5
            } else {
                // Unknown character (OOV - out of vocabulary)
                // Large penalty as last resort for rare/foreign chars
                return -2.0
            }
        } else {
            // Multi-character tokens
            // By construction, isDictWord must be true here (we skip multi-char non-dict words)
            // Reward proportional to length to prefer longer meaningful words
            // Example: "喜欢" (2 chars) scores 3.0, beating "喜" + "欢" (2 × -0.5 = -1.0)
            //
            // TODO Future improvements:
            // - Add word frequency from CEDICT (common words get bonus)
            // - Consider character-level language model scores
            // - Tune coefficients based on segmentation accuracy metrics
            return Double(length) * 1.5
        }
    }

    // MARK: - Dictionary Lookup with Caching

    /// Checks if a word exists in the dictionary, with in-memory caching.
    ///
    /// Cache Strategy:
    /// - Speeds up segmentation by avoiding repeated SQL queries
    /// - FIFO eviction when cache is full (simple but effective)
    /// - Could be improved with LRU eviction for better hit rates
    ///
    /// - Parameter word: Word to look up
    /// - Returns: true if word exists in CEDICT dictionary
    private func contains(_ word: String) async -> Bool {
        // Check cache first
        if let cached = containsCache[word] {
            return cached
        }

        // Cache miss: query dictionary service
        let result = await dict.contains(word)
        
        // Limit cache size to prevent unbounded memory growth
        if containsCache.count >= maxCacheSize {
            // Simple FIFO eviction: Remove oldest 50% of entries
            // Pros: Simple, predictable, prevents pathological cases
            // Cons: Not optimal for access patterns (LRU would be better)
            //
            // Dictionary.keys order is undefined, but this gives us ~FIFO behavior
            // in practice since we add entries sequentially during segmentation
            let keysToRemove = Array(containsCache.keys.prefix(maxCacheSize / 2))
            keysToRemove.forEach { containsCache.removeValue(forKey: $0) }
        }
        
        containsCache[word] = result
        return result
    }
}
