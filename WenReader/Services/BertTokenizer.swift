// Copyright 2025 Oliver Zhang
// Licensed under the MIT License

import Foundation

/// Result of BERT tokenization: token IDs, attention mask, and offset mapping.
struct TokenizerOutput {
    /// Token IDs including [CLS] and [SEP].
    let inputIds: [Int32]
    /// Attention mask: 1 for real tokens, 0 for padding.
    let attentionMask: [Int32]
    /// Maps each token to its character range (start, end) in the original string.
    /// [CLS] and [SEP] map to (0, 0).
    let offsetMapping: [(Int, Int)]
}

/// A minimal BERT WordPiece tokenizer matching HuggingFace `BertTokenizerFast` behavior.
///
/// Designed to be shared between CWS (`ckiplab/bert-base-chinese-ws`) and WSD
/// (`thenlper/gte-base-zh`) models by accepting different vocab files at init.
///
/// Tokenization rules:
/// - CJK characters are each treated as a single token (no WordPiece splitting)
/// - Non-CJK text (Latin, punctuation) uses WordPiece splitting with `##` continuation prefix
/// - Output is `[CLS]` + tokens + `[SEP]` with attention mask of all 1s
/// - Offset mapping tracks each token's character range for subword-to-character alignment
final class BertTokenizer {

    // MARK: - Special Token IDs

    static let padTokenId: Int32 = 0
    static let unkTokenId: Int32 = 100
    static let clsTokenId: Int32 = 101
    static let sepTokenId: Int32 = 102

    /// Maps token string → token ID.
    private let vocab: [String: Int32]

    /// Maximum number of characters for WordPiece splitting of a single word.
    /// Words longer than this are mapped to [UNK].
    private let maxWordPieceLength = 200

    // MARK: - Initialization

    /// Initialize from a vocab file at the given file path.
    /// The file should have one token per line; the line number (0-based) is the token ID.
    init(vocabFilePath: String) throws {
        let content = try String(contentsOfFile: vocabFilePath, encoding: .utf8)
        self.vocab = Self.buildVocab(from: content)
    }

    /// Initialize from a vocab file bundled in the app.
    /// - Parameter resourceName: The resource name (without extension) in the main bundle.
    /// - Parameter extension: The file extension (default: "txt").
    init(resourceName: String, withExtension ext: String = "txt") throws {
        guard let url = Bundle.main.url(forResource: resourceName, withExtension: ext) else {
            throw TokenizerError.vocabFileNotFound(resourceName)
        }
        let content = try String(contentsOf: url, encoding: .utf8)
        self.vocab = Self.buildVocab(from: content)
    }

    /// Initialize directly from vocab file content string (useful for testing).
    init(vocabContent: String) {
        self.vocab = Self.buildVocab(from: vocabContent)
    }

    private static func buildVocab(from content: String) -> [String: Int32] {
        var vocab: [String: Int32] = [:]
        // Split strictly on \n only. CharacterSet.newlines includes NEL (U+0085)
        // which appears as an actual token in some BERT vocabs (e.g. MacBERT),
        // causing ID shifts if treated as a line separator.
        let lines = content.split(separator: "\n", omittingEmptySubsequences: false)
        for (index, line) in lines.enumerated() {
            let token = String(line)
            if !token.isEmpty {
                vocab[token] = Int32(index)
            }
        }
        return vocab
    }

    // MARK: - Public API

    /// Encode a string into BERT token IDs with attention mask and offset mapping.
    ///
    /// Produces `[CLS]` + tokens + `[SEP]`. No padding is applied.
    /// The offset mapping maps each token back to its character range in the original string,
    /// which is critical for CWS to map per-token logits back to per-character probabilities.
    func encode(_ text: String) -> TokenizerOutput {
        let preTokenized = preTokenize(text)

        var inputIds: [Int32] = [Self.clsTokenId]
        var offsets: [(Int, Int)] = [(0, 0)] // [CLS] has no character mapping

        for segment in preTokenized {
            let (ids, segOffsets) = tokenizeSegment(segment)
            inputIds.append(contentsOf: ids)
            offsets.append(contentsOf: segOffsets)
        }

        inputIds.append(Self.sepTokenId)
        offsets.append((0, 0)) // [SEP] has no character mapping

        let attentionMask = [Int32](repeating: 1, count: inputIds.count)

        return TokenizerOutput(
            inputIds: inputIds,
            attentionMask: attentionMask,
            offsetMapping: offsets
        )
    }

    /// Returns the vocab size.
    var vocabSize: Int { vocab.count }

    /// Look up a token's ID, returning [UNK] if not found.
    func tokenToId(_ token: String) -> Int32 {
        vocab[token] ?? Self.unkTokenId
    }

    // MARK: - Pre-tokenization

    /// A segment of text identified during pre-tokenization.
    private struct PreTokenSegment {
        let text: String
        /// Character offset in the original string where this segment starts.
        let startOffset: Int
        /// Whether this segment is a single CJK character.
        let isCJK: Bool
    }

    /// Split text into segments: each CJK character becomes its own segment,
    /// and runs of non-CJK characters are grouped together.
    /// This mirrors `BertTokenizerFast`'s `tokenize_chinese_chars` behavior.
    private func preTokenize(_ text: String) -> [PreTokenSegment] {
        var segments: [PreTokenSegment] = []
        var nonCJKBuffer = ""
        var nonCJKStart = -1
        var charOffset = 0

        for scalar in text.unicodeScalars {
            let char = Character(scalar)
            if Self.isCJKCharacter(scalar) {
                // Flush any accumulated non-CJK text
                if !nonCJKBuffer.isEmpty {
                    segments.append(contentsOf: splitNonCJKBuffer(nonCJKBuffer, startOffset: nonCJKStart))
                    nonCJKBuffer = ""
                    nonCJKStart = -1
                }
                // Each CJK character is its own segment
                segments.append(PreTokenSegment(
                    text: String(char),
                    startOffset: charOffset,
                    isCJK: true
                ))
            } else {
                if nonCJKBuffer.isEmpty {
                    nonCJKStart = charOffset
                }
                nonCJKBuffer.append(char)
            }
            charOffset += 1
        }

        // Flush remaining non-CJK text
        if !nonCJKBuffer.isEmpty {
            segments.append(contentsOf: splitNonCJKBuffer(nonCJKBuffer, startOffset: nonCJKStart))
        }

        return segments
    }

    /// Split a non-CJK buffer into word-level segments by whitespace and punctuation.
    /// BERT's BasicTokenizer splits on whitespace and punctuation characters.
    private func splitNonCJKBuffer(_ buffer: String, startOffset: Int) -> [PreTokenSegment] {
        var segments: [PreTokenSegment] = []
        var currentWord = ""
        var wordStart = startOffset

        for (i, scalar) in buffer.unicodeScalars.enumerated() {
            let char = Character(scalar)
            if Self.isWhitespace(scalar) {
                // Flush current word
                if !currentWord.isEmpty {
                    segments.append(PreTokenSegment(
                        text: currentWord,
                        startOffset: wordStart,
                        isCJK: false
                    ))
                    currentWord = ""
                }
                wordStart = startOffset + i + 1
            } else if Self.isPunctuation(scalar) {
                // Flush current word
                if !currentWord.isEmpty {
                    segments.append(PreTokenSegment(
                        text: currentWord,
                        startOffset: wordStart,
                        isCJK: false
                    ))
                    currentWord = ""
                }
                // Punctuation is its own segment
                segments.append(PreTokenSegment(
                    text: String(char),
                    startOffset: startOffset + i,
                    isCJK: false
                ))
                wordStart = startOffset + i + 1
            } else {
                if currentWord.isEmpty {
                    wordStart = startOffset + i
                }
                currentWord.append(char)
            }
        }

        if !currentWord.isEmpty {
            segments.append(PreTokenSegment(
                text: currentWord,
                startOffset: wordStart,
                isCJK: false
            ))
        }

        return segments
    }

    // MARK: - Tokenization

    /// Tokenize a single segment into token IDs and offset mappings.
    private func tokenizeSegment(_ segment: PreTokenSegment) -> ([Int32], [(Int, Int)]) {
        if segment.isCJK {
            // CJK characters: direct vocab lookup, no WordPiece
            let id = tokenToId(segment.text)
            return ([id], [(segment.startOffset, segment.startOffset + 1)])
        } else {
            // Non-CJK: apply WordPiece splitting
            return wordPieceTokenize(segment.text, startOffset: segment.startOffset)
        }
    }

    /// WordPiece tokenization for non-CJK text.
    ///
    /// Tries to find the longest matching prefix in vocab. If the first character
    /// isn't found, the whole word maps to [UNK]. Continuation pieces use `##` prefix.
    private func wordPieceTokenize(_ word: String, startOffset: Int) -> ([Int32], [(Int, Int)]) {
        let lowercased = word.lowercased()
        let chars = Array(lowercased)

        guard chars.count <= maxWordPieceLength else {
            // Word too long — treat as [UNK] spanning the whole word
            return ([Self.unkTokenId], [(startOffset, startOffset + word.count)])
        }

        var ids: [Int32] = []
        var offsets: [(Int, Int)] = []
        var start = 0

        while start < chars.count {
            var end = chars.count
            var matched = false

            while start < end {
                let substr: String
                if start == 0 {
                    substr = String(chars[start..<end])
                } else {
                    substr = "##" + String(chars[start..<end])
                }

                if let id = vocab[substr] {
                    ids.append(id)
                    offsets.append((startOffset + start, startOffset + end))
                    matched = true
                    start = end
                    break
                }

                end -= 1
            }

            if !matched {
                // No match found even for a single character — whole word is [UNK]
                return ([Self.unkTokenId], [(startOffset, startOffset + word.count)])
            }
        }

        return (ids, offsets)
    }

    // MARK: - Character Classification

    /// Returns true if the Unicode scalar is a CJK Unified Ideograph.
    /// Covers the ranges specified in the design document.
    static func isCJKCharacter(_ scalar: Unicode.Scalar) -> Bool {
        let cp = scalar.value
        return (cp >= 0x4E00 && cp <= 0x9FFF)       // CJK Unified Ideographs
            || (cp >= 0x3400 && cp <= 0x4DBF)        // CJK Unified Ideographs Extension A
            || (cp >= 0x20000 && cp <= 0x2A6DF)      // CJK Unified Ideographs Extension B
            || (cp >= 0x2A700 && cp <= 0x2B73F)      // CJK Unified Ideographs Extension C
            || (cp >= 0x2B740 && cp <= 0x2B81F)      // CJK Unified Ideographs Extension D
            || (cp >= 0x2B820 && cp <= 0x2CEAF)      // CJK Unified Ideographs Extension E
            || (cp >= 0xF900 && cp <= 0xFAFF)        // CJK Compatibility Ideographs
            || (cp >= 0x2F800 && cp <= 0x2FA1F)      // CJK Compatibility Ideographs Supplement
    }

    /// Returns true if the scalar is whitespace (matching BERT's definition).
    private static func isWhitespace(_ scalar: Unicode.Scalar) -> Bool {
        let cp = scalar.value
        if cp == 0x20 || cp == 0x09 || cp == 0x0A || cp == 0x0D {
            return true
        }
        return CharacterSet.whitespaces.contains(scalar)
    }

    /// Returns true if the scalar is punctuation (matching BERT's definition).
    /// BERT treats ASCII punctuation and Unicode punctuation categories as punctuation.
    private static func isPunctuation(_ scalar: Unicode.Scalar) -> Bool {
        let cp = scalar.value
        // ASCII punctuation ranges
        if (cp >= 33 && cp <= 47) || (cp >= 58 && cp <= 64)
            || (cp >= 91 && cp <= 96) || (cp >= 123 && cp <= 126) {
            return true
        }
        // Unicode general punctuation categories
        return CharacterSet.punctuationCharacters.contains(scalar)
    }
}

// MARK: - Errors

enum TokenizerError: Error, LocalizedError {
    case vocabFileNotFound(String)

    var errorDescription: String? {
        switch self {
        case .vocabFileNotFound(let name):
            return "Vocab file '\(name)' not found in app bundle"
        }
    }
}
