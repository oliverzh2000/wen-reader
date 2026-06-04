// Copyright 2025 Oliver Zhang
// Licensed under the MIT License

import UIKit

// Payload returned from JS side.
struct WordHit: Equatable {
    let block: String
    let sentence: String
    let run: String
    let word: String
    let hitPoint: CGPoint
    let rects: [CGRect]
    /// Character offset of the word within `sentence`. Used for WSD disambiguation
    /// when the same word appears multiple times in a sentence.
    let wordOffsetInSentence: Int
}
