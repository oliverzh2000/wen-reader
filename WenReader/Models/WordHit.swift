//
//  WordHit.swift
//  WenReader
//
//  Created by Oliver Zhang on 2025-12-08.
//

import UIKit

// Payload returned from JS side.
struct WordHit: Equatable {
    let block: String
    let sentence: String
    let run: String
    let word: String
    let hitPoint: CGPoint
    let rects: [CGRect]
}
