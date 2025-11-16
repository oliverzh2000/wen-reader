//
//  DictionaryService.swift
//  ChineseReader
//
//  Created by Oliver Zhang on 2025-11-16.
//

import Foundation

protocol DictionaryService {
    func lookup(word: String) async -> DictionaryEntry?
}

struct DictionaryEntry {
    let headword: String
    let pinyin: [String]
    let definitions: [String]
    let gloss: String
}

final class CEDICTWithLLM: DictionaryService {
    func lookup(word: String) async -> DictionaryEntry? {
        // Use CC-CEDICT lookup to feed into LLM.
        return DictionaryEntry(headword: "", pinyin: ["ni", "hao"], definitions: [], gloss: "hello (greeting)")
    }
}
