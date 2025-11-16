//
//  DictionaryService.swift
//  ChineseReader
//
//  Created by Oliver Zhang on 2025-11-16.
//

import Foundation

protocol DictionaryService {
    /// Return the full dictionary entry (all senses) for a word, if present.
    func lookup(_ word: String) async -> DictionaryEntry?
    
    /// Return a concise gloss for the word at `wordIndex` in `sentence`.
    /// Later we’ll implement this with the LLM; for now it can be a dumb fallback.
    func gloss(atIndex wordIndex: Int, in sentence: [String]) async throws -> String?
}

struct DictionaryEntry {
    let headword: String        // The lookup key, e.g. "长"
    let senses: [Sense]         // All possible readings/pronunciations/meanings
    
    struct Sense {
        let traditional: String // "長"
        let simplified: String  // "长"
        let pinyin: [String]    // ["chang2"], or ["ji1", "chu3"] for multi-char words
        let definitions: [String]
    }
}

final class CEDICTWithLLM: DictionaryService {
    // MARK: - Static dictionary storage
    /// Loaded once per process, shared by all instances.
    private static let entries: [String: DictionaryEntry] = {
        loadDictionary()
    }()
    
    // MARK: - DictionaryService
    func lookup(_ word: String) async -> DictionaryEntry? {
        Self.entries[word]
    }
    
    func gloss(atIndex wordIndex: Int, in sentence: [String]) async throws -> String? {
        guard sentence.indices.contains(wordIndex) else { return nil }
        
        let word = sentence[wordIndex]
        guard let entry = Self.entries[word] else { return nil }
        
        // TODO: later
        //  - send `word`, `sentence`, and `entry` to the LLM
        //  - pick the best sense and compress to a concise gloss.
        //
        // For now: dumb fallback – first definition of the first sense.
        return entry.senses.first?.definitions.first
    }
    
    // MARK: - CEDICT loading
    private static func loadDictionary() -> [String: DictionaryEntry] {
        guard let url = Bundle.main.url(forResource: "cedict", withExtension: "json") else {
            fatalError("Could not find cedict.json in bundle resources.")
        }
        
        do {
            let data = try Data(contentsOf: url)
            
            // { "长": [RawCEDICTEntry, ...], "系": [...], ... }
            let raw = try JSONDecoder().decode([String: [RawCEDICTEntry]].self, from: data)
            
            var result: [String: DictionaryEntry] = [:]
            result.reserveCapacity(raw.count)
            
            for (headword, rawSenses) in raw {
                let senses: [DictionaryEntry.Sense] = rawSenses.map { raw in
                    DictionaryEntry.Sense(
                        traditional: raw.t,
                        simplified: raw.s,
                        pinyin: raw.p
                            .split(separator: " ")
                            .map { numberedToAccentedPinyin(String($0)) },
                        definitions: raw.d
                    )
                }
                
                result[headword] = DictionaryEntry(
                    headword: headword,
                    senses: senses
                )
            }
            
            return result
        } catch {
            fatalError("Failed to load cedict.json: \(error)")
        }
    }
    
    private static func numberedToAccentedPinyin(_ numbered: String) -> String {
        // 1. Extract tone number (1–5)
        guard let toneChar = numbered.last,
              let tone = Int(String(toneChar)),
              tone >= 1 && tone <= 5 else {
            return numbered  // already accented or malformed
        }
        
        let base = String(numbered.dropLast())
        
        // 2. Convert u:, v → ü normalization
        let normalized = base
            .replacingOccurrences(of: "u:", with: "ü")
            .replacingOccurrences(of: "v", with: "ü")
        
        // 3. Vowel priority list for tone placement
        let vowels = ["a", "e", "o", "u", "i", "ü"]
        
        // 4. Which vowel gets the mark?
        var targetIndex: String.Index? = nil
        
        // Rule: a → e → ou → last vowel
        if let i = normalized.firstIndex(of: "a") {
            targetIndex = i
        } else if let i = normalized.firstIndex(of: "e") {
            targetIndex = i
        } else if normalized.contains("ou"),
                  let i = normalized.firstIndex(of: "o") {
            targetIndex = i
        } else {
            // last vowel
            targetIndex = normalized.lastIndex(where: { vowels.contains(String($0)) })
        }
        
        guard let idx = targetIndex else {
            return normalized // no vowel? return unchanged
        }
        
        let vowel = normalized[idx]
        
        // 5. Mapping vowel + tone → accented vowel
        let toneMarks: [Character: [Character]] = [
            "a": ["ā","á","ǎ","à","a"],
            "e": ["ē","é","ě","è","e"],
            "i": ["ī","í","ǐ","ì","i"],
            "o": ["ō","ó","ǒ","ò","o"],
            "u": ["ū","ú","ǔ","ù","u"],
            "ü": ["ǖ","ǘ","ǚ","ǜ","ü"]
        ]
        
        let accented = toneMarks[vowel]?[tone - 1] ?? vowel
        
        // 6. Replace vowel with accented vowel
        var result = normalized
        result.replaceSubrange(idx...idx, with: String(accented))
        
        return result
    }

    
    private struct RawCEDICTEntry: Decodable {
        let t: String       // traditional
        let s: String       // simplified
        let p: String       // numbered pinyin like "chang2" or "ji1 chu3"
        let d: [String]     // definitions
    }
}
