//
//  DictionaryService.swift
//  ChineseReader
//
//  Created by Oliver Zhang on 2025-11-16.
//

import Foundation
import SQLite3

protocol DictionaryService {
    /// Return the full dictionary entry (all senses) for a word, if present.
    func lookup(_ word: String) async -> DictionaryResult?
}

// All dictionary entries returned for a lookup.
// Typically: all possible readings (pronunciations) and meanings for a given written form.
struct DictionaryResult {
    let entries: [Entry]
}

// One pronunciation of a specific written form (unique trad/simp/pinyin triple in CC-CEDICT).
struct Entry {
    let traditional: String
    let simplified: String
    let accentedPinyin: [String]
    let senses: [Sense]
}

// One logically distinct meaning for this pronunciation.
struct Sense {
    // Each gloss can be made up of plain text and clickable links.
    let glosses: [Gloss]
}

/// A single gloss, made up of fragments.
struct Gloss: Hashable {
    let fragments: [GlossFragment]
}

/// A piece of a gloss: either plain text, pinyin, or link to another headword.
enum GlossFragment: Hashable {
    case text(String)
    case accentedPinyin([String])
    case link(LinkedHeadword)
}

/// A cross-reference like 個|个[ge4].
struct LinkedHeadword: Hashable {
    let traditional: String
    let simplified: String
    let accentedPinyin: [String]
}


private let SQLITE_TRANSIENT = unsafeBitCast(-1, to: sqlite3_destructor_type.self)

final class CedictSqlService: DictionaryService {
    static let shared = CedictSqlService()
    private var db: OpaquePointer?

    // Adjust to match how you bundle the DB
    private let dbFileName = "cedict"      // cedict.sqlite -> "cedict"
    private let dbFileExtension = "sqlite"

    private init() {
        openDatabaseIfNeeded()
    }

    deinit {
        if let db {
            sqlite3_close(db)
        }
    }

    // Optional: eager load
    func forceLoad() {
        openDatabaseIfNeeded()
    }

    func lookup(_ word: String) async -> DictionaryResult? {
        openDatabaseIfNeeded()
        guard let db else { return nil }

        let sql = """
        SELECT trad, simp, pinyin, senses_raw
        FROM cedict_entries
        WHERE trad = ?1 OR simp = ?1;
        """

        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
            let msg = String(cString: sqlite3_errmsg(db))
            print("CEDICTWithLLM: prepare failed: \(msg)")
            return nil
        }
        defer { sqlite3_finalize(stmt) }

        // Bind the same word to both trad and simp via ?1
        (word as NSString).utf8String.map {
            sqlite3_bind_text(stmt, 1, $0, -1, SQLITE_TRANSIENT)
        }

        var entries: [Entry] = []

        while sqlite3_step(stmt) == SQLITE_ROW {
            guard
                let tradC = sqlite3_column_text(stmt, 0),
                let simpC = sqlite3_column_text(stmt, 1),
                let pinyinC = sqlite3_column_text(stmt, 2),
                let sensesC = sqlite3_column_text(stmt, 3)
            else {
                continue
            }

            let trad = String(cString: tradC)
            let simp = String(cString: simpC)
            let pinyinRaw = String(cString: pinyinC)
            let sensesRaw = String(cString: sensesC)

            let accentedPinyin: [String] = pinyinRaw
                .split(separator: " ")
                .map { Self.numberedToAccentedPinyin(String($0)) }

            let senses = Self.parseSenses(from: sensesRaw)

            entries.append(
                Entry(
                    traditional: trad,
                    simplified: simp,
                    accentedPinyin: accentedPinyin,
                    senses: senses
                )
            )
        }

        guard !entries.isEmpty else { return nil }
        return DictionaryResult(entries: entries)
    }

    // MARK: - DB open
    private func openDatabaseIfNeeded() {
        guard db == nil else { return }

        guard let url = Bundle.main.url(forResource: dbFileName, withExtension: dbFileExtension) else {
            print("CEDICTWithLLM: could not find cedict DB in bundle")
            return
        }

        var handle: OpaquePointer?
        let rc = sqlite3_open_v2(url.path, &handle, SQLITE_OPEN_READONLY, nil)
        if rc != SQLITE_OK {
            let msg = String(cString: sqlite3_errmsg(handle))
            print("CEDICTWithLLM: failed to open DB: \(msg)")
            if let handle { sqlite3_close(handle) }
            return
        }

        db = handle
    }

    // MARK: - Parsing helpers
    private static func parseSenses(from raw: String) -> [Sense] {
        // "sense1/sense2/sense3", each sense = "gloss1; gloss2"
        let senseStrings = raw
            .split(separator: "/")
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }

        return senseStrings.map { senseStr in
            let glossStrings = senseStr
                .split(separator: ";")
                .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                .filter { !$0.isEmpty }

            let glosses: [Gloss] = glossStrings.map { glossStr in
                parseGloss(String(glossStr))
            }

            return Sense(glosses: glosses)
        }
    }
    
    private static func parseGloss(_ raw: String) -> Gloss {
        let text = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else {
            return Gloss(fragments: [])
        }

        // Pattern for trad|simp[pinyin] (standard CC-CEDICT form).
        // \p{Han} covers Chinese characters reasonably well.
        let pattern = #"([\p{Han}]+)\|([\p{Han}]+)\[([A-Za-z0-9 ]+)\]"#

        guard let regex = try? NSRegularExpression(pattern: pattern, options: []) else {
            // No link regex? Still try to pull out [pinyin].
            let fragments = splitTextAndPinyin(text)
            return Gloss(fragments: fragments)
        }

        var fragments: [GlossFragment] = []
        var currentLocation = text.startIndex

        let nsText = text as NSString
        let matches = regex.matches(
            in: text,
            range: NSRange(location: 0, length: nsText.length)
        )

        for match in matches {
            guard
                let range = Range(match.range, in: text),
                let tradRange = Range(match.range(at: 1), in: text),
                let simpRange = Range(match.range(at: 2), in: text),
                let pinyinRange = Range(match.range(at: 3), in: text)
            else {
                continue
            }

            // Text before this link → may contain [pinyin], so split it.
            if currentLocation < range.lowerBound {
                let prefix = String(text[currentLocation..<range.lowerBound])
                if !prefix.isEmpty {
                    fragments.append(contentsOf: splitTextAndPinyin(prefix))
                }
            }

            // The matched headword → link fragment.
            let trad = String(text[tradRange])
            let simp = String(text[simpRange])

            let accentedPinyin: [String] = text[pinyinRange]
                .split(separator: " ")
                .map { Self.numberedToAccentedPinyin(String($0)) }

            let headword = LinkedHeadword(
                traditional: trad,
                simplified: simp,
                accentedPinyin: accentedPinyin
            )
            fragments.append(.link(headword))

            currentLocation = range.upperBound
        }

        // Any remaining text after the last link → may contain [pinyin].
        if currentLocation < text.endIndex {
            let suffix = String(text[currentLocation..<text.endIndex])
            if !suffix.isEmpty {
                fragments.append(contentsOf: splitTextAndPinyin(suffix))
            }
        }

        // If we somehow didn’t find any fragments, try pure [pinyin] detection.
        if fragments.isEmpty {
            fragments = splitTextAndPinyin(text)
        }

        return Gloss(fragments: fragments)
    }


    private static func splitTextAndPinyin(_ text: String) -> [GlossFragment] {
        guard !text.isEmpty else { return [] }

        // Matches [fu4 qin5] with content captured inside the brackets.
        let pattern = #"\[([A-Za-z0-9 ]+)\]"#

        guard let regex = try? NSRegularExpression(pattern: pattern, options: []) else {
            return [.text(text)]
        }

        var fragments: [GlossFragment] = []
        var currentLocation = text.startIndex

        let nsText = text as NSString
        let matches = regex.matches(
            in: text,
            range: NSRange(location: 0, length: nsText.length)
        )

        for match in matches {
            guard
                let range = Range(match.range, in: text),
                let pinyinRange = Range(match.range(at: 1), in: text)
            else {
                continue
            }

            // Text before this [pinyin] → plain text.
            if currentLocation < range.lowerBound {
                let prefix = String(text[currentLocation..<range.lowerBound])
                if !prefix.isEmpty {
                    fragments.append(.text(prefix))
                }
            }

            // The pinyin inside [...] → accent-converted pinyin fragment.
            let numbered = String(text[pinyinRange])
            let accentedPinyin: [String] = numbered
                .split(separator: " ")
                .map { Self.numberedToAccentedPinyin(String($0)) }
            fragments.append(.accentedPinyin(accentedPinyin))

            currentLocation = range.upperBound
        }

        // Any trailing text after the last [pinyin]
        if currentLocation < text.endIndex {
            let suffix = String(text[currentLocation..<text.endIndex])
            if !suffix.isEmpty {
                fragments.append(.text(suffix))
            }
        }

        if fragments.isEmpty {
            fragments = [.text(text)]
        }

        return fragments
    }


    // MARK: - Pinyin conversion
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
}
