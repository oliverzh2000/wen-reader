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

/// A cross-reference like '個|个'.
/// Note that the pinyin fragment is parsed separately and will not form part of the clickable link.
struct LinkedHeadword: Hashable {
    let traditional: String
    let simplified: String
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

        // Two cases in one regex:
        //  1) ([\p{Han}]+)(?:\|([\p{Han}]+))?\[([A-Za-z0-9 ]+)\]
        //        ^head1         ^head2?           ^pinyinWithHead
        //
        //  2) \[([A-Za-z0-9 ]+)\]
        //           ^barePinyin
        //
        // So captures:
        //  1: head1 (Han+)
        //  2: head2 (Han+), optional
        //  3: pinyin when a headword is present
        //  4: pinyin when there is no headword (bare [fu4 qin5])
        let pattern = #"([\p{Han}]+)(?:\|([\p{Han}]+))?\[([A-Za-z0-9 ]+)\]|\[([A-Za-z0-9 ]+)\]"#

        guard let regex = try? NSRegularExpression(pattern: pattern, options: []) else {
            return Gloss(fragments: [.text(text)])
        }

        var fragments: [GlossFragment] = []
        var currentLocation = text.startIndex

        let nsText = text as NSString
        let matches = regex.matches(
            in: text,
            range: NSRange(location: 0, length: nsText.length)
        )

        for match in matches {
            guard let range = Range(match.range, in: text) else {
                continue
            }

            // Text before this match → plain text fragment.
            if currentLocation < range.lowerBound {
                let prefix = String(text[currentLocation..<range.lowerBound])
                if !prefix.isEmpty {
                    fragments.append(.text(prefix))
                }
            }

            // Case 1: headword + pinyin, e.g. 件[jian4] or 樁|桩[zhuang1]
            if
                let head1Range = Range(match.range(at: 1), in: text),
                let pinyinWithHeadRange = Range(match.range(at: 3), in: text)
            {
                let head1 = String(text[head1Range])
                var trad = head1
                var simp = head1

                if let head2Range = Range(match.range(at: 2), in: text) {
                    let head2 = String(text[head2Range])
                    // CC-CEDICT convention is trad|simp
                    trad = head1
                    simp = head2
                }

                let headword = LinkedHeadword(traditional: trad, simplified: simp)
                fragments.append(.link(headword))

                let numberedPinyin = String(text[pinyinWithHeadRange])
                let accented: [String] = numberedPinyin
                    .split(separator: " ")
                    .map { Self.numberedToAccentedPinyin(String($0)) }

                if !accented.isEmpty {
                    fragments.append(.accentedPinyin(accented))
                }
            }
            // Case 2: bare pinyin, e.g. [fu4 qin5]
            else if let barePinyinRange = Range(match.range(at: 4), in: text) {
                let numberedPinyin = String(text[barePinyinRange])
                let accented: [String] = numberedPinyin
                    .split(separator: " ")
                    .map { Self.numberedToAccentedPinyin(String($0)) }

                if !accented.isEmpty {
                    fragments.append(.accentedPinyin(accented))
                }
            }

            currentLocation = range.upperBound
        }

        // Any remaining trailing text after the last match.
        if currentLocation < text.endIndex {
            let suffix = String(text[currentLocation..<text.endIndex])
            if !suffix.isEmpty {
                fragments.append(.text(suffix))
            }
        }

        // If nothing was recognized, fall back to plain text.
        if fragments.isEmpty {
            fragments = [.text(text)]
        }

        return Gloss(fragments: fragments)
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
