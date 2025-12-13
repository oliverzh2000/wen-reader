// Copyright 2025 Oliver Zhang
// Licensed under the MIT License

import Foundation
import SQLite3

protocol DictionaryService {
    /// Return the full dictionary entry (all senses) for a word, if present.
    func lookup(_ word: String) async -> DictionaryResult?

    /// Return true if `word` exists in CEDICT (as trad or simp), false otherwise.
    func contains(_ word: String) async -> Bool
}

private let SQLITE_TRANSIENT = unsafeBitCast(
    -1,
    to: sqlite3_destructor_type.self
)

final class CedictSqlService: DictionaryService {
    static let shared = CedictSqlService()
    private var db: OpaquePointer?
    private let dbQueue = DispatchQueue(label: "com.wenreader.dictionary", qos: .userInitiated)

    // Adjust to match how you bundle the DB
    private let dbFileName = "cedict"  // cedict.sqlite -> "cedict"
    private let dbFileExtension = "sqlite"

    private init() {
        openDatabaseIfNeeded()
    }

    deinit {
        closeDatabase()
    }
    
    private func closeDatabase() {
        if let db {
            sqlite3_close(db)
            self.db = nil
        }
    }

    // Optional: eager load
    func forceLoad() {
        openDatabaseIfNeeded()
    }

    func lookup(_ word: String) async -> DictionaryResult? {
        return await withCheckedContinuation { continuation in
            dbQueue.async { [weak self] in
                guard let self else {
                    continuation.resume(returning: nil)
                    return
                }
                
                self.openDatabaseIfNeeded()
                guard self.db != nil else {
                    continuation.resume(returning: nil)
                    return
                }

                let sql = """
                    SELECT trad, simp, pinyin, senses_raw
                    FROM cedict_entries
                    WHERE trad = ?1 OR simp = ?1;
                    """

                var stmt: OpaquePointer?
                guard self.prepareAndBind(sql, word: word, stmt: &stmt) else {
                    continuation.resume(returning: nil)
                    return
                }
                defer { 
                    if let stmt {
                        sqlite3_finalize(stmt)
                    }
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

                    let accentedPinyin: [String] =
                        pinyinRaw
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

                let result = entries.isEmpty ? nil : DictionaryResult(entries: entries)
                continuation.resume(returning: result)
            }
        }
    }

    func contains(_ word: String) async -> Bool {
        return await withCheckedContinuation { continuation in
            dbQueue.async { [weak self] in
                guard let self else {
                    continuation.resume(returning: false)
                    return
                }
                
                self.openDatabaseIfNeeded()
                guard let db = self.db else {
                    continuation.resume(returning: false)
                    return
                }

                let sql = """
                    SELECT 1
                    FROM cedict_entries
                    WHERE trad = ?1 OR simp = ?1
                    LIMIT 1;
                    """

                var stmt: OpaquePointer?
                guard self.prepareAndBind(sql, word: word, stmt: &stmt) else {
                    continuation.resume(returning: false)
                    return
                }
                defer { 
                    if let stmt {
                        sqlite3_finalize(stmt)
                    }
                }

                // If we get a row, the word exists.
                let stepResult = sqlite3_step(stmt)
                let result: Bool
                switch stepResult {
                case SQLITE_ROW:
                    result = true
                case SQLITE_DONE:
                    result = false
                default:
                    let msg = String(cString: sqlite3_errmsg(db))
                    Log.error("CEDICT: contains() step failed: \(msg)")
                    result = false
                }
                
                continuation.resume(returning: result)
            }
        }
    }

    /// Prepare a statement and bind `word` to parameter 1 (?1).
    @discardableResult
    private func prepareAndBind(
        _ sql: String,
        word: String,
        stmt: inout OpaquePointer?
    ) -> Bool {
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
            let msg = String(cString: sqlite3_errmsg(db))
            Log.error("CEDICT: prepare failed: \(msg)")
            return false
        }

        guard let cString = (word as NSString).utf8String else {
            Log.error("CEDICT: failed to get UTF-8 for word: \(word)")
            return false
        }

        sqlite3_bind_text(stmt, 1, cString, -1, SQLITE_TRANSIENT)
        return true
    }

    // MARK: - DB open
    private func openDatabaseIfNeeded() {
        guard db == nil else { return }

        guard
            let url = Bundle.main.url(
                forResource: dbFileName,
                withExtension: dbFileExtension
            )
        else {
            Log.error("CEDICT: could not find cedict DB in bundle")
            return
        }

        var handle: OpaquePointer?
        let rc = sqlite3_open_v2(url.path, &handle, SQLITE_OPEN_READONLY, nil)
        if rc != SQLITE_OK {
            let msg = String(cString: sqlite3_errmsg(handle))
            Log.error("CEDICT: failed to open DB: \(msg)")
            if let handle { sqlite3_close(handle) }
            return
        }

        db = handle
    }

    // MARK: - Sense Parsing
    
    /// Parses raw sense string from CEDICT into structured Sense objects.
    ///
    /// CEDICT Format:
    /// Senses are slash-delimited: "sense1/sense2/sense3"
    /// Each sense contains semicolon-delimited glosses: "gloss1; gloss2; gloss3"
    ///
    /// Special Handling:
    /// - Classifier senses prefixed with "CL:" (e.g., "CL:個|个[ge4]")
    /// - Glosses may contain embedded Chinese with pinyin
    ///
    /// Example input: "to like; to love/CL:個|个[ge4]"
    /// Result: 2 senses, first with 2 glosses, second is classifier
    ///
    /// - Parameter raw: Raw sense string from CEDICT database
    /// - Returns: Array of structured Sense objects with parsed glosses
    private static func parseSenses(from raw: String) -> [Sense] {
        // raw string = "sense1/sense2/sense3", where each sense = "gloss1; gloss2"
        let senseStrings =
            raw
            .split(separator: "/")
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }

        return senseStrings.map { senseStr in
            // Classifier Detection
            // CEDICT marks classifiers with "CL:" prefix
            // Example: "CL:個|个[ge4]" means this word is measured with 個/个
            var isClassifier = false
            var strippedSenseStr = senseStr
            if senseStr.hasPrefix("CL:") {
                isClassifier = true
                strippedSenseStr = String(senseStr.dropFirst(3))  // Remove "CL:" prefix
            }

            let glossStrings =
                strippedSenseStr
                .split(separator: ";")
                .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                .filter { !$0.isEmpty }

            let glosses: [Gloss] = glossStrings.map { glossStr in
                parseGloss(String(glossStr))
            }

            return Sense(glosses: glosses, isClassifier: isClassifier)
        }
    }

    /// Parses a single gloss string into structured fragments.
    ///
    /// Fragment Types:
    /// 1. Plain text: Regular English definition text
    /// 2. Linked headwords: Chinese words with optional trad|simp variants
    /// 3. Pinyin: Pronunciation guide in brackets
    ///
    /// CEDICT Embedded Format Examples:
    /// - `父親|父亲[fu4 qin5]` → Link("父親", "父亲") + Pinyin(["fù", "qīn"])
    /// - `see 東西|东西[dong1 xi5]` → Text("see ") + Link + Pinyin
    /// - `[dong1 xi5]` → Pinyin only (no headword)
    ///
    /// Regex Pattern Explanation:
    /// ```
    /// Pattern A: ([\p{Han}]+)(?:\|([\p{Han}]+))?\[([A-Za-z0-9 ]+)\]
    ///            └─ head1 ─┘  └──── head2? ────┘ └─── pinyin ───┘
    ///            Matches: 東西|东西[dong1 xi5] or 我[wo3]
    ///
    /// Pattern B: \[([A-Za-z0-9 ]+)\]
    ///            └──── pinyin ────┘
    ///            Matches: [fu4 qin5] (bare pinyin, no Chinese)
    /// ```
    ///
    /// - Parameter raw: Raw gloss string from CEDICT
    /// - Returns: Gloss with structured fragments (text, links, pinyin)
    private static func parseGloss(_ raw: String) -> Gloss {
        // Trim outer whitespace once
        let text = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        if text.isEmpty {
            return Gloss(fragments: [])
        }

        // Regex Pattern for Embedded Chinese + Pinyin
        // Matches two cases:
        //   a) Chinese headword(s) + pinyin: 東西|东西[dong1 xi5]
        //   b) Bare pinyin only: [dong1 xi5]
        let pattern =
            #"([\p{Han}]+)(?:\|([\p{Han}]+))?\[([A-Za-z0-9 ]+)\]|\[([A-Za-z0-9 ]+)\]"#

        guard
            let regex = try? NSRegularExpression(pattern: pattern, options: [])
        else {
            return Gloss(
                fragments: [.text(text)]
            )
        }

        var fragments: [GlossFragment] = []
        var currentLocation = text.startIndex

        let nsText = text as NSString
        let matches = regex.matches(
            in: text,
            range: NSRange(location: 0, length: nsText.length)
        )

        func appendNormalizedTextFragment(_ substring: Substring) {
            let raw = String(substring)
            let normalized = raw.trimmingCharacters(in: .whitespacesAndNewlines)
            if !normalized.isEmpty {
                fragments.append(.text(normalized))
            }
        }

        for match in matches {
            guard let range = Range(match.range, in: text) else {
                continue
            }

            // Plain text before match
            // Any text between matches becomes a text fragment
            if currentLocation < range.lowerBound {
                appendNormalizedTextFragment(
                    text[currentLocation..<range.lowerBound]
                )
            }

            // Case A: Headword + Pinyin
            // Examples: 件[jian4], 樁|桩[zhuang1]
            if let head1Range = Range(match.range(at: 1), in: text),
                let pinyinWithHeadRange = Range(match.range(at: 3), in: text)
            {
                let head1 = String(text[head1Range])
                var trad = head1
                var simp = head1

                // Check for traditional|simplified variant
                if let head2Range = Range(match.range(at: 2), in: text) {
                    let head2 = String(text[head2Range])
                    // CC-CEDICT Convention: First is traditional, second is simplified
                    trad = head1
                    simp = head2
                }

                // Create clickable link for cross-reference
                let headword = LinkedHeadword(
                    traditional: trad,
                    simplified: simp
                )
                fragments.append(.link(headword))

                // Convert numbered pinyin (dong1 xi5) to accented (dōng xī)
                let numberedPinyin = String(text[pinyinWithHeadRange])
                let accented: [String] =
                    numberedPinyin
                    .split(separator: " ")
                    .map { Self.numberedToAccentedPinyin(String($0)) }

                if !accented.isEmpty {
                    fragments.append(.accentedPinyin(accented))
                }
            }
            // Case B: Bare Pinyin
            // Example: [fu4 qin5] with no Chinese characters
            else if let barePinyinRange = Range(match.range(at: 4), in: text) {
                let numberedPinyin = String(text[barePinyinRange])
                let accented: [String] =
                    numberedPinyin
                    .split(separator: " ")
                    .map { Self.numberedToAccentedPinyin(String($0)) }

                if !accented.isEmpty {
                    fragments.append(.accentedPinyin(accented))
                }
            }

            currentLocation = range.upperBound
        }

        // Trailing text after last match
        if currentLocation < text.endIndex {
            appendNormalizedTextFragment(text[currentLocation..<text.endIndex])
        }

        // Fallback for unparseable glosses
        // If no patterns matched, treat entire string as plain text
        if fragments.isEmpty {
            fragments = [.text(text)]
        }

        return Gloss(fragments: fragments)
    }

    // MARK: - Pinyin Tone Conversion
    
    /// Converts numbered pinyin to accented pinyin with proper Unicode tone marks.
    ///
    /// # Pinyin Tone System
    ///
    /// Chinese has 4 tones plus a neutral tone (5), indicated by numbers in CEDICT:
    /// 1. First tone (¯): High level - mā (mother)
    /// 2. Second tone (´): Rising - má (hemp)
    /// 3. Third tone (ˇ): Falling-rising - mǎ (horse)
    /// 4. Fourth tone (`): Falling - mà (scold)
    /// 5. Neutral tone: No mark - ma (question particle)
    ///
    /// # Tone Mark Placement Rules
    ///
    /// The tone mark goes on the main vowel, following these priority rules:
    ///
    /// 1. 'a' always wins if present: bai → bǎi (third tone on 'a')
    /// 2. 'e' is second if no 'a': mei → měi (third tone on 'e')
    /// 3. 'ou' special case: tone goes on 'o': gou → gǒu (third tone on 'o')
    /// 4. Otherwise last vowel: gui → guǐ (third tone on 'i', the last vowel)
    ///
    /// # Character Normalization
    ///
    /// CEDICT uses various representations for ü:
    /// - `u:` → `ü` (e.g., `lu:4` → `lǜ`)
    /// - `v` → `ü` (e.g., `lv4` → `lǜ`)
    ///
    /// # Examples
    ///
    /// ```
    /// "ni3"     → "nǐ"      (tone 3 on 'i', last vowel)
    /// "hao3"    → "hǎo"     (tone 3 on 'a', highest priority)
    /// "mei3"    → "měi"     (tone 3 on 'e', no 'a' present)
    /// "gou3"    → "gǒu"     (tone 3 on 'o', special 'ou' case)
    /// "lu:4"    → "lǜ"      (tone 4 on normalized 'ü')
    /// "ma5"     → "ma"      (neutral tone, no mark)
    /// ```
    ///
    /// - Parameter numbered: Pinyin with trailing tone number (1-5)
    /// - Returns: Pinyin with Unicode tone mark, or original if malformed
    private static func numberedToAccentedPinyin(_ numbered: String) -> String {
        // Step 1: Extract and validate tone number
        // Tone must be 1-5 and at end of string
        guard let toneChar = numbered.last,
            let tone = Int(String(toneChar)),
            tone >= 1 && tone <= 5
        else {
            // Not numbered pinyin (already accented or invalid format)
            return numbered
        }

        let base = String(numbered.dropLast())

        // Step 2: Normalize ü character representations
        // CEDICT uses both 'u:' and 'v' to represent ü
        // Examples: nu:3 → nǚ, nv3 → nǚ
        let normalized =
            base
            .replacingOccurrences(of: "u:", with: "ü")
            .replacingOccurrences(of: "v", with: "ü")

        // Step 3: Determine which vowel receives the tone mark
        // Chinese pinyin tone placement follows strict rules
        let vowels = ["a", "e", "o", "u", "i", "ü"]
        var targetIndex: String.Index? = nil

        // Priority Rule 1: 'a' always gets the mark
        // Example: bai3 → bǎi (not baǐ)
        if let i = normalized.firstIndex(of: "a") {
            targetIndex = i
        }
        // Priority Rule 2: 'e' is second priority
        // Example: mei3 → měi (not meǐ)
        else if let i = normalized.firstIndex(of: "e") {
            targetIndex = i
        }
        // Priority Rule 3: 'ou' diphthong gets mark on 'o'
        // Example: gou3 → gǒu (not goǔ)
        else if normalized.contains("ou"),
            let i = normalized.firstIndex(of: "o")
        {
            targetIndex = i
        }
        // Priority Rule 4: Otherwise, last vowel wins
        // Example: gui3 → guǐ ('i' is last vowel)
        else {
            targetIndex = normalized.lastIndex(where: {
                vowels.contains(String($0))
            })
        }

        // Validate we found a vowel
        guard let idx = targetIndex else {
            // No vowels found (malformed pinyin)
            return normalized
        }

        let vowel = normalized[idx]

        // Step 4: Map vowel + tone to Unicode character
        // Each vowel has 5 forms: [tone1, tone2, tone3, tone4, neutral]
        // Using Unicode combining marks for tone accents
        let toneMarks: [Character: [Character]] = [
            "a": ["ā", "á", "ǎ", "à", "a"],  // U+0101, U+00E1, U+01CE, U+00E0
            "e": ["ē", "é", "ě", "è", "e"],  // U+0113, U+00E9, U+011B, U+00E8
            "i": ["ī", "í", "ǐ", "ì", "i"],  // U+012B, U+00ED, U+01D0, U+00EC
            "o": ["ō", "ó", "ǒ", "ò", "o"],  // U+014D, U+00F3, U+01D2, U+00F2
            "u": ["ū", "ú", "ǔ", "ù", "u"],  // U+016B, U+00FA, U+01D4, U+00F9
            "ü": ["ǖ", "ǘ", "ǚ", "ǜ", "ü"],  // U+01D6, U+01D8, U+01DA, U+01DC
        ]

        // Get accented character (tone 5 = neutral = no mark)
        let accented = toneMarks[vowel]?[tone - 1] ?? vowel

        // Step 5: Replace original vowel with accented version
        var result = normalized
        result.replaceSubrange(idx...idx, with: String(accented))

        return result
    }
}
