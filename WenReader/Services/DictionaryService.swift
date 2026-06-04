// Copyright 2025 Oliver Zhang
// Licensed under the MIT License

import Foundation
import SQLite3

protocol DictionaryService {
    /// Return the full dictionary entry (all senses) for a word, if present.
    /// When `sentence` is provided and WSD is available, senses are sorted by contextual relevance.
    /// `wordOffsetInSentence` disambiguates repeated occurrences of the same word.
    func lookup(_ word: String, sentence: String?, wordOffsetInSentence: Int) async -> DictionaryResult?

    /// Return true if `word` exists in the dictionary (as trad or simp), false otherwise.
    func contains(_ word: String) async -> Bool
}

/// Convenience extension preserving the original single-argument lookup signature.
extension DictionaryService {
    func lookup(_ word: String) async -> DictionaryResult? {
        await lookup(word, sentence: nil, wordOffsetInSentence: 0)
    }
    
    func lookup(_ word: String, sentence: String?) async -> DictionaryResult? {
        await lookup(word, sentence: sentence, wordOffsetInSentence: 0)
    }
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

    /// Optional WSD service for sense ranking. Set after init when WSD model is available.
    var wsdService: WSDService?

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

    func lookup(_ word: String, sentence: String?, wordOffsetInSentence: Int) async -> DictionaryResult? {
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
                    SELECT e.id, e.traditional, e.simplified, e.pinyin,
                           sc.id as cluster_id, sc.is_trivial, sc.embedding,
                           s.id as sense_id, s.is_classifier,
                           g.gloss_text
                    FROM entries e
                    JOIN sense_clusters sc ON sc.entry_id = e.id
                    JOIN senses s ON s.sense_cluster_id = sc.id
                    JOIN glosses g ON g.sense_id = s.id
                    WHERE e.simplified = ?1 OR e.traditional = ?1
                    ORDER BY e.id, sc.id, s.rowid, g.rowid;
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

                // Collect flat rows and group into Entry → Cluster → Sense → Gloss hierarchy
                var entries: [Entry] = []
                var clusterEmbeddings: [(clusterId: Int, isTrivial: Bool, embedding: Data?)] = []
                // Maps each sense (by flat index across all entries) to its cluster ID
                var senseClusterIds: [Int] = []

                // Grouping state
                var currentEntryId: Int64 = -1
                var currentClusterId: Int64 = -1
                var currentSenseId: Int64 = -1
                var currentTrad = ""
                var currentSimp = ""
                var currentPinyin = ""
                var currentSenses: [Sense] = []
                var currentGlosses: [Gloss] = []
                var currentIsClassifier = false
                var currentSenseClusterId: Int = 0

                func flushGlossesIntoSense() {
                    guard !currentGlosses.isEmpty else { return }
                    currentSenses.append(Sense(glosses: currentGlosses, isClassifier: currentIsClassifier))
                    currentGlosses = []
                }

                func flushSensesIntoEntry() {
                    flushGlossesIntoSense()
                    guard !currentSenses.isEmpty else { return }
                    let accentedPinyin: [String] =
                        currentPinyin
                        .split(separator: " ")
                        .map { Self.numberedToAccentedPinyin(String($0)) }
                    entries.append(Entry(
                        traditional: currentTrad,
                        simplified: currentSimp,
                        accentedPinyin: accentedPinyin,
                        senses: currentSenses
                    ))
                    currentSenses = []
                }

                while sqlite3_step(stmt) == SQLITE_ROW {
                    let entryId = sqlite3_column_int64(stmt, 0)

                    guard
                        let tradC = sqlite3_column_text(stmt, 1),
                        let simpC = sqlite3_column_text(stmt, 2),
                        let pinyinC = sqlite3_column_text(stmt, 3)
                    else { continue }

                    let clusterId = sqlite3_column_int64(stmt, 4)
                    let isTrivial = sqlite3_column_int(stmt, 5) != 0

                    // Read cluster embedding BLOB (nullable)
                    var embeddingData: Data? = nil
                    if sqlite3_column_type(stmt, 6) == SQLITE_BLOB,
                       let blobPtr = sqlite3_column_blob(stmt, 6) {
                        let blobLen = sqlite3_column_bytes(stmt, 6)
                        embeddingData = Data(bytes: blobPtr, count: Int(blobLen))
                    }

                    let senseId = sqlite3_column_int64(stmt, 7)
                    let isClassifier = sqlite3_column_int(stmt, 8) != 0

                    let glossText: String
                    if let glossC = sqlite3_column_text(stmt, 9) {
                        glossText = String(cString: glossC)
                    } else {
                        continue
                    }

                    // Detect entry boundary
                    if entryId != currentEntryId {
                        flushSensesIntoEntry()
                        currentEntryId = entryId
                        currentTrad = String(cString: tradC)
                        currentSimp = String(cString: simpC)
                        currentPinyin = String(cString: pinyinC)
                        currentClusterId = -1
                        currentSenseId = -1
                    }

                    // Detect cluster boundary — track embedding per cluster
                    if clusterId != currentClusterId {
                        currentClusterId = clusterId
                        clusterEmbeddings.append((
                            clusterId: Int(clusterId),
                            isTrivial: isTrivial,
                            embedding: embeddingData
                        ))
                    }

                    // Detect sense boundary
                    if senseId != currentSenseId {
                        flushGlossesIntoSense()
                        currentSenseId = senseId
                        currentIsClassifier = isClassifier
                        currentSenseClusterId = Int(clusterId)
                        senseClusterIds.append(currentSenseClusterId)
                    }

                    // Parse gloss text through existing parseGloss() for inline references
                    let gloss = Self.parseGloss(glossText)
                    currentGlosses.append(gloss)
                }

                // Flush final entry
                flushSensesIntoEntry()

                guard !entries.isEmpty else {
                    continuation.resume(returning: nil)
                    return
                }

                // Capture values for async WSD call
                let capturedEntries = entries
                let capturedEmbeddings = clusterEmbeddings
                let capturedSenseClusterIds = senseClusterIds
                let capturedSentence = sentence
                let capturedWsd = self.wsdService
                let capturedWordOffset = wordOffsetInSentence

                // Skip WSD if there's only one non-trivial sense cluster (nothing to disambiguate)
                let nonTrivialCount = capturedEmbeddings.filter { !$0.isTrivial }.count

                // If sentence is provided and WSD is available, rank senses
                if let sentence = capturedSentence, !sentence.isEmpty, let wsd = capturedWsd, nonTrivialCount > 1 {
                    Task {
                        // Check cancellation before expensive WSD inference
                        guard !Task.isCancelled else {
                            continuation.resume(returning: DictionaryResult(entries: capturedEntries))
                            return
                        }
                        let sorted = await wsd.rankSenses(
                            word: word,
                            sentenceContext: sentence,
                            wordOffsetInSentence: capturedWordOffset,
                            entries: capturedEntries,
                            clusterEmbeddings: capturedEmbeddings,
                            senseClusterIds: capturedSenseClusterIds
                        )
                        continuation.resume(returning: DictionaryResult(entries: sorted))
                    }
                } else {
                    continuation.resume(returning: DictionaryResult(entries: capturedEntries))
                }
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
                    FROM entries
                    WHERE simplified = ?1 OR traditional = ?1
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

    // MARK: - Gloss Parsing

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
    static func parseGloss(_ raw: String) -> Gloss {
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
    static func numberedToAccentedPinyin(_ numbered: String) -> String {
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
