// Copyright 2025 Oliver Zhang
// Licensed under the MIT License

import Foundation
import SQLite3

/// In-memory prefix trie of CEDICT simplified forms for fast span enumeration.
///
/// Built once at app startup from the SQLite database. Provides
/// `getWordsAt(text:position:)` to enumerate all CEDICT entries starting
/// at a given character position — needed by the span scorer to generate
/// candidate spans for scoring.
///
/// The trie stores ~120K simplified forms. Build time is <100ms on modern
/// iOS devices. Memory footprint is modest since Chinese words are short
/// (1–6 characters) and share many prefixes.
final class CedictTrie {

    /// Each node maps a Character to its child node.
    /// `isTerminal` marks the end of a valid CEDICT word.
    private final class Node {
        var children: [Character: Node] = [:]
        var isTerminal: Bool = false
    }

    private let root = Node()

    /// Total number of words inserted.
    private(set) var wordCount: Int = 0

    // MARK: - Building

    /// Insert a single word into the trie.
    func insert(_ word: String) {
        var node = root
        for ch in word {
            if let child = node.children[ch] {
                node = child
            } else {
                let child = Node()
                node.children[ch] = child
                node = child
            }
        }
        if !node.isTerminal {
            node.isTerminal = true
            wordCount += 1
        }
    }

    /// Check if a word exists in the trie.
    func contains(_ word: String) -> Bool {
        var node = root
        for ch in word {
            guard let child = node.children[ch] else { return false }
            node = child
        }
        return node.isTerminal
    }

    // MARK: - Span Enumeration

    /// Return all CEDICT words starting at `position` in `text`.
    ///
    /// Walks the trie character by character from the given position,
    /// collecting every terminal node encountered. Returns words in
    /// order of increasing length.
    ///
    /// - Parameters:
    ///   - text: The full sentence as an array of Characters (for O(1) indexing).
    ///   - position: The starting character index.
    /// - Returns: All CEDICT words starting at that position.
    func getWordsAt(_ chars: [Character], position: Int) -> [String] {
        var words: [String] = []
        var node = root
        for i in position..<chars.count {
            guard let child = node.children[chars[i]] else { break }
            node = child
            if node.isTerminal {
                words.append(String(chars[position...i]))
            }
        }
        return words
    }

    // MARK: - Factory

    /// Build a trie from the CEDICT SQLite database.
    ///
    /// Queries all simplified forms from the entries table and inserts
    /// them into the trie. Returns nil if the database cannot be opened.
    static func fromDatabase(
        resourceName: String = "cedict",
        resourceExtension: String = "sqlite"
    ) -> CedictTrie? {
        guard let url = Bundle.main.url(
            forResource: resourceName,
            withExtension: resourceExtension
        ) else {
            Log.error("CedictTrie: could not find \(resourceName).\(resourceExtension) in bundle")
            return nil
        }

        var db: OpaquePointer?
        guard sqlite3_open_v2(url.path, &db, SQLITE_OPEN_READONLY, nil) == SQLITE_OK else {
            let msg = db.map { String(cString: sqlite3_errmsg($0)) } ?? "unknown"
            Log.error("CedictTrie: failed to open DB: \(msg)")
            if let db { sqlite3_close(db) }
            return nil
        }
        defer { sqlite3_close(db) }

        let sql = "SELECT DISTINCT simplified FROM entries;"
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
            let msg = String(cString: sqlite3_errmsg(db))
            Log.error("CedictTrie: prepare failed: \(msg)")
            return nil
        }
        defer { sqlite3_finalize(stmt) }

        let trie = CedictTrie()
        while sqlite3_step(stmt) == SQLITE_ROW {
            guard let cStr = sqlite3_column_text(stmt, 0) else { continue }
            trie.insert(String(cString: cStr))
        }

        Log.info("CedictTrie: built with \(trie.wordCount) words")
        return trie
    }
}
