//
//  Core.swift
//  ChineseReader
//
//  Created by Oliver Zhang on 2025-11-08.
//

import SwiftUI
import Foundation
import Combine
import UniformTypeIdentifiers

// MARK: - Models
struct BookItem: Identifiable, Codable, Hashable {
    let id: UUID
    var displayName: String
    /// File name of the local copy inside Application Support/Books (sandbox)
    var relativePath: String
}

// MARK: - Global UI State
@MainActor
final class UiState: ObservableObject {
    @Published var hideStatusBar = false
}

// MARK: - Catalog Store (UI-facing; IO off main thread)
@MainActor
final class CatalogStore: ObservableObject {
    @Published private(set) var books: [BookItem] = [] { didSet { persist() } }
    private let storageKey = "catalog.books.v1"

    init() { restore() }

    /// Import by copying the selected EPUB into our sandbox. Avoids security-scope persistence.
    func add(url: URL) {
        Task.detached { [books] in
            let accessed = url.startAccessingSecurityScopedResource()
            defer { if accessed { url.stopAccessingSecurityScopedResource() } }

            do {
                let name = url.deletingPathExtension().lastPathComponent

                // De-dupe by display name
                if books.contains(where: { $0.displayName == name }) { return }

                let id   = UUID()
                let dest = FileManager.appSupportBooksDir.appendingPathComponent("\(id.uuidString).epub")

                if !FileManager.default.fileExists(atPath: dest.path) {
                    try FileManager.default.copyItem(at: url, to: dest)
                }

                let item = BookItem(id: id, displayName: name, relativePath: dest.lastPathComponent)

                await MainActor.run { [weak self] in
                    self?.books.insert(item, at: 0)
                }
            } catch {
                Log.error("Import error: \(error)")
            }
        }
    }

    /// Remove from catalog only (keeps sandboxed copy to honor “don’t delete from disk”).
    func remove(_ book: BookItem) {
        guard let idx = books.firstIndex(of: book) else { return }
        books.remove(at: idx)
    }

    func localURL(for book: BookItem) -> URL {
        FileManager.appSupportBooksDir.appendingPathComponent(book.relativePath)
    }

    private func persist() {
        Defaults.setCodable(books, forKey: storageKey)
    }
    private func restore() {
        books = Defaults.codable([BookItem].self, forKey: storageKey, default: [])
    }
}

// MARK: - Persistence Helpers
enum Defaults {
    static func setCodable<T: Codable>(_ value: T, forKey key: String) {
        if let data = try? JSONEncoder().encode(value) {
            UserDefaults.standard.set(data, forKey: key)
        }
    }
    static func codable<T: Codable>(_ type: T.Type, forKey key: String, default def: T) -> T {
        guard let data = UserDefaults.standard.data(forKey: key),
              let decoded = try? JSONDecoder().decode(type, from: data) else { return def }
        return decoded
    }
}

// MARK: - Utilities
enum Log {
    nonisolated static func info(_ msg: String)  { print("I:  \(msg)") }
    nonisolated static func error(_ msg: String) { print("E: \(msg)") }
}

extension FileManager {
    nonisolated static var appSupportBooksDir: URL {
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        let dir  = base.appendingPathComponent("Books", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }
}

extension UTType {
    static let epub = UTType(importedAs: "org.idpf.epub-container")
}
