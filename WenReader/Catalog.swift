//
//  Catalog.swift
//  ChineseReader
//
//  Created by Oliver Zhang on 2025-11-08.
//

import Combine
import Foundation
import SwiftUI
import UniformTypeIdentifiers
import UIKit

// MARK: - Models
struct BookItem: Identifiable, Codable, Hashable {
    let id: UUID
    var title: String?
    var authors: [String]
    /// Unique ID for this book, such as ISBN
    var canonicalID: String?
    /// File name of the local copy inside Application Support/Books (sandbox)
    var bookFileName: String
    /// File name of the saved cover image inside sandbox
    var coverFileName: String?
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
        Task.detached { [weak self] in
            guard let self else { return }

            let accessed = url.startAccessingSecurityScopedResource()
            defer { if accessed { url.stopAccessingSecurityScopedResource() } }

            do {
                let id = UUID()
                let dest = FileManager.appSupportBooksDir
                    .appendingPathComponent("\(id.uuidString).epub")
                
                // Load metadata (title, authors, cover image) from the copied EPUB
                let metadata = await EpubMetadataLoader.load(from: url)
                // De-dupe by book ID.
                if await books.contains(where: { $0.canonicalID == metadata?.canonicalID }) { return }

                // Make a copy of this book in app support dir
                if !FileManager.default.fileExists(atPath: dest.path) {
                    try FileManager.default.copyItem(at: url, to: dest)
                }

                // Save cover image (if any) to disk and remember its filename
                var coverFileName: String?
                if let coverImage = metadata?.cover,
                   let data = coverImage.jpegData(compressionQuality: 0.1) {
                    let fileName = "\(id.uuidString)-cover.jpg"
                    let coverURL = FileManager.appSupportBooksDir
                        .appendingPathComponent(fileName)
                    do {
                        try data.write(to: coverURL, options: .atomic)
                        coverFileName = fileName
                    } catch {
                        Log.error("Failed to write cover image: \(error)")
                    }
                }

                let item = BookItem(
                    id: id,
                    title: metadata?.title,
                    authors: metadata?.authors ?? [],
                    canonicalID: metadata?.canonicalID,
                    bookFileName: dest.lastPathComponent,
                    coverFileName: coverFileName,
                )

                await MainActor.run {
                    self.books.insert(item, at: 0)
                }
            } catch {
                Log.error("Import error: \(error)")
            }
        }
    }
    
    func update(_ book: BookItem) {
        guard let idx = books.firstIndex(where: { $0.id == book.id }) else { return }
        books[idx] = book
    }

    /// Delete from catalog and the sandboxed copy that was created on import.
    func remove(_ book: BookItem) {
        // Delete EPUB file
        let local = localURL(for: book)
        try? FileManager.default.removeItem(at: local)

        // Delete cover file, if present
        if let coverURL = coverURL(for: book) {
            try? FileManager.default.removeItem(at: coverURL)
        }

        guard let idx = books.firstIndex(of: book) else { return }
        books.remove(at: idx)
    }

    func localURL(for book: BookItem) -> URL {
        FileManager.appSupportBooksDir.appendingPathComponent(book.bookFileName)
    }

    /// Convenience: URL for cover image on disk
    func coverURL(for book: BookItem) -> URL? {
        guard let name = book.coverFileName else { return nil }
        return FileManager.appSupportBooksDir.appendingPathComponent(name)
    }

    /// Convenience: load a UIImage for a book's cover (simple, sync)
    func coverImage(for book: BookItem) -> UIImage? {
        guard let url = coverURL(for: book) else { return nil }
        return UIImage(contentsOfFile: url.path)
    }

    private func persist() {
        Defaults.setCodable(books, forKey: storageKey)
    }

    private func restore() {
        books = Defaults.codable(
            [BookItem].self,
            forKey: storageKey,
            default: []
        )
    }
}

// MARK: - Persistence Helpers
enum Defaults {
    static func setCodable<T: Codable>(_ value: T, forKey key: String) {
        if let data = try? JSONEncoder().encode(value) {
            UserDefaults.standard.set(data, forKey: key)
        }
    }
    static func codable<T: Codable>(
        _ type: T.Type,
        forKey key: String,
        default def: T
    ) -> T {
        guard let data = UserDefaults.standard.data(forKey: key),
              let decoded = try? JSONDecoder().decode(type, from: data)
        else { return def }
        return decoded
    }
}

// MARK: - Utilities
enum Log {
    nonisolated static func info(_ msg: String) { print("I:  \(msg)") }
    nonisolated static func error(_ msg: String) { print("E: \(msg)") }
}

extension FileManager {
    nonisolated static var appSupportBooksDir: URL {
        let base = FileManager.default.urls(
            for: .applicationSupportDirectory,
            in: .userDomainMask
        )[0]
        let dir = base.appendingPathComponent("Books", isDirectory: true)
        try? FileManager.default.createDirectory(
            at: dir,
            withIntermediateDirectories: true
        )
        return dir
    }
}

extension UTType {
    static let epub = UTType(importedAs: "org.idpf.epub-container")
}
