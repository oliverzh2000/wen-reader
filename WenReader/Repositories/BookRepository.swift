// Copyright 2025 Oliver Zhang
// Licensed under the MIT License

import Foundation
import UIKit

// MARK: - Protocol

/// Protocol defining book data operations
protocol BookRepository {
    /// Load all books from persistent storage
    func loadBooks() async -> [BookItem]
    
    /// Save a book to persistent storage
    func saveBook(_ book: BookItem) async throws
    
    /// Update an existing book
    func updateBook(_ book: BookItem) async throws
    
    /// Delete a book from persistent storage
    func deleteBook(_ book: BookItem) async throws
    
    /// Import a book from an external URL
    func importBook(from url: URL) async throws -> BookItem
    
    /// Get the local file URL for a book
    func localURL(for book: BookItem) -> URL
    
    /// Get the cover image URL for a book
    func coverURL(for book: BookItem) -> URL?
    
    /// Load cover image for a book
    func coverImage(for book: BookItem) -> UIImage?
}

/// Default implementation of BookRepository
final class DefaultBookRepository: BookRepository {
    private let storageKey = "catalog.books.v1"
    private let fileManager = FileManager.default
    
    // MARK: - Public API
    
    func loadBooks() async -> [BookItem] {
        return UserDefaults.standard.codable(
            [BookItem].self,
            forKey: storageKey,
            default: []
        )
    }
    
    func saveBook(_ book: BookItem) async throws {
        var books = await loadBooks()
        
        // Check for duplicates
        if books.contains(where: { $0.id == book.id }) {
            throw AppError.bookImportFailed(reason: "Book already exists")
        }
        
        books.insert(book, at: 0)
        UserDefaults.standard.setCodable(books, forKey: storageKey)
    }
    
    func updateBook(_ book: BookItem) async throws {
        var books = await loadBooks()
        
        guard let index = books.firstIndex(where: { $0.id == book.id }) else {
            throw AppError.bookImportFailed(reason: "Book not found")
        }
        
        books[index] = book
        UserDefaults.standard.setCodable(books, forKey: storageKey)
    }
    
    func deleteBook(_ book: BookItem) async throws {
        // Delete EPUB file
        let epubURL = localURL(for: book)
        do {
            try fileManager.removeItem(at: epubURL)
        } catch {
            Log.error("Failed to delete EPUB file for '\(book.title ?? "Unknown")': \(error)")
            // Non-fatal: continue with catalog removal even if file deletion fails
        }
        
        // Delete cover file if present
        if let coverFileURL = coverURL(for: book) {
            do {
                try fileManager.removeItem(at: coverFileURL)
            } catch {
                Log.error("Failed to delete cover file for '\(book.title ?? "Unknown")': \(error)")
                // Non-fatal: continue with catalog removal
            }
        }
        
        // Remove from storage
        var books = await loadBooks()
        books.removeAll { $0.id == book.id }
        UserDefaults.standard.setCodable(books, forKey: storageKey)
    }
    
    func importBook(from url: URL) async throws -> BookItem {
        let accessed = url.startAccessingSecurityScopedResource()
        defer { if accessed { url.stopAccessingSecurityScopedResource() } }
        
        // Load metadata from EPUB (non-fatal - use defaults if it fails)
        let metadata = await EpubMetadataLoader.load(from: url)
        if metadata == nil {
            Log.info("Failed to load metadata from '\(url.lastPathComponent)', using fallback values")
        }
        
        // Use filename as fallback title if metadata loading failed
        let fallbackTitle = url.deletingPathExtension().lastPathComponent
        
        // Check for duplicates by canonical ID (only if we have one)
        if let canonicalID = metadata?.canonicalID {
            let existingBooks = await loadBooks()
            if existingBooks.contains(where: { $0.canonicalID == canonicalID }) {
                throw AppError.bookImportFailed(reason: "This book has already been imported")
            }
        }
        
        // Generate ID and destination path
        let id = UUID()
        let dest = FileManager.appSupportBooksDir
            .appendingPathComponent("\(id.uuidString).epub")
        
        // Copy EPUB to app support directory
        if !fileManager.fileExists(atPath: dest.path) {
            try fileManager.copyItem(at: url, to: dest)
        }
        
        // Save cover image if available
        var coverFileName: String?
        if let coverImage = metadata?.cover,
           let data = coverImage.jpegData(compressionQuality: ReaderConstants.Image.coverCompressionQuality) {
            let fileName = "\(id.uuidString)-cover.jpg"
            let coverFileURL = FileManager.appSupportBooksDir
                .appendingPathComponent(fileName)
            do {
                try data.write(to: coverFileURL, options: .atomic)
                coverFileName = fileName
            } catch {
                // Non-fatal: continue without cover
                Log.error("Failed to write cover image: \(error)")
            }
        }
        
        // Create book item with metadata or fallback values
        let book = BookItem(
            id: id,
            title: metadata?.title ?? fallbackTitle,
            authors: metadata?.authors ?? [],
            canonicalID: metadata?.canonicalID,
            bookFileName: dest.lastPathComponent,
            coverFileName: coverFileName
        )
        
        return book
    }
    
    func localURL(for book: BookItem) -> URL {
        FileManager.appSupportBooksDir.appendingPathComponent(book.bookFileName)
    }
    
    func coverURL(for book: BookItem) -> URL? {
        guard let fileName = book.coverFileName else { return nil }
        return FileManager.appSupportBooksDir.appendingPathComponent(fileName)
    }
    
    func coverImage(for book: BookItem) -> UIImage? {
        guard let url = coverURL(for: book) else { return nil }
        return UIImage(contentsOfFile: url.path)
    }
}
