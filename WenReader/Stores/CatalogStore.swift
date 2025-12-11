//
//  CatalogStore.swift
//  WenReader
//
//  Created by Oliver Zhang on 2025-11-08.
//  Refactored on 2025-12-10.
//

import Combine
import Foundation
import SwiftUI
import UIKit

/// Catalog Store (UI-facing; delegates to repository for data operations)
@MainActor
final class CatalogStore: ObservableObject {
    @Published private(set) var books: [BookItem] = []
    @Published var lastError: AppError?
    
    private let repository: BookRepository

    init(repository: BookRepository = DefaultBookRepository()) {
        self.repository = repository
        Task {
            await loadBooks()
        }
    }
    
    // MARK: - Public API

    /// Import by copying the selected EPUB into our sandbox
    func add(url: URL) {
        Task {
            do {
                let book = try await repository.importBook(from: url)
                try await repository.saveBook(book)
                await loadBooks()
            } catch let error as AppError {
                lastError = error
            } catch {
                lastError = .bookImportFailed(reason: error.localizedDescription)
            }
        }
    }
    
    func update(_ book: BookItem) {
        Task {
            do {
                try await repository.updateBook(book)
                await loadBooks()
            } catch let error as AppError {
                lastError = error
            } catch {
                lastError = .bookImportFailed(reason: "Failed to update book")
            }
        }
    }

    /// Delete from catalog and the sandboxed copy
    func remove(_ book: BookItem) {
        Task {
            do {
                try await repository.deleteBook(book)
                await loadBooks()
            } catch {
                lastError = .fileOperationFailed(operation: "delete book", underlying: error)
            }
        }
    }

    func localURL(for book: BookItem) -> URL {
        repository.localURL(for: book)
    }

    /// Convenience: URL for cover image on disk
    func coverURL(for book: BookItem) -> URL? {
        repository.coverURL(for: book)
    }

    /// Convenience: load a UIImage for a book's cover
    func coverImage(for book: BookItem) -> UIImage? {
        repository.coverImage(for: book)
    }
    
    // MARK: - Private
    
    private func loadBooks() async {
        books = await repository.loadBooks()
    }
}
