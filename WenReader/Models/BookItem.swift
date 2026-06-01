// Copyright 2025 Oliver Zhang
// Licensed under the MIT License

import Foundation

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
    /// Total number of Readium positions (stable page-equivalent count)
    var pageCount: Int?
}
