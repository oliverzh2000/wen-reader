//
//  BookItem.swift
//  WenReader
//
//  Created by Oliver Zhang on 2025-12-08.
//

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
}
