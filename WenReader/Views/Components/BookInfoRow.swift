// Copyright 2025 Oliver Zhang
// Licensed under the MIT License

import SwiftUI
import UIKit

/// Reusable book display showing cover, title, author, and page count.
/// Used in both the library list and the ToC sheet header.
struct BookInfoRow: View {
    let book: BookItem
    let coverImage: UIImage?
    
    private let coverHeight: CGFloat = 70
    
    var body: some View {
        HStack(spacing: 14) {
            Group {
                if let coverImage {
                    Image(uiImage: coverImage)
                        .resizable()
                        .scaledToFit()
                } else {
                    Image(systemName: "book.closed")
                        .resizable()
                        .scaledToFit()
                        .foregroundStyle(.secondary)
                }
            }
            .frame(width: coverHeight, height: coverHeight)

            VStack(alignment: .leading) {
                Text(book.title ?? "No Title").font(.headline)
                Text(book.authors.isEmpty ? "Unknown Author" : book.authors.joined(separator: ", ")).font(.subheadline).foregroundStyle(.secondary)
                if let pageCount = book.pageCount {
                    Text("\(pageCount) pages")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
        }
    }
}
