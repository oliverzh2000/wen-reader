//
//  LibraryUI.swift
//  ChineseReader
//
//  Created by Oliver Zhang on 2025-11-08.
//

import SwiftUI
import UniformTypeIdentifiers

struct LibraryView: View {
    @EnvironmentObject private var catalog: CatalogStore
    @State private var showImporter = false
    @State private var importError: String?
    
    @State private var renamingBook: BookItem?
    @State private var newTitle: String = ""

    var body: some View {
        List {
            Section("Imported EPUBs") {
                if catalog.books.isEmpty {
                    ContentUnavailableView(
                        "No books yet",
                        systemImage: "book",
                        description: Text("Tap the '+' button to add books.")
                    )
                } else {
                    ForEach(catalog.books) { book in
                        NavigationLink(value: book) {
                            LibraryRow(
                                book: book,
                                coverImage: catalog.coverImage(for: book),
                                onRename: {
                                    renamingBook = book
                                    newTitle = book.title ?? ""
                                },
                                onDelete: {
                                    catalog.remove(book)
                                }
                            )
                            .alert("Rename Book", isPresented: Binding(
                                get: { renamingBook != nil },
                                set: { if !$0 { renamingBook = nil } }
                            )) {
                                TextField("Title", text: $newTitle)

                                Button("Save") {
                                    if var book = renamingBook {
                                        book.title = newTitle.trimmingCharacters(in: .whitespacesAndNewlines)
                                        catalog.update(book)
                                    }
                                    renamingBook = nil
                                }

                                Button("Cancel", role: .cancel) {
                                    renamingBook = nil
                                }
                            } message: {
                                Text("You can override the original EPUB title with your own.")
                            }
                        }
                    }
                }
            }
        }
        .navigationTitle("Library")
        .toolbar {
            ToolbarItem(placement: .bottomBar) {
                Button {
                    showImporter = true
                } label: {
                    Label("Add", systemImage: "plus")
                }
            }
        }
        .navigationDestination(for: BookItem.self) { book in
            ReaderView(book: book)
        }
        .fileImporter(
            isPresented: $showImporter,
            allowedContentTypes: [.epub],
            allowsMultipleSelection: false
        ) { result in
            switch result {
            case .success(let urls):
                guard let first = urls.first else { return }
                withAnimation { catalog.add(url: first) }
            case .failure(let err):
                importError = err.localizedDescription
            }
        }
        .alert(
            "Import failed",
            isPresented: Binding(
                get: { importError != nil },
                set: { _ in importError = nil }
            )
        ) {
            Button("OK", role: .cancel) {}
        } message: {
            Text(importError ?? "Unknown error")
        }
        .scrollBounceBehavior(.basedOnSize)
    }
}

private struct LibraryRow: View {
    let book: BookItem
    let coverImage: UIImage?
    let onRename: () -> Void
    let onDelete: () -> Void
    
    @ScaledMetric(relativeTo: .largeTitle)
    private var iconHeight: CGFloat = 70

    var body: some View {
        HStack(spacing: 12) {
            Group {
                if let coverImage {
                    Image(uiImage: coverImage)
                        .resizable()
                        .scaledToFit()
                        .clipShape(RoundedRectangle(cornerRadius: 4, style: .continuous))
                } else {
                    Image(systemName: "book.closed")
                        .resizable()
                        .scaledToFit()
                }
            }
            .frame(width: iconHeight * 0.8, height: iconHeight)
            
            VStack(alignment: .leading) {
                Text(book.title ?? "No Title").font(.headline)
                Text(book.authors.isEmpty ? "Unknown Author" : book.authors.joined(separator: ", ")).font(.caption).foregroundStyle(.secondary)
            }
            
            Spacer()

            // Trailing menu for row actions
            Menu {
                Button {
                    onRename()
                } label: {
                    Label("Rename Title", systemImage: "pencil")
                }

                Button(role: .destructive) {
                    onDelete()
                } label: {
                    Label("Delete Book", systemImage: "trash")
                }
            } label: {
                Image(systemName: "ellipsis.circle")
            }
            .buttonStyle(.borderless)        }
    }
}
