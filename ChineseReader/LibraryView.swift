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

    var body: some View {
        List {
            Section("Imported Books") {
                if catalog.books.isEmpty {
                    ContentUnavailableView(
                        "No books yet",
                        systemImage: "book",
                        description: Text("Tap the '+' button to add books.")
                    )
                } else {
                    ForEach(catalog.books) { book in
                        NavigationLink(value: book) {
                            LibraryRow(book: book) {
                                withAnimation { catalog.remove(book) }
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
    let onDelete: () -> Void

    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: "book.closed")
            VStack(alignment: .leading) {
                Text(book.displayName).font(.headline)
                Text("EPUB").font(.caption).foregroundStyle(.secondary)
            }
            Spacer()
            Button(role: .destructive, action: onDelete) {
                Image(systemName: "trash")
            }
            .buttonStyle(.borderless)
        }
    }
}
