// Copyright 2025 Oliver Zhang
// Licensed under the MIT License

import Foundation
import ReadiumShared
import SwiftUI
import UIKit

struct TableOfContentsSheet: View {
    let publication: Publication?
    let book: BookItem
    let coverImage: UIImage?
    let onSelect: (RLink) -> Void

    @State private var tocEntries: [(link: RLink, depth: Int, progress: Double?)] = []

    @Environment(\.dismiss) private var dismiss
    @EnvironmentObject var settingsStore: SettingsStore

    var body: some View {
        NavigationStack {
            Group {
                if tocEntries.isEmpty {
                    Text("No table of contents.").foregroundStyle(.secondary)
                } else {
                    List {
                        // Book cover + title header
                        Section {
                            BookInfoRow(
                                book: book,
                                coverImage: coverImage
                            )
                            .padding(.vertical)
                            .listRowSeparator(.hidden)
                        }

                        // Chapter list with indentation and progress
                        Section {
                            ForEach(Array(tocEntries.enumerated()), id: \.offset) { _, entry in
                                Button {
                                    onSelect(entry.link)
                                    dismiss()
                                } label: {
                                    HStack {
                                        Text(entry.link.title ?? entry.link.href)
                                            .lineLimit(2)
                                        Spacer()
                                        if let progress = entry.progress {
                                            Text(String(format: "%.0f%%", progress * 100))
                                                .font(.caption)
                                                .foregroundStyle(.secondary)
                                        }
                                    }
                                    .padding(.leading, CGFloat(entry.depth) * 16)
                                }
                                .tint(.primary)
                            }
                        }
                    }
                    .listStyle(.plain)
                }
            }
            .navigationTitle("Contents")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button { dismiss() } label: {
                        Image(systemName: "xmark")
                            .foregroundStyle(.secondary)
                    }
                }
            }
            .preferredColorScheme(preferredColorScheme)
        }
        .task { await loadTOC() }
    }
    
    // Compute color scheme from settings to update sheet immediately
    private var preferredColorScheme: ColorScheme? {
        switch settingsStore.settings.theme {
        case .light:
            return .light
        case .dark:
            return .dark
        case .system:
            return nil  // nil = follow system appearance
        }
    }

    private func loadTOC() async {
        guard let pub = publication else { return }
        
        let links: [RLink]
        switch await pub.tableOfContents() {
        case .success(let l): links = l
        case .failure: links = []
        }
        
        // Flatten with depth tracking
        var flat: [(link: RLink, depth: Int)] = []
        func walk(_ node: RLink, depth: Int) {
            flat.append((node, depth))
            for child in node.children {
                walk(child, depth: depth + 1)
            }
        }
        links.forEach { walk($0, depth: 0) }
        
        // Get full positions list for progression calculation
        let allPositions: [Locator]
        switch await pub.positions() {
        case .success(let p): allPositions = p
        case .failure: allPositions = []
        }
        let totalCount = Double(allPositions.count)
        
        // Build a lookup: href → first position index for that resource
        var hrefToProgression: [AnyURL: Double] = [:]
        if totalCount > 0 {
            for (index, locator) in allPositions.enumerated() {
                let href = locator.href
                if hrefToProgression[href] == nil {
                    hrefToProgression[href] = Double(index) / totalCount
                }
            }
        }
        
        // Resolve each link to get its progression
        var entries: [(link: RLink, depth: Int, progress: Double?)] = []
        for (link, depth) in flat {
            // Try locate first, fall back to positions lookup
            var progress: Double? = nil
            if let locator = await pub.locate(link) {
                progress = locator.locations.totalProgression
            }
            if progress == nil {
                // Fall back: match by href
                progress = hrefToProgression[link.url()]
            }
            entries.append((link, depth, progress))
        }
        
        tocEntries = entries
    }
}
