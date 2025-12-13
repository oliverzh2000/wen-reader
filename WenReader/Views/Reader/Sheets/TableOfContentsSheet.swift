// Copyright 2025 Oliver Zhang
// Licensed under the MIT License

import Foundation
import ReadiumShared
import SwiftUI

struct TableOfContentsSheet: View {
    let publication: Publication?
    let onSelect: (RLink) -> Void

    @State private var tocLinks: [RLink] = []
    @State private var isLoading = true

    @Environment(\.dismiss) private var dismiss
    @EnvironmentObject var settingsStore: SettingsStore

    var body: some View {
        NavigationStack {
            Group {
                if isLoading {
                    ProgressView("Loading TOC…")
                } else if tocLinks.isEmpty {
                    Text("No table of contents.").foregroundStyle(.secondary)
                } else {
                    List {
                        ForEach(flattenTOC(tocLinks), id: \.href) { link in
                            Button(link.title ?? link.href) {
                                onSelect(link)
                                dismiss()
                            }
                        }
                    }
                }
            }
            .navigationTitle("Contents")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Done") { dismiss() }
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
        switch await pub.tableOfContents() {
        case .success(let links): tocLinks = links
        case .failure: tocLinks = []
        }
        isLoading = false
    }

    private func flattenTOC(_ links: [RLink]) -> [RLink] {
        var out: [RLink] = []
        func walk(_ n: RLink) {
            out.append(n)
            n.children.forEach(walk)
        }
        links.forEach(walk)
        return out
    }
}
