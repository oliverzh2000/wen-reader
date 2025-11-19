//
//  ReaderUI.swift
//  ChineseReader
//
//  Created by Oliver Zhang on 2025-11-08.
//

import ReadiumShared
import SwiftUI

// MARK: - Surface
private struct ReaderSurface: View {
    @ObservedObject var engine: ReadiumEngine
    @EnvironmentObject var settingsStore: SettingsStore
    @Environment(\.colorScheme) private var systemColorScheme

    var body: some View {
        Group {
            if engine.isOpening {
                ProgressView("Opening book…")
            } else if let error = engine.openError {
                VStack(spacing: 12) {
                    Text("Failed to open").font(.headline)
                    Text(error.localizedDescription).font(.footnote)
                        .multilineTextAlignment(.center)
                }
                .padding()
            } else if let nav = engine.navigatorVC {
                // Readium's EPUB rendering window.
                NavigatorHost(navigatorVC: nav)
                    // TODO: Fix the source of the extra top/bottom margins.
                    // From investigation using Safari inspect element, seems that
                    // Readium is adding it's own stubborn margin between html element and the edge of the NavigatorHost.
                    .ignoresSafeArea(edges: .bottom)
                    .onAppear {
                        engine.apply(settingsStore.settings, systemColorScheme)
                    }
                    .onChange(of: systemColorScheme) { _, _ in
                        engine.apply(settingsStore.settings, systemColorScheme)
                    }
                    .onChange(of: settingsStore.settings) { _, newSettings in
                        engine.apply(newSettings, systemColorScheme)
                    }
            } else {
                Text("No content")
            }
        }
    }
}

// MARK: - Reader
struct ReaderView: View {
    @EnvironmentObject private var chrome: UiState
    @EnvironmentObject private var catalog: CatalogStore

    let book: BookItem

    @StateObject private var engine = ReadiumEngine()

    @State private var showChrome = false
    @State private var showChapters = false
    @State private var showSettings = false
    @State private var didSync = false

    var body: some View {
        ZStack {
            ReaderSurface(engine: engine)
            
            // Dictionary popover overlay
            if let entry = engine.currentDictEntry {
                // Dimmed backdrop to indicate “modal-ish” state
                Color.gray
                    .opacity(0.0001)
                    .ignoresSafeArea()
                    .onTapGesture {
                        // Tap outside to dismiss
                        withAnimation(.spring()) {
                            engine.currentDictEntry = nil
                        }
                    }

                // Centered popover
                DictionaryPopover(entry: entry)
                    .padding()
                    .frame(maxWidth: 360)
                    .transition(
                        .scale.combined(with: .opacity)
                    )
                    .zIndex(1)
            }
        }
        .navigationTitle(book.displayName)
        .navigationBarTitleDisplayMode(.inline)
        .readerChrome(
            title: book.displayName,
            showChrome: $showChrome,
            showChapters: $showChapters,
            showSettings: $showSettings
        )
        .onAppear {
            guard !didSync else { return }
            chrome.hideStatusBar = !showChrome
            didSync = true
        }
        .task {
            // Open the book once we have a view in place.
            // We pass the top UIView via UIWindowScene to let DRM prompts present if you add LCP later.
            if engine.navigatorVC == nil {
                let url = catalog.localURL(for: book)
                let rootView = UIApplication.shared
                    .connectedScenes
                    .compactMap { $0 as? UIWindowScene }
                    .flatMap { $0.windows }
                    .first?.rootViewController?.view
                await engine.open(
                    bookId: book.id,
                    fileURL: url,
                    sender: rootView
                )

                engine.installInputObservers(
                    onSingleTap: {
                        // Toggle chrome on any single tap
                        withAnimation(.easeInOut) {
                            showChrome.toggle()
                            chrome.hideStatusBar = !showChrome
                        }
                    },
                )
            }
        }
        .sheet(isPresented: $showChapters) {
            TableOfContentsSheet(
                publication: engine.publication,
                onSelect: {
                    link in
                    Task {
                        await engine.go(to: link)
                    }
                }
            )
            .presentationDetents([.large])
        }
        .sheet(isPresented: $showSettings) {
            SettingsSheet()
                .presentationDetents([.medium, .large])
        }
    }
}

// MARK: - Chrome Modifier
struct ReaderChromeModifier: SwiftUI.ViewModifier {
    @EnvironmentObject private var chrome: UiState
    @EnvironmentObject var settingsStore: SettingsStore
    let title: String
    @Binding var showChrome: Bool
    @Binding var showChapters: Bool
    @Binding var showSettings: Bool

    // Disambiguate SwiftUI's Content explicitly for this modifier type.
    // Namespace collision between SwiftUI and ReadiumShared Content!
    typealias Content = SwiftUI._ViewModifier_Content<ReaderChromeModifier>

    func body(content: Content) -> some View {
        content
            .statusBarHidden(!showChrome)
            .toolbar {
                ToolbarItem(placement: .principal) {
                    // Title acts like a button to open chapters
                    Button {
                        UIImpactFeedbackGenerator(style: .light)
                            .impactOccurred()
                        withAnimation(.easeInOut) {
                            showChrome = true
                            chrome.hideStatusBar = false
                        }
                        showChapters = true
                    } label: {
                        HStack {
                            Text(title).lineLimit(1)
                            Image(systemName: "chevron.right")  // subtle disclosure cue
                                .font(.footnote.weight(.semibold))
                                .opacity(0.6)
                        }
                        .contentShape(Rectangle())
                    }
                    .buttonStyle(.plain)
                }
                ToolbarItemGroup(placement: .topBarTrailing) {
                    if showChrome {
                        // Reader interaction mode toggle
                        Button {
                            settingsStore.settings.interactionMode.toggle()
                        } label: {
                            Image(
                                systemName: settingsStore.settings.interactionMode
                                    == .customMagnifier
                                    ? "sparkles" : "text.magnifyingglass"
                            )
                        }
                        Button {
                            showSettings = true
                        } label: {
                            Image(systemName: "textformat.size")
                        }
                    }
                }
            }
            .navigationBarBackButtonHidden(!showChrome)
    }
}

extension View {
    func readerChrome(
        title: String,
        showChrome: Binding<Bool>,
        showChapters: Binding<Bool>,
        showSettings: Binding<Bool>
    ) -> some View {
        modifier(
            ReaderChromeModifier(
                title: title,
                showChrome: showChrome,
                showChapters: showChapters,
                showSettings: showSettings
            )
        )
    }
}

// MARK: - Dictionary Popover
struct DictionaryPopover: View {
    let entry: DictionaryEntry
    
    @State private var selectedSenseIndex: Int
    
    init(entry: DictionaryEntry, initialSenseIndex: Int = 0) {
        self.entry = entry
        let clamped = entry.senses.isEmpty
            ? 0
            : min(max(0, initialSenseIndex), entry.senses.count - 1)
        _selectedSenseIndex = State(initialValue: clamped)
    }
    
    private var currentSense: DictionaryEntry.Sense? {
        guard entry.senses.indices.contains(selectedSenseIndex) else { return nil }
        return entry.senses[selectedSenseIndex]
    }
    
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            // Header: pinyin + headword for current sense
            if let sense = currentSense {
                // Pinyin
                if !sense.accentedPinyin.isEmpty {
                    Text(sense.accentedPinyin.joined(separator: " "))
                        .font(.footnote)            // smaller
                        .fontWeight(.semibold)      // bold-ish
                        .foregroundStyle(.secondary)
                }
                
                // Headword: simplified [traditional]
                HStack(alignment: .firstTextBaseline, spacing: 4) {
                    Text(sense.simplified)
                        .font(.title2)
                        .fontWeight(.semibold)
                    
                    Text("[\(sense.traditional)]")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                    
                    Spacer()
                    
                    // Optional: sense index indicator
                    if !entry.senses.isEmpty {
                        Text("\(selectedSenseIndex + 1) / \(entry.senses.count)")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
            }
            
            // Senses: horizontally swipable with snapping
            TabView(selection: $selectedSenseIndex) {
                ForEach(Array(entry.senses.enumerated()), id: \.offset) { index, sense in
                    VStack(alignment: .leading, spacing: 6) {
                        ForEach(Array(sense.definitions.enumerated()), id: \.offset) { defIndex, definition in
                            HStack(alignment: .top, spacing: 6) {
                                Text("\(defIndex + 1).")
                                    .font(.caption)
                                    .fontWeight(.medium)
                                    .padding(.top, 2)
                                
                                Text(definition)
                                    .font(.body)
                                    .fixedSize(horizontal: false, vertical: true)
                            }
                        }
                        
                        Spacer(minLength: 0)
                    }
                    .padding(.horizontal, 4)
                    .tag(index)
                }
            }
            .tabViewStyle(.page(indexDisplayMode: .automatic)) // horizontal snap between senses
            .frame(minHeight: 120, maxHeight: 220)              // tweak as needed
        }
        .padding(12)
        .background(.thinMaterial)
        .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
        .shadow(radius: 8)
    }
}

// MARK: - Bottom Sheets
struct TableOfContentsSheet: View {
    let publication: Publication?
    let onSelect: (RLink) -> Void

    @State private var tocLinks: [RLink] = []
    @State private var isLoading = true

    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            Group {
                if isLoading {
                    ProgressView("Loading TOC…")
                } else if tocLinks.isEmpty {
                    Text("No table of contents.").foregroundStyle(.secondary)
                } else {
                    List {
                        ForEach(flattenTOC(tocLinks), id: \.hrefOrId) { link in
                            Button(link.title ?? link.hrefOrId) {
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
        }
        .task { await loadTOC() }
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

extension RLink {
    fileprivate var hrefOrId: String { href ?? title ?? UUID().uuidString }
}

struct SettingsSheet: View {
    @EnvironmentObject var settingsStore: SettingsStore

    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            Form {
                Section("Typography") {
                    Picker("Font", selection: $settingsStore.settings.font) {
                        ForEach(ReaderFont.allCases) { font in
                            Text(font.displayName).tag(font)
                        }
                    }
                    .pickerStyle(.automatic)

                    VStack(alignment: .leading) {
                        Text("Font Size")
                        Slider(
                            value: $settingsStore.settings.fontSize,
                            in: 1.0...2.0,
                            step: 0.1
                        ) {
                        } minimumValueLabel: {
                            Text("")
                        } maximumValueLabel: {
                            Text(
                                "\(String(format: "%.1f", settingsStore.settings.fontSize))"
                            )
                            .foregroundStyle(.secondary)
                        }
                    }

                    VStack(alignment: .leading) {
                        Text("Line Height")
                        Slider(
                            value: $settingsStore.settings.lineHeight,
                            in: 1.0...2.0,
                            step: 0.1
                        ) {
                        } minimumValueLabel: {
                            Text("")
                        } maximumValueLabel: {
                            Text(
                                "\(String(format: "%.1f", settingsStore.settings.lineHeight))"
                            )
                            .foregroundStyle(.secondary)
                        }
                    }

                    VStack(alignment: .leading) {
                        Text("Margins")
                        Slider(
                            value: $settingsStore.settings.margins,
                            in: 0.0...2.0,
                            step: 0.1
                        ) {
                        } minimumValueLabel: {
                            Text("")
                        } maximumValueLabel: {
                            Text(
                                "\(String(format: "%.1f", settingsStore.settings.margins))"
                            )
                            .foregroundStyle(.secondary)
                        }
                    }

                    Toggle(
                        "Justify Text",
                        isOn: $settingsStore.settings.justify
                    )
                }

                Section("Appearance") {
                    Picker("Theme", selection: $settingsStore.settings.theme) {
                        ForEach(ReaderTheme.allCases) { theme in
                            Text(theme.displayName).tag(theme)
                        }
                    }
                    .pickerStyle(.automatic)
                }

                Section {
                    Button("Reset to Defaults", role: .destructive) {
                        settingsStore.settings = ReaderSettings()
                    }
                }
            }
            .navigationTitle("Reading Settings")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Done") { dismiss() }
                }
            }
        }
    }
}

// MARK: - Preview
struct DictionaryPopover_Previews: PreviewProvider {
    static var previews: some View {
        let sample = DictionaryEntry(
            headword: "长",
            senses: [
                .init(
                    traditional: "長",
                    simplified: "长",
                    accentedPinyin: ["cháng"],
                    definitions: [
                        "long; lengthy",
                        "to grow; to develop"
                    ]
                ),
                .init(
                    traditional: "長",
                    simplified: "长",
                    accentedPinyin: ["zhǎng"],
                    definitions: [
                        "to head; to lead",
                        "elder; senior; chief"
                    ]
                )
            ]
        )
        
        DictionaryPopover(entry: sample)
            .padding()
            .previewLayout(.sizeThatFits)
    }
}
