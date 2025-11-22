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

            GeometryReader { proxy in
                if let result = engine.currentDictResult {
                    // If the word is in the bottom half of the screen, show popover on top, else bottom
                    let screenHeight = proxy.size.height
                    let hitY =
                        engine.currentWordHit?.hitPoint.y ?? screenHeight / 2
                    let alignment: Alignment =
                        (hitY > screenHeight / 2) ? .top : .bottom
                    let edge: Edge = (alignment == .top) ? .top : .bottom

                    // Popover pinned to top or bottom
                    DictionaryPopover(
                        result: result,
                        initialSenseIndex: 0,
                        canGoBack: engine.canGoBackInDictionary,
                        onBack: {
                            engine.popDictionary()
                        },
                        onLinkTap: { headword in
                            // Use either trad or simp; the SQL WHERE matches both.
                            engine.pushDictionary(for: headword.simplified)
                        }
                    )
                    .padding()
                    .frame(maxHeight: 300)
                    .frame(
                        maxWidth: .infinity,
                        maxHeight: .infinity,
                        alignment: alignment
                    )
                    .transition(
                        .opacity
                            .combined(with: .move(edge: edge))
                    )
                    .zIndex(1)
                }
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
                        // Single tap will hide dict if present, otherwise toggle chrome.
                        if engine.currentDictResult != nil {
                            engine.closeDictionaryAndClearHighlight()
                        } else {
                            // Toggle chrome on any single tap
                            withAnimation(.easeInOut) {
                                showChrome.toggle()
                                chrome.hideStatusBar = !showChrome
                            }
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
                                systemName: settingsStore.settings
                                    .interactionMode
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

struct GlossView: View {
    let gloss: Gloss
    let onLinkTap: (LinkedHeadword) -> Void

    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: 0) {
            ForEach(Array(gloss.fragments.enumerated()), id: \.offset) { _, fragment in
                switch fragment {
                case .text(let text):
                    Text(text)
                case .accentedPinyin(let syllables):
                    Text("\(syllables.joined(separator: " "))")
                        .bold()
                        .foregroundStyle(.secondary)
                case .link(let headword):
                    Button {
                        onLinkTap(headword)
                    } label: {
                        HStack(spacing: 2) {
                            Text(headword.simplified)
                            
                            if headword.traditional != headword.simplified {
                                Text("[\(headword.traditional)]")
                                    .foregroundStyle(.secondary)
                            }
                            
                            Text("\(headword.accentedPinyin.joined(separator: " "))")
                                .bold()
                                .foregroundStyle(.secondary)
                        }
                    }
                    .buttonStyle(.plain)
                    .foregroundStyle(.blue)
                }
            }
        }
    }
}


// MARK: - Dictionary Popover
struct DictionaryPopover: View {
    let result: DictionaryResult
    let canGoBack: Bool
    let onBack: () -> Void
    let onLinkTap: (LinkedHeadword) -> Void

    @State private var selectedSenseIndex: Int

    init(
        result: DictionaryResult,
        initialSenseIndex: Int = 0,
        canGoBack: Bool,
        onBack: @escaping () -> Void,
        onLinkTap: @escaping (LinkedHeadword) -> Void
    ) {
        self.result = result
        self.canGoBack = canGoBack
        self.onBack = onBack
        self.onLinkTap = onLinkTap
        _selectedSenseIndex = State(initialValue: initialSenseIndex)
    }

    private var currentEntry: Entry? {
        guard !result.entries.isEmpty,
            selectedSenseIndex >= 0,
            selectedSenseIndex < result.entries.count
        else {
            return nil
        }
        return result.entries[selectedSenseIndex]
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {

            // Header: Back button (if stack > 1), pinyin, index
            HStack(alignment: .firstTextBaseline) {
                if canGoBack {
                    Button {
                        onBack()
                    } label: {
                        Image(systemName: "chevron.backward")
                    }
                    .buttonStyle(.plain)
                    .padding(.trailing, 4)
                }

                if let entry = currentEntry {
                    Text(entry.accentedPinyin.joined(separator: " "))
                        .font(.headline)
                        .bold()
                        .foregroundStyle(.secondary)
                }

                Spacer()

                Text("\(selectedSenseIndex + 1) / \(result.entries.count)")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            // Headword: simplified [traditional]
            if let entry = currentEntry {
                HStack(alignment: .firstTextBaseline, spacing: 4) {
                    Text(entry.simplified)
                        .font(.title2)

                    if entry.traditional != entry.simplified {
                        Text("[\(entry.traditional)]")
                            .font(.title2)
                            .foregroundStyle(.secondary)
                    }
                }
            }

            // Entries: horizontally swipable, each page shows all senses
            TabView(selection: $selectedSenseIndex) {
                ForEach(Array(result.entries.enumerated()), id: \.offset) {
                    index,
                    entry in
                    ScrollView {
                        VStack(alignment: .leading, spacing: 8) {
                            ForEach(
                                Array(entry.senses.enumerated()),
                                id: \.offset
                            ) { defIndex, sense in
                                HStack(alignment: .top, spacing: 8) {
                                    // Sense index
                                    Text("\(defIndex + 1).")
                                        .fontDesign(.monospaced)
                                        .font(.caption)
                                        .fontWeight(.medium)
                                        .foregroundStyle(.secondary)

                                    // All glosses for this sense
                                    VStack(alignment: .leading, spacing: 4) {
                                        ForEach(
                                            Array(sense.glosses.enumerated()),
                                            id: \.offset
                                        ) { _, gloss in
                                            GlossView(
                                                gloss: gloss,
                                                onLinkTap: onLinkTap
                                            )
                                        }
                                    }
                                    .font(.subheadline)
                                }
                                .padding(.vertical, 2)
                            }
                            .frame(maxWidth: .infinity, alignment: .leading)
                        }
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(.top, 4)
                    }
                    .tag(index)
                }
            }
            .tabViewStyle(.page(indexDisplayMode: .never))
        }
        .padding()
        .background(Color(.secondarySystemBackground))
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
//struct DictionaryPopover_Previews: PreviewProvider {
//    static var previews: some View {
//        let sample = DictionaryEntry(
//            headword: "长",
//            senses: [
//                .init(
//                    traditional: "長",
//                    simplified: "长",
//                    accentedPinyin: ["cháng"],
//                    glosses: [
//                        "long; lengthy",
//                        "to grow; to develop",
//                    ]
//                ),
//                .init(
//                    traditional: "長",
//                    simplified: "长",
//                    accentedPinyin: ["zhǎng"],
//                    glosses: [
//                        "to head; to lead",
//                        "elder; senior; chief",
//                    ]
//                ),
//            ]
//        )
//
//        DictionaryPopover(entry: sample)
//            .padding()
//            .previewLayout(.sizeThatFits)
//    }
//}
