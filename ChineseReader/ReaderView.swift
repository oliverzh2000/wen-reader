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
            if engine.navigatorVC != nil {
                // Readium's EPUB rendering window.
                NavigatorHost(
                    navigatorVC: engine.navigatorVC!,
                    onLayout: {
                        engine.tightenVerticalMargins()
                    }
                )
                    .onAppear {
                        engine.apply(settingsStore.settings, systemColorScheme)
                    }
                    .onChange(of: systemColorScheme) { _, _ in
                        engine.apply(settingsStore.settings, systemColorScheme)
                    }
                    .onChange(of: settingsStore.settings) { _, newSettings in
                        engine.apply(newSettings, systemColorScheme)
                    }
            } else if let error = engine.openError {
                VStack(spacing: 12) {
                    Text("Failed to open").font(.headline)
                    Text(error.localizedDescription).font(.footnote)
                        .multilineTextAlignment(.center)
                }
                .padding()
            } else if engine.isOpening {
                // Very brief - no need to show anything like "Opening...".
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
                if let hit = engine.currentWordHit, let result = engine.currentDictResult {
                    // If the word is in the bottom half of the screen, show popover on top, else bottom
                    let screenHeight = proxy.size.height
                    let hitY =
                        engine.currentWordHit?.hitPoint.y ?? screenHeight / 2
                    let alignment: Alignment =
                        (hitY > screenHeight / 2) ? .top : .bottom

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
                    .padding([.leading, .trailing, .bottom])
                    .ignoresSafeArea(edges: .bottom)
                    .frame(maxHeight: 300)
                    .frame(
                        maxWidth: .infinity,
                        maxHeight: .infinity,
                        alignment: alignment
                    )
                    .transition(
                        .opacity
                        )
                    .zIndex(1)
                }
            }
        }
        .animation(
            .spring(response: 0.25, dampingFraction: 1.0),
            value: engine.currentDictResult != nil
        )
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

// MARK: - SenseView
struct SenseView: View {
    let sense: Sense
    // We don't call this directly; instead we encode into URL and
    // let DictionaryPopover's .openURL handler call onLinkTap.
    // Keeping the closure here in case you want to evolve this later.
    let makeLinkURL: (LinkedHeadword) -> URL?

    var body: some View {
        Text(makeAttributedString())
            .font(.subheadline)
            .fixedSize(horizontal: false, vertical: true) // allow multiline wrap
    }

    private func makeAttributedString() -> AttributedString {
        var result = AttributedString()

        for (glossIndex, gloss) in sense.glosses.enumerated() {
            if glossIndex > 0 {
                result.append(AttributedString("; "))
            }
            result.append(attributedString(for: gloss))
        }

        return result
    }

    private func attributedString(for gloss: Gloss) -> AttributedString {
        var output = AttributedString()

        for (fragmentIndex, fragment) in gloss.fragments.enumerated() {
            if fragmentIndex > 0 {
                output.append(AttributedString(" "))
            }

            switch fragment {
            case .text(let text):
                output.append(AttributedString(text))

            case .accentedPinyin(let syllables):
                var pinyin = AttributedString(syllables.joined(separator: " "))
                pinyin.inlinePresentationIntent = .stronglyEmphasized
                pinyin.foregroundColor = .secondary
                output.append(pinyin)

            case .link(let headword):
                var label = AttributedString(headword.simplified)
                if headword.traditional != headword.simplified {
                    label.append(AttributedString("[\(headword.traditional)]"))
                }

                // Style like a link
                label.foregroundColor = .blue

                // Add URL so SwiftUI treats it as a tappable link
                if let url = makeLinkURL(headword) {
                    label.link = url
                }

                output.append(label)
            }
        }

        return output
    }
}

// MARK: - Dictionary Popover

struct DictionaryPopover: View {
    let result: DictionaryResult
    let canGoBack: Bool
    let onBack: () -> Void
    let onLinkTap: (LinkedHeadword) -> Void

    @State private var selectedSenseIndex: Int
    
    // For animations to track.
    var contentKey = "constant_placeholder_no_effect"

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
                ForEach(Array(result.entries.enumerated()), id: \.offset) { index, entry in
                    ScrollView {
                        // Preprocess senses so we can:
                        // - skip numbering classifiers
                        // - decide if a CL is attached (indented) or global (not indented)
                        let rows: [(id: Int, sense: Sense, marker: String, isAttachedClassifier: Bool)] = {
                            var result: [(Int, Sense, String, Bool)] = []
                            var runningNumber = 0
                            let senses = entry.senses

                            for (i, sense) in senses.enumerated() {
                                let isLast = i == senses.count - 1

                                if sense.isClassifier {
                                    if isLast {
                                        // Global classifier for the whole word → show "CL:" in marker column, no indent.
                                        result.append((i, sense, "CL:", false))
                                    } else {
                                        // Attached classifier for the *previous* numbered sense
                                        // → empty marker (so numbering doesn't jump), and indent in content column.
                                        result.append((i, sense, "", true))
                                    }
                                } else {
                                    // Normal sense gets a number
                                    runningNumber += 1
                                    result.append((i, sense, "\(runningNumber).", false))
                                }
                            }

                            return result
                        }()

                        Grid(alignment: .leadingFirstTextBaseline, verticalSpacing: 8) {
                            ForEach(rows, id: \.id) { row in
                                GridRow {
                                    // Marker column: numbers or "CL:" for global classifier.
                                    Text(row.marker)
                                        .font(.caption)
                                        .fontWeight(.medium)
                                        .foregroundStyle(.secondary)
                                        .gridColumnAlignment(.trailing)

                                    // Content column
                                    if row.isAttachedClassifier {
                                        // Attached CL → indent by keeping marker col empty and
                                        // putting "CL:" + SenseView in an HStack.
                                        HStack(alignment: .firstTextBaseline, spacing: 4) {
                                            Text("CL:")
                                                .font(.caption)
                                                .fontWeight(.medium)
                                                .foregroundStyle(.secondary)

                                            SenseView(
                                                sense: row.sense,
                                                makeLinkURL: { headword in
                                                    linkURL(for: headword)
                                                }
                                            )
                                        }
                                    } else {
                                        // Normal sense or global CL
                                        SenseView(
                                            sense: row.sense,
                                            makeLinkURL: { headword in
                                                linkURL(for: headword)
                                            }
                                        )
                                    }
                                }
                            }
                        }
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(.top, 4)
                    }
                    .tag(index)
                }
            }
            .tabViewStyle(.page(indexDisplayMode: .never))
        }
        .animation(
            .easeInOut(duration: 0.18),
            value: contentKey
        )
        .padding()
        .background(Color(.secondarySystemBackground))
        .clipShape(RoundedRectangle(cornerRadius: 32, style: .continuous))
        .shadow(radius: 32)
        // Handle taps on AttributedString links
        .environment(\.openURL, OpenURLAction { url in
            if let headword = decodeLinkedHeadword(from: url) {
                onLinkTap(headword)
                return .handled
            } else {
                return .discarded
            }
        })
    }

    // MARK: - Link encoding/decoding
    private func linkURL(for headword: LinkedHeadword) -> URL? {
        let allowed = CharacterSet.urlQueryAllowed
        let s = headword.simplified.addingPercentEncoding(withAllowedCharacters: allowed) ?? ""
        let t = headword.traditional.addingPercentEncoding(withAllowedCharacters: allowed) ?? ""

        return URL(string: "crdict://headword?s=\(s)&t=\(t)")
    }

    private func decodeLinkedHeadword(from url: URL) -> LinkedHeadword? {
        guard url.scheme == "crdict" else { return nil }
        guard let components = URLComponents(url: url, resolvingAgainstBaseURL: false),
              components.host == "headword"
        else { return nil }

        var simplified: String?
        var traditional: String?

        components.queryItems?.forEach { item in
            switch item.name {
            case "s": simplified = item.value
            case "t": traditional = item.value
            default: break
            }
        }

        guard let s = simplified, !s.isEmpty else { return nil }
        let t = (traditional?.isEmpty ?? true) ? s : traditional!

        // Adjust to your actual LinkedHeadword initializer
        return LinkedHeadword(traditional: t, simplified: s)
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
