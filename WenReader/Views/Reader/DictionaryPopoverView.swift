// Copyright 2025 Oliver Zhang
// Licensed under the MIT License

import Foundation
import SwiftUI

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
            .foregroundStyle(.primary)
            .fixedSize(horizontal: false, vertical: true)
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

// MARK: - DictionaryPopover
struct DictionaryPopover: View {
    let result: DictionaryResult
    let wordHit: WordHit
    let canGoBack: Bool
    let onBack: () -> Void
    let onLinkTap: (LinkedHeadword) -> Void

    @State private var selectedSenseIndex: Int
    
    // For animations to track.
    var contentKey = "constant_placeholder_no_effect"

    init(
        result: DictionaryResult,
        wordHit: WordHit,
        initialSenseIndex: Int = 0,
        canGoBack: Bool,
        onBack: @escaping () -> Void,
        onLinkTap: @escaping (LinkedHeadword) -> Void
    ) {
        self.result = result
        self.wordHit = wordHit
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

    @EnvironmentObject var settingsStore: SettingsStore

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
                
                TextExportMenu(
                    wordHit: wordHit,
                    promptStyle: $settingsStore.settings.promptStyle
                )
            }

            // Headword: simplified [traditional] (with diff masking)
            if let entry = currentEntry {
                HStack(alignment: .firstTextBaseline, spacing: 4) {
                    Text(entry.simplified)
                        .font(.title2)

                    if entry.traditional != entry.simplified {
                        Text("[\(maskedTraditional(simp: entry.simplified, trad: entry.traditional))]")
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

    // MARK: - Traditional diff masking
    
    /// For multi-char words, replaces trad chars that match simp with ー (fullwidth)
    /// so only the *different* characters stand out visually.
    /// Single-char words or length mismatches show the full traditional form.
    private func maskedTraditional(simp: String, trad: String) -> String {
        let simpChars = Array(simp)
        let tradChars = Array(trad)
        
        // Only mask when lengths match and word is multi-char
        guard simpChars.count == tradChars.count, simpChars.count > 1 else {
            return trad
        }
        
        // U+FF0D fullwidth hyphen-minus — same visual width as a CJK char
        let placeholder: Character = "\u{FF0D}"
        
        return String(zip(simpChars, tradChars).map { s, t in
            s == t ? placeholder : t
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

        return LinkedHeadword(traditional: t, simplified: s)
    }
}
