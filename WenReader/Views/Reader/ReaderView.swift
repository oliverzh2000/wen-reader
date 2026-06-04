// Copyright 2025 Oliver Zhang
// Licensed under the MIT License

import SwiftUI
import UIKit

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
        .toolbar(.hidden, for: .tabBar)
    }
}

// MARK: - Reader
struct ReaderView: View {
    @EnvironmentObject private var chrome: UiState
    @EnvironmentObject private var catalog: CatalogStore
    @EnvironmentObject private var settingsStore: SettingsStore
    @Environment(\.horizontalSizeClass) private var horizontalSizeClass

    let book: BookItem

    @StateObject private var engine = ReadiumEngine()
    
    // Observe dictionary manager changes explicitly so SwiftUI re-renders on dictionary updates
    @ObservedObject private var dictionaryManager: ReaderDictionaryManager

    @State private var showChrome = false
    @State private var showChapters = false
    @State private var showSettings = false
    @State private var didSync = false
    
    init(book: BookItem) {
        self.book = book
        
        // Initialize engine first
        let engine = ReadiumEngine()
        _engine = StateObject(wrappedValue: engine)
        
        // Then observe its dictionary manager
        _dictionaryManager = ObservedObject(wrappedValue: engine.dictionaryManager)
    }

    var body: some View {
        ZStack {
            ReaderSurface(engine: engine)

            GeometryReader { proxy in
                if let hit = engine.currentWordHit, let result = dictionaryManager.currentResult {
                    let screenWidth = proxy.size.width
                    let screenHeight = proxy.size.height
                    let hitX = hit.hitPoint.x
                    let hitY = hit.hitPoint.y

                    // On iPad 2-column: place dictionary on the OPPOSITE column,
                    // vertically aligned with the word. On iPhone: top/bottom as before.
                    let isRegular = horizontalSizeClass == .regular
                    let alignment: Alignment = {
                        if isRegular {
                            // Opposite column, same vertical half
                            let onLeft = hitX < screenWidth / 2
                            let onTop = hitY < screenHeight / 2
                            if onLeft && onTop { return .topTrailing }
                            if onLeft && !onTop { return .bottomTrailing }
                            if !onLeft && onTop { return .topLeading }
                            return .bottomLeading
                        } else {
                            // iPhone: top or bottom based on word position
                            return (hitY > screenHeight / 2) ? .top : .bottom
                        }
                    }()

                    DictionaryPopover(
                        result: result,
                        wordHit: hit,
                        initialSenseIndex: 0,
                        canGoBack: engine.canGoBackInDictionary,
                        onBack: {
                            engine.popDictionary()
                        },
                        onLinkTap: { headword in
                            Task {
                                await engine.pushDictionary(for: headword.simplified)
                            }
                        }
                    )
                    .padding(.horizontal)
                    .frame(width: isRegular ? screenWidth / 2 : nil)
                    .frame(maxWidth: isRegular ? nil : .infinity)
                    .frame(maxHeight: ReaderConstants.Dictionary.popoverMaxHeight)
                    .frame(
                        maxWidth: .infinity,
                        maxHeight: .infinity,
                        alignment: alignment
                    )
                    .transition(.opacity)
                    .zIndex(1)
                }
            }
        }
       .safeAreaInset(edge: .bottom, spacing: 0) {
           // Show word adjustment bar when dictionary is active, otherwise show reading progress.
           // Only one is visible at a time, both occupy the same bottom slot.
           // ignoresSafeArea lets the content extend into the unsafe area;
           // the inner frame centers it vertically within the full height.
           Group {
               if engine.currentWordHit != nil {
                   WordAdjustmentBar(
                       onPrev: { Task { await engine.navigateWord(.prev) } },
                       onShrink: { Task { await engine.shrinkSelection() } },
                       onGrow: { Task { await engine.expandSelection() } },
                       onNext: { Task { await engine.navigateWord(.next) } },
                       autoAdvanceInterval: settingsStore.settings.autoAdvanceInterval
                   )
                   .transition(.opacity)
               } else if let progression = engine.currentProgression {
                   Text(String(format: "%.1f%%", progression * 100))
                       .font(.caption)
                       .foregroundStyle(.secondary)
                       .frame(maxWidth: .infinity)
               }
           }
           .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .center)
           .padding(.top)
           .frame(height: 28)
       }
        .animation(
            .spring(
                response: ReaderConstants.Dictionary.animationResponse,
                dampingFraction: ReaderConstants.Dictionary.animationDamping
            ),
            value: engine.dictionaryManager.currentResult != nil
        )
        .navigationTitle(book.title ?? "")
        .navigationBarTitleDisplayMode(.inline)
        .readerChrome(
            title: book.title ?? "",
            showChrome: $showChrome,
            showChapters: $showChapters,
            showSettings: $showSettings,
            showReturnButton: engine.canReturn,
            onReturn: {
                Task { await engine.goBackToReturnLocator() }
            }
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

                // Record that this book was just opened (for sorting by recency)
                if engine.navigatorVC != nil {
                    var updated = book
                    updated.lastOpened = Date()
                    catalog.update(updated)
                }

                engine.installInputObservers(
                    onSingleTap: {
                        // Single tap will dismiss highlight/dict if present, otherwise toggle chrome.
                        if engine.currentWordHit != nil {
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
        .errorAlert(title: "Failed to Open Book", error: $engine.openError)
        .sheet(isPresented: $showChapters) {
            TableOfContentsSheet(
                publication: engine.publication,
                book: book,
                coverImage: catalog.coverImage(for: book),
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
    let showReturnButton: Bool
    let onReturn: () -> Void

    // Disambiguate SwiftUI's Content explicitly for this modifier type.
    // Namespace collision between SwiftUI and ReadiumShared Content!
    typealias Content = SwiftUI._ViewModifier_Content<ReaderChromeModifier>

    func body(content: Content) -> some View {
        content
            .statusBarHidden(!showChrome)
            .toolbar {
                ToolbarItemGroup(placement: .topBarLeading) {
                    if showReturnButton {
                        Button(action: onReturn) {
                            Image(systemName: "arrow.uturn.backward")
                        }
                    }
                }
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
                        HStack(spacing: 0) {
                            Text(title).lineLimit(1)
                                .font(.body)
                                .foregroundStyle(.secondary)
                            Image(systemName: "chevron.right")  // subtle disclosure cue
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                        .fontWeight(.semibold)
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
                            Image(systemName: "text.magnifyingglass")
                                .foregroundColor(
                                    settingsStore.settings.interactionMode == .custom
                                    ? .accentColor
                                    : .secondary
                                )
                        }
                        Button {
                            showSettings = true
                        } label: {
                            Image(systemName: "slider.horizontal.3")
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
        showSettings: Binding<Bool>,
        showReturnButton: Bool,
        onReturn: @escaping () -> Void
    ) -> some View {
        modifier(
            ReaderChromeModifier(
                title: title,
                showChrome: showChrome,
                showChapters: showChapters,
                showSettings: showSettings,
                showReturnButton: showReturnButton,
                onReturn: onReturn
            )
        )
    }
}
