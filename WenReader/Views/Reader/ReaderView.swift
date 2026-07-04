// Copyright 2025 Oliver Zhang
// Licensed under the MIT License

import ReadiumShared
import SwiftUI
import UIKit

// MARK: - Surface
private struct ReaderSurface: View {
    @ObservedObject var engine: ReadiumEngine
    @EnvironmentObject var settingsStore: SettingsStore
    @Environment(\.colorScheme) private var systemColorScheme

    var keyActions = KeyActions()

    var body: some View {
        Group {
            if engine.navigatorVC != nil {
                // Readium's EPUB rendering window.
                NavigatorHost(
                    navigatorVC: engine.navigatorVC!,
                    onLayout: { engine.tightenVerticalMargins() },
                    keyActions: keyActions
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

    let book: BookItem

    @StateObject private var engine = ReadiumEngine()
    
    // Observe dictionary manager changes explicitly so SwiftUI re-renders on dictionary updates
    @ObservedObject private var dictionaryManager: ReaderDictionaryManager

    @State private var showChrome = ProcessInfo.processInfo.isMacCatalystApp
    @State private var showChapters = false
    @State private var showSettings = false
    @State private var didSync = false

    private var isMac: Bool { ProcessInfo.processInfo.isMacCatalystApp }
    
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
            ReaderSurface(
                engine: engine,
                keyActions: isMac ? KeyActions(
                    onArrowLeft:  { Task { await engine.navigateWord(.prev) } },
                    onArrowRight: { Task { await engine.navigateWord(.next) } },
                    onArrowUp:    { Task { await engine.shrinkSelection() } },
                    onArrowDown:  { Task { await engine.expandSelection() } },
                    onSpace:      { engine.toggleAutoAdvance(interval: settingsStore.settings.autoAdvanceInterval) },
                    onEscape:     {
                        if showChapters { showChapters = false }
                        else if showSettings { showSettings = false }
                        else if engine.currentWordHit != nil { engine.closeDictionaryAndClearHighlight() }
                    }
                ) : KeyActions()
            )

            GeometryReader { proxy in
                if let hit = engine.currentWordHit, let result = dictionaryManager.currentResult {
                    let screenHeight = proxy.size.height
                    let hitY = hit.hitPoint.y

                    let alignment: Alignment = (hitY > screenHeight / 2) ? .top : .bottom

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
                    .frame(maxWidth: ReaderConstants.Dictionary.popoverMaxWidth)
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
           Group {
               if engine.currentWordHit != nil {
                   WordAdjustmentBar(
                       onPrev: { engine.stopAutoAdvance(); Task { await engine.navigateWord(.prev) } },
                       onShrink: { engine.stopAutoAdvance(); Task { await engine.shrinkSelection() } },
                       onGrow: { engine.stopAutoAdvance(); Task { await engine.expandSelection() } },
                       onNext: { engine.stopAutoAdvance(); Task { await engine.navigateWord(.next) } },
                       onToggleAutoAdvance: { engine.toggleAutoAdvance(interval: settingsStore.settings.autoAdvanceInterval) },
                       isAutoAdvancing: engine.isAutoAdvancing
                   )
                   .transition(.opacity)
               } else if let progression = engine.currentProgression {
                   Text(String(format: "%.1f%%", progression * 100))
                       .font(.caption)
                       .foregroundStyle(.secondary)
                       .frame(maxWidth: .infinity)
               }
           }
           .frame(maxWidth: ReaderConstants.Dictionary.popoverMaxWidth, maxHeight: .infinity, alignment: .center)
           .padding(.top)
           .padding(.bottom, isMac ? 8 : 0)
           .frame(height: isMac ? 36 : 28)
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
            chrome.hideStatusBar = isMac ? false : !showChrome
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
                        if showChapters { showChapters = false; return }
                        if showSettings { showSettings = false; return }
                        if engine.currentWordHit != nil {
                            engine.closeDictionaryAndClearHighlight()
                        } else if !isMac {
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
        .modifier(TOCPresentation(
            isPresented: $showChapters,
            publication: engine.publication,
            book: book,
            coverImage: catalog.coverImage(for: book),
            onSelect: { link in
                Task { await engine.go(to: link) }
            }
        ))
        .modifier(SettingsPresentation(isPresented: $showSettings))
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

    private var isMac: Bool { ProcessInfo.processInfo.isMacCatalystApp }

    func body(content: Content) -> some View {
        content
            .statusBarHidden(isMac ? false : !showChrome)
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
                        if !isMac {
                            UIImpactFeedbackGenerator(style: .light)
                                .impactOccurred()
                            withAnimation(.easeInOut) {
                                showChrome = true
                                chrome.hideStatusBar = false
                            }
                        }
                        showChapters = true
                    } label: {
                        HStack(spacing: 0) {
                            Text(title).lineLimit(1)
                                .font(.body)
                                .foregroundStyle(.secondary)
                            Image(systemName: "chevron.right")
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
            .navigationBarBackButtonHidden(isMac ? false : !showChrome)
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

// MARK: - TOC Presentation (popover on Mac, sheet on iOS)

private struct TOCPresentation: ViewModifier {
    @Binding var isPresented: Bool
    let publication: Publication?
    let book: BookItem
    let coverImage: UIImage?
    let onSelect: (RLink) -> Void

    @ViewBuilder
    func body(content: SwiftUI._ViewModifier_Content<Self>) -> some View {
        if ProcessInfo.processInfo.isMacCatalystApp {
            content.popover(isPresented: $isPresented, arrowEdge: .top) {
                TableOfContentsSheet(
                    publication: publication,
                    book: book,
                    coverImage: coverImage,
                    onSelect: onSelect
                )
                .frame(width: 320, height: 500)
            }
        } else {
            content.sheet(isPresented: $isPresented) {
                TableOfContentsSheet(
                    publication: publication,
                    book: book,
                    coverImage: coverImage,
                    onSelect: onSelect
                )
                .presentationDetents([.large])
            }
        }
    }
}

// MARK: - Settings Presentation (popover on Mac, sheet on iOS)

private struct SettingsPresentation: ViewModifier {
    @Binding var isPresented: Bool

    @ViewBuilder
    func body(content: SwiftUI._ViewModifier_Content<Self>) -> some View {
        if ProcessInfo.processInfo.isMacCatalystApp {
            content.popover(isPresented: $isPresented, arrowEdge: .top) {
                SettingsSheet()
                    .frame(width: 360, height: 480)
            }
        } else {
            content.sheet(isPresented: $isPresented) {
                SettingsSheet()
                    .presentationDetents([.medium, .large])
            }
        }
    }
}
