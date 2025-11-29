//
//  ReadiumEngine.swift
//  ChineseReader
//
//  Created by Oliver Zhang on 2025-11-08.
//

import Combine
import ReadiumAdapterGCDWebServer
import ReadiumNavigator
import ReadiumShared
import ReadiumStreamer
import SwiftUI
import WebKit

typealias RLink = ReadiumShared.Link

extension FontFamily {
    public static let notoSerifSC: FontFamily = "Noto Serif SC"
    public static let pingFangSC: FontFamily = "PingFang SC"
}

@MainActor
final class ReadiumEngine: ObservableObject {
    // Outputs for the UI
    @Published var publication: Publication?
    @Published var navigatorVC: EPUBNavigatorViewController?
    @Published var openError: Error?
    @Published var isOpening: Bool = false
    @Published var currentLocation: Locator?
    
    // MARK: Core components (kept for the app lifetime of this engine)
    private lazy var httpClient = DefaultHTTPClient()
    // Use Readium’s GCD-based HTTP server; this serves resources to the navigator
    private lazy var httpServer = GCDHTTPServer(assetRetriever: assetRetriever)
    private lazy var assetRetriever = AssetRetriever(httpClient: httpClient)
    private lazy var publicationOpener = PublicationOpener(
        parser: DefaultPublicationParser(
            httpClient: httpClient,
            assetRetriever: assetRetriever,
            pdfFactory: DefaultPDFDocumentFactory()
        )
    )
    
    // Book identity for persistence
    private var bookId: UUID?
    private var lastSavedAt = Date.distantPast
    
    private let interactionManager = ReaderInteractionManager()
    
    // MARK: Dictionary
    @Published var currentWordHit: WordHit?
    @Published private(set) var currentDictResult: DictionaryResult?
    private var dictStack: [DictionaryResult] = []
    private let dictionaryService: DictionaryService = CedictSqlService.shared

    func updateDictionaryResult(for word: String?) {
        Task {
            guard let word else {
                self.closeDictionaryAndClearHighlight()
                return
            }

            if let result = await self.dictionaryService.lookup(word) {
                self.dictStack.removeAll()
                self.dictStack.append(result)
                self.currentDictResult = result
                return
            }
        }
    }

    func closeDictionaryAndClearHighlight() {
        dictStack.removeAll()
        currentDictResult = nil
        interactionManager.clearHighlight()
    }
    
    // To click into a link.
    func pushDictionary(for word: String) {
        Task {
            if let result = await dictionaryService.lookup(word) {
                dictStack.append(result)
                currentDictResult = result
            }
        }
    }

    // To go back after clicking into a link.
    func popDictionary() {
        guard !dictStack.isEmpty else { return }
        dictStack.removeLast()
        currentDictResult = dictStack.last
    }

    var canGoBackInDictionary: Bool {
        dictStack.count > 1
    }

    // MARK: Open
    func open(bookId: UUID, fileURL: URL, sender: UIView?) async {
        guard !isOpening else { return }
        self.bookId = bookId
        isOpening = true
        openError = nil

        do {
            let assetResult = await assetRetriever.retrieve(
                url: fileURL.anyURL.absoluteURL!
            )
            guard case .success(let asset) = assetResult else {
                if case .failure(let e) = assetResult { throw e }
                return
            }

            let openResult = await publicationOpener.open(
                asset: asset,
                // If you add LCP later, set to true so the toolkit may prompt
                allowUserInteraction: true,
                sender: sender
            )
            guard case .success(let pub) = openResult else {
                if case .failure(let e) = openResult { throw e }
                return
            }

            self.publication = pub

            // Decide profile; we’re targeting EPUB here
            guard pub.conforms(to: .epub) else {
                throw NSError(
                    domain: "Reader",
                    code: 1,
                    userInfo: [
                        NSLocalizedDescriptionKey:
                            "Unsupported format (not EPUB)."
                    ]
                )
            }

            let initial = loadLastLocation()
            let resources = Bundle.main.resourceURL!  // Bundle root
            let navigator = try EPUBNavigatorViewController(
                publication: pub,
                initialLocation: initial,
                config: .init(
                    fontFamilyDeclarations: [
                        CSSFontFamilyDeclaration(
                            fontFamily: .notoSerifSC,
                            fontFaces: [
                                CSSFontFace(
                                    file: FileURL(
                                        url: resources.appendingPathComponent(
                                            "NotoSerifSC-VariableFont_wght.ttf"
                                        )
                                    )!,
                                    style: .normal,
                                    weight: .variable(200...900)
                                )
                            ]
                        ).eraseToAnyHTMLFontFamilyDeclaration()
                    ]
                ),
                httpServer: httpServer
            )
            navigator.delegate = self
            self.navigatorVC = navigator
            
            // Bind interaction manager to navigator
            interactionManager.bind(to: navigator)
            interactionManager.onWordHit = { [weak self] hit in
                guard let self else { return }
                self.currentWordHit = hit
                updateDictionaryResult(for: hit?.word)
            }
        } catch {
            self.openError = error
        }

        isOpening = false
    }

    // MARK: Navigation helpers you can call from SwiftUI buttons
    func go(to link: RLink) async {
//        closeDictionary()
        await navigatorVC?.go(to: link)
    }

    func go(to locator: Locator) async {
//        closeDictionary()
        await navigatorVC?.go(to: locator)
    }

    // MARK: Persistence for last location
    private func saveLastLocation(_ locator: Locator) {
        guard let id = bookId else { return }
        // Throttle writes a bit (avoid writing several times per second)
        let now = Date()
        guard now.timeIntervalSince(lastSavedAt) > 0.5 else { return }
        lastSavedAt = now

        let key = "lastLocation.\(id.uuidString)"
        UserDefaults.standard.set(locator.jsonString, forKey: key)
    }

    private func loadLastLocation() -> Locator? {
        guard let id = bookId else { return nil }
        let key = "lastLocation.\(id.uuidString)"
        guard let s = UserDefaults.standard.string(forKey: key) else {
            return nil
        }
        return try? Locator(jsonString: s)
    }
}

// MARK: - NavigatorDelegate
extension ReadiumEngine: EPUBNavigatorDelegate {
    func navigator(
        _ navigator: any ReadiumNavigator.Navigator,
        presentError error: ReadiumNavigator.NavigatorError
    ) {
    }

    func navigator(_ navigator: Navigator, locationDidChange locator: Locator) {
        closeDictionaryAndClearHighlight()
        
        currentLocation = locator
        saveLastLocation(locator)
        
        // Re-inject helpers for new spine doc and re-apply the selection toggle
        interactionManager.reapplyAfterNavigation()
    }

    func apply(_ s: ReaderSettings, _ systemColorScheme: ColorScheme) {
        guard let nav = navigatorVC else { return }

        // 1. Create a preferences editor.
        let preferences = EPUBPreferences(
            publisherStyles: false
        )
        let editor = nav.editor(of: preferences)

        // 2. Modify the preferences through the editor.
        switch s.font {
        case .notoSerifSC:
            editor.fontFamily.set(.notoSerifSC)
        case .pingFangSC:
            editor.fontFamily.set(.pingFangSC)
        }

        switch s.theme {
        case .light:
            editor.theme.set(.light)
        case .dark:
            editor.theme.set(.dark)
        case .sepia:
            editor.theme.set(.sepia)
        case .system:
            editor.theme.set(systemColorScheme == .light ? .light : .dark)
        }
        
        switch s.interactionMode {
        case .systemSelection:
            interactionManager.setMode(.systemSelection)
        case .customMagnifier:
            interactionManager.setMode(.customMagnifier)
        }

        // The following prefs can potentially cause a reflow of the document.
        // TODO: save location before reflow and restore it at next "layout finished" callback.
        editor.fontSize.set(s.fontSize)
        editor.lineHeight.set(s.lineHeight)
        editor.pageMargins.set(s.margins)
        editor.textAlign.set(
            s.justify ? TextAlignment.justify : TextAlignment.start
        )

        // 3. Submit the edited preferences.
        nav.submitPreferences(editor.preferences)
    }

    // Use this for reliable and link-friendly tapping.
    func installInputObservers(
        onSingleTap: @escaping () -> Void
    ) {
        guard let nav = navigatorVC else { return }

        // Single tap anywhere
        nav.addObserver(
            .tap { [weak self] event in
                guard let self else { return false }

                // If a long-press just ended, swallow this tap
                if self.interactionManager.consumeSuppressedTap() {
                    // Return true to mark the event as handled and
                    // prevent further tap listeners from firing.
                    return true
                }

                // Normal single-tap: let the caller toggle chrome, etc.
                onSingleTap()

                // Return false so Readium can still deliver the tap
                // to links/images inside the page.
                return false
            }
        )
    }
}

/// This is to fix the stubborn top and bottom margins in the EPUB navigator
extension ReadiumEngine {
    func tightenVerticalMargins() {
        guard let root = self.navigatorVC?.view else { return }
        removeVerticalInsets(in: root)
        root.layoutIfNeeded()   // let Auto Layout apply updated constants
    }

    private func removeVerticalInsets(in view: UIView) {
        // Look for each spread view
        let typeName = String(describing: type(of: view))
        if typeName == "PaginationView" || typeName == "EPUBReflowableSpreadView" {
            fixSpreadConstraints(view)
        }

        view.subviews.forEach(removeVerticalInsets)
    }

    private func fixSpreadConstraints(_ spreadView: UIView) {
        // First find the 'WebView' child.
        guard let webView = spreadView.subviews.first(where: {
            String(describing: type(of: $0)) == "WebView"
        }) else { return }

        // Adjust constraints on the spread itself that involve webView's top/bottom.
        for constraint in spreadView.constraints {
            let firstIsWeb = constraint.firstItem as AnyObject === webView
            let secondIsWeb = constraint.secondItem as AnyObject === webView

            if firstIsWeb || secondIsWeb {
                switch (constraint.firstAttribute, constraint.secondAttribute) {
                case (.top, _), (_, .top),
                     (.bottom, _), (_, .bottom):
                    // Kill the 62pt constants
                    constraint.constant = 0
                default:
                    break
                }
            }
        }
    }
}
