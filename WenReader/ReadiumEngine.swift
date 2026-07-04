// Copyright 2025 Oliver Zhang
// Licensed under the MIT License

import Combine
import ReadiumNavigator
import ReadiumShared
import SwiftUI
import WebKit

/// Orchestrates reader components and manages overall reading experience
/// 
/// State Ownership:
/// - Engine owns: publication, navigator, location, interaction state (currentWordHit)
/// - DictionaryManager owns: dictionary results and navigation stack (exposed via dictionaryManager)
/// - Views observe both Engine and Engine.dictionaryManager for their respective states
@MainActor
final class ReadiumEngine: ObservableObject {
    // MARK: - Published State (owned by Engine)
    @Published var publication: Publication?
    @Published var navigatorVC: EPUBNavigatorViewController?
    @Published var openError: AppError?
    @Published var isOpening: Bool = false
    @Published var currentLocation: Locator?
    @Published var currentWordHit: WordHit?
    /// Saved locator to return to after following an internal link (e.g. footnote)
    @Published var returnLocator: Locator?
    private var isReturning = false
    private var justFollowedLink = false
    /// When true, suppress dictionary dismissal on the next locationDidChange.
    /// Used during programmatic page flips triggered by word navigation.
    private var suppressDismissOnLocationChange = false

    /// Whether auto-advance is currently running.
    @Published var isAutoAdvancing: Bool = false
    private var autoAdvanceTask: Task<Void, Never>?
    
    // MARK: - Managers
    private let publicationManager = ReadiumPublicationManager()
    private let locationManager = ReaderLocationManager()
    private let settingsManager = ReaderSettingsManager()
    private let interactionManager = ReaderInteractionManager()
    
    /// Task for the current dictionary/WSD lookup. Cancelled when a new word is selected.
    private var dictionaryLookupTask: Task<Void, Never>?
    
    /// Dictionary manager exposed for views to observe dictionary state directly
    /// Views can observe this via @ObservedObject or access via engine.dictionaryManager
    let dictionaryManager = ReaderDictionaryManager()
    
    // MARK: - Computed Properties
    
    /// Current reading progression as a 0.0–1.0 value, or nil if unavailable.
    var currentProgression: Double? {
        currentLocation?.locations.totalProgression
    }
    
    var canGoBackInDictionary: Bool {
        dictionaryManager.canGoBack
    }
    
    /// Whether the user can return to a previous location after following an internal link.
    var canReturn: Bool {
        returnLocator != nil
    }
    
    // MARK: - Dictionary Operations
    
    /// Look up a word in the dictionary.
    /// When `sentence` is provided, WSD is used to sort senses by contextual relevance.
    /// `wordOffsetInSentence` disambiguates repeated occurrences of the same word.
    func updateDictionaryResult(for word: String?, sentence: String? = nil, wordOffsetInSentence: Int = 0) async {
        // Bail early if this task was cancelled (new word selected before we started)
        guard !Task.isCancelled else { return }
        await dictionaryManager.lookup(word, sentence: sentence, wordOffsetInSentence: wordOffsetInSentence)
        if word == nil {
            interactionManager.clearHighlight()
        }
    }
    
    func closeDictionaryAndClearHighlight() {
        stopAutoAdvance()
        currentWordHit = nil
        dictionaryManager.clear()
        interactionManager.clearHighlight()
    }
    
    /// Push a new word onto the dictionary stack (for cross-references)
    func pushDictionary(for word: String) async {
        await dictionaryManager.push(word)
    }
    
    func popDictionary() {
        dictionaryManager.pop()
    }
    
    // MARK: - Word Navigation
    
    func navigateWord(_ direction: WordSelectionGestureHandler.Direction) async {
        guard let hit = await interactionManager.navigateWord(direction) else { return }
        currentWordHit = hit

        // If the highlight landed off-screen, page-flip to bring it into view.
        // Use animated: false for an instant transition (no swipe animation).
        if interactionManager.highlightIsOffScreen {
            suppressDismissOnLocationChange = true
            let options = NavigatorGoOptions(animated: false)
            if direction == .next {
                await navigatorVC?.goForward(options: options)
            } else {
                await navigatorVC?.goBackward(options: options)
            }
        }

        await updateDictionaryResult(for: hit.word, sentence: hit.sentence, wordOffsetInSentence: hit.wordOffsetInSentence)
    }
    
    func expandSelection() async {
        guard let hit = await interactionManager.expandRight() else { return }
        currentWordHit = hit
        await updateDictionaryResult(for: hit.word, sentence: hit.sentence, wordOffsetInSentence: hit.wordOffsetInSentence)
    }
    
    func shrinkSelection() async {
        guard let hit = await interactionManager.shrinkRight() else { return }
        currentWordHit = hit
        await updateDictionaryResult(for: hit.word, sentence: hit.sentence, wordOffsetInSentence: hit.wordOffsetInSentence)
    }

    func toggleAutoAdvance(interval: Double = 1.0) {
        if isAutoAdvancing {
            stopAutoAdvance()
        } else {
            startAutoAdvance(interval: interval)
        }
    }

    func startAutoAdvance(interval: Double) {
        guard !isAutoAdvancing else { return }
        isAutoAdvancing = true
        autoAdvanceTask = Task {
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: UInt64(interval * 1_000_000_000))
                guard !Task.isCancelled else { break }
                await self.navigateWord(.next)
            }
            self.isAutoAdvancing = false
        }
    }

    func stopAutoAdvance() {
        autoAdvanceTask?.cancel()
        autoAdvanceTask = nil
        isAutoAdvancing = false
    }

    // MARK: - Open Publication
    
    func open(bookId: UUID, fileURL: URL, sender: UIView?) async {
        guard !isOpening else { return }
        
        locationManager.setBookId(bookId)
        isOpening = true
        openError = nil
        
        let initialLocation = locationManager.loadLocation()
        let result = await publicationManager.open(
            fileURL: fileURL,
            initialLocation: initialLocation,
            sender: sender
        )
        
        switch result {
        case .success(let (pub, navigator)):
            self.publication = pub
            self.navigatorVC = navigator
            navigator.delegate = self
            
            // Bind interaction manager to navigator
            interactionManager.bind(to: navigator)
            interactionManager.onWordHit = { [weak self] hit in
                guard let self else { return }
                self.currentWordHit = hit
                // Cancel any in-flight dictionary/WSD lookup before starting a new one
                self.dictionaryLookupTask?.cancel()
                self.dictionaryLookupTask = Task {
                    await self.updateDictionaryResult(for: hit?.word, sentence: hit?.sentence, wordOffsetInSentence: hit?.wordOffsetInSentence ?? 0)
                }
            }
            
        case .failure(let error):
            self.openError = error
        }
        
        isOpening = false
    }

    // MARK: - Navigation
    
    func go(to link: RLink) async {
        await navigatorVC?.go(to: link)
    }

    func go(to locator: Locator) async {
        await navigatorVC?.go(to: locator)
    }

    /// Navigate back to the saved return locator (e.g. after visiting a footnote)
    func goBackToReturnLocator() async {
        guard let locator = returnLocator else { return }
        isReturning = true
        returnLocator = nil
        await navigatorVC?.go(to: locator)
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
        // If this location change was triggered by a programmatic page flip
        // (from word navigation buttons), don't dismiss the dictionary.
        if suppressDismissOnLocationChange {
            suppressDismissOnLocationChange = false
        } else {
            closeDictionaryAndClearHighlight()
        }
        
        // Clear the return locator when the user navigates away from the
        // footnote page (e.g. swipe/page turn), but NOT when we just arrived
        // at the footnote itself or are returning via goBackToReturnLocator().
        if isReturning {
            isReturning = false
        } else if returnLocator != nil {
            // We already have a return locator set (meaning we followed a link).
            // If this location change is the result of that link navigation,
            // keep it. If it's a subsequent navigation (user swiped away), clear it.
            // We distinguish by checking if we just set it in shouldNavigateToLink.
            if !justFollowedLink {
                withAnimation(.easeInOut) {
                    returnLocator = nil
                }
            }
            justFollowedLink = false
        }
        
        currentLocation = locator
        locationManager.saveLocation(locator)

        interactionManager.reapplyAfterNavigation()
        tightenVerticalMargins()
    }

    func apply(_ settings: ReaderSettings, _ systemColorScheme: ColorScheme) {
        guard let navigator = navigatorVC else { return }
        settingsManager.apply(
            settings,
            systemColorScheme: systemColorScheme,
            to: navigator,
            interactionManager: interactionManager
        )

        // Propagate ML settings
        interactionManager.setCwsEnabled(settings.cwsEnabled)
        WSDServiceFactory.initialize(wsdEnabled: settings.wsdEnabled)
    }

    func navigator(_ navigator: VisualNavigator, shouldNavigateToLink link: ReadiumShared.Link) -> Bool {
        // Save current location so user can return after following the link
        if let locator = currentLocation {
            withAnimation(.easeInOut) {
                returnLocator = locator
            }
            justFollowedLink = true
        }
        return true
    }

    /// Install tap/click handling for the reader.
    /// On iOS, uses Readium's tap observer (coexists with the long-press GR).
    /// On Mac, the click gesture handler handles everything — CJK clicks do
    /// word lookup, non-CJK clicks call `onSingleTap` to dismiss popups/chrome.
    func installInputObservers(
        onSingleTap: @escaping () -> Void
    ) {
        guard let nav = navigatorVC else { return }

        if ProcessInfo.processInfo.isMacCatalystApp {
            interactionManager.onClickMiss = { onSingleTap() }
        } else {
            nav.addObserver(
                .tap { [weak self] event in
                    guard let self else { return false }

                    // If a long-press just ended, swallow this tap
                    if self.interactionManager.consumeSuppressedTap() {
                        return true
                    }

                    onSingleTap()

                    // Return false so Readium can still deliver the tap
                    // to links/images inside the page.
                    return false
                }
            )
        }
    }
}

/// This is to fix the stubborn top and bottom margins in the EPUB navigator.
/// Readium's spread views add ~62pt top/bottom constraints on the web view.
/// We deactivate those and replace with zero-constant constraints at required priority
/// so Readium cannot re-apply them on subsequent layout passes.
extension ReadiumEngine {
    private static let fixedTag = 9999

    func tightenVerticalMargins() {
        guard let root = self.navigatorVC?.view else { return }
        removeVerticalInsets(in: root)
        root.layoutIfNeeded()
    }

    private func removeVerticalInsets(in view: UIView) {
        let typeName = String(describing: type(of: view))
        if typeName == "PaginationView" || typeName == "EPUBReflowableSpreadView" {
            fixSpreadConstraints(view)
        }
        view.subviews.forEach(removeVerticalInsets)
    }

    private func fixSpreadConstraints(_ spreadView: UIView) {
        guard let webView = spreadView.subviews.first(where: {
            String(describing: type(of: $0)) == "WebView"
        }) else { return }

        // Skip if we already fixed this spread view
        guard spreadView.viewWithTag(Self.fixedTag) == nil else { return }

        // Deactivate Readium's top/bottom constraints on the web view
        var toDeactivate: [NSLayoutConstraint] = []
        for constraint in spreadView.constraints {
            let firstIsWeb = constraint.firstItem as AnyObject === webView
            let secondIsWeb = constraint.secondItem as AnyObject === webView
            guard firstIsWeb || secondIsWeb else { continue }

            switch (constraint.firstAttribute, constraint.secondAttribute) {
            case (.top, _), (_, .top),
                 (.bottom, _), (_, .bottom):
                toDeactivate.append(constraint)
            default:
                break
            }
        }

        NSLayoutConstraint.deactivate(toDeactivate)

        // Add our own zero-inset constraints
        let top = webView.topAnchor.constraint(equalTo: spreadView.topAnchor)
        let bottom = webView.bottomAnchor.constraint(equalTo: spreadView.bottomAnchor)
        top.priority = .required
        bottom.priority = .required
        NSLayoutConstraint.activate([top, bottom])

        // Mark as fixed with an invisible tag view
        let marker = UIView()
        marker.tag = Self.fixedTag
        marker.isHidden = true
        spreadView.addSubview(marker)
    }
}
