//
//  ReadiumEngine.swift
//  ChineseReader
//
//  Created by Oliver Zhang on 2025-11-08.
//

import Combine
import ReadiumNavigator
import ReadiumShared
import SwiftUI
import WebKit

/// Orchestrates reader components and manages overall reading experience
@MainActor
final class ReadiumEngine: ObservableObject {
    // MARK: - Published State
    @Published var publication: Publication?
    @Published var navigatorVC: EPUBNavigatorViewController?
    @Published var openError: AppError?
    @Published var isOpening: Bool = false
    @Published var currentLocation: Locator?
    @Published var currentWordHit: WordHit?
    @Published var currentDictResult: DictionaryResult?
    
    // MARK: - Managers
    private let publicationManager = ReadiumPublicationManager()
    private let locationManager = ReaderLocationManager()
    private let settingsManager = ReaderSettingsManager()
    private let dictionaryManager = ReaderDictionaryManager()
    private let interactionManager = ReaderInteractionManager()
    
    // MARK: - Initialization
    
    init() {
        setupDictionaryObservers()
    }
    
    private func setupDictionaryObservers() {
        // Forward dictionary manager's published values to our own @Published properties
        // This ensures SwiftUI gets notified when dictionary state changes
        dictionaryManager.$currentResult
            .assign(to: &$currentDictResult)
        
        dictionaryManager.$currentWordHit
            .assign(to: &$currentWordHit)
    }
    
    // MARK: - Computed Properties
    
    var canGoBackInDictionary: Bool {
        dictionaryManager.canGoBack
    }
    
    // MARK: - Dictionary Operations
    
    /// Look up a word in the dictionary
    func updateDictionaryResult(for word: String?) async {
        await dictionaryManager.lookup(word)
        if word == nil {
            interactionManager.clearHighlight()
        }
    }
    
    func closeDictionaryAndClearHighlight() {
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
                // Task wrapper at sync/async boundary (closure callback)
                Task {
                    await self.updateDictionaryResult(for: hit?.word)
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
        locationManager.saveLocation(locator)
        
        interactionManager.reapplyAfterNavigation()
    }

    func apply(_ settings: ReaderSettings, _ systemColorScheme: ColorScheme) {
        guard let navigator = navigatorVC else { return }
        settingsManager.apply(
            settings,
            systemColorScheme: systemColorScheme,
            to: navigator,
            interactionManager: interactionManager
        )
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
