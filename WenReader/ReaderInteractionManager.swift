// Copyright 2025 Oliver Zhang
// Licensed under the MIT License

import Foundation
import ReadiumNavigator
import UIKit
import WebKit

/// Coordinates reader interaction components (gesture handling, JS injection, etc.)
/// This is a lightweight coordinator that delegates to specialized handlers
@MainActor
final class ReaderInteractionManager {
    enum Mode {
        case systemSelection
        case customMagnifier
    }
    
    // MARK: - Properties
    
    private weak var navigatorVC: EPUBNavigatorViewController?
    private var currentMode: Mode = .systemSelection
    
    // Specialized handlers
    private let injector: WebViewInjector
    private let gestureHandler: WordSelectionGestureHandler
    
    // Callbacks
    var onWordHit: ((WordHit?) -> Void)? {
        didSet {
            gestureHandler.onWordHit = onWordHit
        }
    }
    var onClickMiss: (() -> Void)? {
        didSet {
            gestureHandler.onClickMiss = onClickMiss
        }
    }
    
    // MARK: - Initialization
    
    init() {
        // Initialize injector with bundled resources
        guard let injector = WebViewInjector(
            javascriptResource: "reader_inject",
            cssResource: "reader_inject"
        ) else {
            Log.error("CRITICAL: Failed to initialize WebViewInjector")
            fatalError("Required injection resources not found")
        }
        
        self.injector = injector
        self.gestureHandler = WordSelectionGestureHandler()
        
        // Setup gesture handler callbacks
        self.gestureHandler.onScrollingStateChange = { [weak self] enabled in
            self?.setScrollingEnabled(enabled)
        }
    }
    
    // MARK: - Public API
    
    func bind(to navigatorVC: EPUBNavigatorViewController) {
        self.navigatorVC = navigatorVC
        
        // Install gesture handler
        gestureHandler.install(on: navigatorVC)
        
        // Inject helpers and apply current mode
        injectHelpersIntoAllVisibleWebViews()
        applyMode(currentMode)
    }
    
    func setMode(_ mode: Mode) {
        currentMode = mode
        applyMode(mode)
    }
    
    /// Call this from navigator locationDidChange to re-apply on new pages
    func reapplyAfterNavigation() {
        injectHelpersIntoAllVisibleWebViews()
        applyMode(currentMode)
    }
    
    /// Call this from engine to remove highlighted word
    func clearHighlight() {
        guard let root = navigatorVC?.view else { return }
        let webViews = ViewHierarchyHelper.findWebViews(in: root)
        injector.clearHighlight(in: webViews)
        
        // Reset gesture handler cache so same word can be looked up again
        gestureHandler.resetWordHitCache()
    }
    
    /// Returns true if a long-press ended very recently and this tap should be suppressed
    func consumeSuppressedTap() -> Bool {
        return gestureHandler.consumeSuppressedTap()
    }

    /// Update ML-based segmentation enabled state.
    func setCwsEnabled(_ enabled: Bool) {
        gestureHandler.cwsEnabled = enabled
    }
    
    // MARK: - Word Navigation
    
    /// Navigate to the previous word.
    func navigateWord(_ direction: WordSelectionGestureHandler.Direction) async -> WordHit? {
        return await gestureHandler.navigateWord(direction)
    }
    
    /// Whether the last navigation placed the highlight off-screen.
    var highlightIsOffScreen: Bool {
        gestureHandler.highlightIsOffScreen
    }
    
    /// Expand the current selection by one character to the right.
    func expandRight() async -> WordHit? {
        return await gestureHandler.expandRight()
    }
    
    /// Shrink the current selection by one character from the right.
    func shrinkRight() async -> WordHit? {
        return await gestureHandler.shrinkRight()
    }
    
    // MARK: - Private Helpers
    
    private func applyMode(_ mode: Mode) {
        guard let root = navigatorVC?.view else { return }
        let webViews = ViewHierarchyHelper.findWebViews(in: root)
        
        switch mode {
        case .systemSelection:
            injector.setSelectionEnabled(true, in: webViews)
            gestureHandler.setEnabled(false)
            
        case .customMagnifier:
            injector.setSelectionEnabled(false, in: webViews)
            gestureHandler.setEnabled(true)
        }
    }
    
    private func injectHelpersIntoAllVisibleWebViews() {
        guard let root = navigatorVC?.view else { return }
        let webViews = ViewHierarchyHelper.findWebViews(in: root)
        injector.injectHelpers(into: webViews)
    }
    
    private func setScrollingEnabled(_ enabled: Bool) {
        guard let root = navigatorVC?.view else { return }
        ViewHierarchyHelper.setScrollingEnabled(enabled, in: root)
    }
}
