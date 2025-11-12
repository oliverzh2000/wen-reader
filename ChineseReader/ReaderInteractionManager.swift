//
//  ReaderInteractionManager.swift
//  ChineseReader
//
//  Created by Oliver Zhang on 2025-11-12.
//

import Foundation
import UIKit
import WebKit
import ReadiumNavigator

/// Owns: selection toggle JS, long-press gesture, and reapplication on chapter changes.
@MainActor
final class ReaderInteractionManager: NSObject, UIGestureRecognizerDelegate {

    enum Mode {
        case systemSelection
        case customMagnifier
    }

    private weak var navigatorVC: EPUBNavigatorViewController?
    private var longPress: UILongPressGestureRecognizer?
    private var currentMode: Mode = .systemSelection
    private var isMagnifierActive = false

    // Cache JS/CSS payloads from bundle
    private let injectJS: String
    private let injectCSS: String

    override init() {
        // Load the files once (fail-quiet with empty string if missing)
        self.injectJS = (try? ReaderInteractionManager.loadBundledText(named: "reader_inject", ext: "js")) ?? ""
        self.injectCSS = (try? ReaderInteractionManager.loadBundledText(named: "reader_inject", ext: "css")) ?? ""
        super.init()
    }

    static private func loadBundledText(named: String, ext: String) throws -> String {
        guard let url = Bundle.main.url(forResource: named, withExtension: ext) else {
            throw NSError(domain: "ReaderInteractionManager", code: 1, userInfo: [NSLocalizedDescriptionKey: "Missing \(named).\(ext) in bundle"])
        }
        return try String(contentsOf: url, encoding: .utf8)
    }

    // MARK: public API
    func bind(to navigatorVC: EPUBNavigatorViewController) {
        self.navigatorVC = navigatorVC
        installLongPress(on: navigatorVC.view)

        // Inject our helper JS+CSS into the currently visible web view (and re-apply on chapter change)
        // We do document-end injection by evaluating scripts after load.
        // If you later move to WKUserScript(.atDocumentStart), wire that at navigator creation time.
        injectHelpersIntoAllVisibleWebViews()
        applyMode(currentMode) // in case bind happens after a mode is already chosen
    }

    func setMode(_ mode: Mode) {
        currentMode = mode
        applyMode(mode)
    }

    /// Call this from navigator locationDidChange to re-apply toggles on new spine docs.
    func reapplyAfterNavigation() {
        injectHelpersIntoAllVisibleWebViews()
        applyMode(currentMode)
    }

    // MARK: selection toggling

    private func applyMode(_ mode: Mode) {
        switch mode {
        case .systemSelection:
            enableSystemSelection()
            longPress?.isEnabled = false
        case .customMagnifier:
            disableSystemSelection()
            longPress?.isEnabled = true
        }
    }

    private func enableSystemSelection() {
        evalInAllWebViews("""
          try { window.CR && window.CR.setSelectable(true); } catch(e) { /* noop */ }
        """)
    }

    private func disableSystemSelection() {
        evalInAllWebViews("""
          try { window.CR && window.CR.setSelectable(false); } catch(e) { /* noop */ }
        """)
    }

    private func injectHelpersIntoAllVisibleWebViews() {
        // 1) Inject CSS (once per doc) by appending a <style> tag
        let escapedCSS = injectCSS
            .replacingOccurrences(of: "\\", with: "\\\\")
            .replacingOccurrences(of: "`", with: "\\`")
            .replacingOccurrences(of: "$", with: "\\$")
            .replacingOccurrences(of: "\n", with: "\\n")

        let cssJS = """
        (function(){
          try {
            if (!document.getElementById('cr-nonselectable-style')) {
              const s = document.createElement('style'); s.id='cr-nonselectable-style'; s.type='text/css';
              s.appendChild(document.createTextNode(`\(escapedCSS)`));
              document.head.appendChild(s);
            }
          } catch(e) {}
        })();
        """

        // 2) Inject the helper JS namespace (window.CR)
        let escapedJS = injectJS
            .replacingOccurrences(of: "\\", with: "\\\\")
            .replacingOccurrences(of: "`", with: "\\`")
            .replacingOccurrences(of: "$", with: "\\$")
            .replacingOccurrences(of: "\n", with: "\\n")

        let helperJS = """
        (function(){
          try {
            const script = document.createElement('script');
            script.type = 'text/javascript';
            script.appendChild(document.createTextNode(`\(escapedJS)`));
            document.head.appendChild(script);
          } catch(e) {}
        })();
        """

        evalInAllWebViews(cssJS)
        evalInAllWebViews(helperJS)
    }

    // MARK: helpers
    private func installLongPress(on view: UIView) {
        let lp = UILongPressGestureRecognizer(target: self, action: #selector(handleLongPress(_:)))
        lp.minimumPressDuration = 0.2  // keeps quick horizontal flicks for page turns
        lp.cancelsTouchesInView = false // don't steal the pan
        lp.delegate = self
        view.addGestureRecognizer(lp)
        self.longPress = lp
        lp.isEnabled = (currentMode == .customMagnifier)
    }

    @objc private func handleLongPress(_ gr: UILongPressGestureRecognizer) {
        guard let hostView = gr.view, currentMode == .customMagnifier else { return }
        let p = gr.location(in: hostView)

        switch gr.state {
        case .began:
            isMagnifierActive = true
            setScrollingEnabled(false)   // <— temporarily disable page swipes
            print("Magnifier began at: x=\(p.x), y=\(p.y)")
            // later: show Metal loupe here

        case .changed:
            if isMagnifierActive {
                print("Magnifier moved to: x=\(p.x), y=\(p.y)")
                // later: update Metal loupe position here
            }

        case .ended, .cancelled, .failed:
            if isMagnifierActive {
                isMagnifierActive = false
                setScrollingEnabled(true)  // <— re-enable page swipes
                print("Magnifier ended at: x=\(p.x), y=\(p.y)")
                // later: hide Metal loupe here
            }

        default:
            break
        }
    }

    // Allow coexistence with Readium's own recognizers (paging)
    func gestureRecognizer(_ gestureRecognizer: UIGestureRecognizer,
                           shouldRecognizeSimultaneouslyWith other: UIGestureRecognizer) -> Bool {
        return true
    }

    // MARK: helpers

    private func evalInAllWebViews(_ js: String) {
        for webView in findDescendantWKWebViews() {
            webView.evaluateJavaScript(js, completionHandler: nil)
        }
    }

    private func findDescendantWKWebViews() -> [WKWebView] {
        guard let root = navigatorVC?.view else { return [] }
        var result: [WKWebView] = []
        func walk(_ v: UIView) {
            if let wkv = v as? WKWebView { result.append(wkv) }
            for sub in v.subviews { walk(sub) }
        }
        walk(root)
        return result
    }
    
    // toggle scrolling for any UIScrollView inside the navigator
    private func setScrollingEnabled(_ enabled: Bool) {
        guard let root = navigatorVC?.view else { return }
        func walk(_ v: UIView) {
            if let scroll = v as? UIScrollView {
                scroll.isScrollEnabled = enabled
            }
            for sub in v.subviews { walk(sub) }
        }
        walk(root)
    }
}
