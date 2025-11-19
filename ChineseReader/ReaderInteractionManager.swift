//
//  ReaderInteractionManager.swift
//  ChineseReader
//
//  Created by Oliver Zhang on 2025-11-12.
//

import Foundation
import ReadiumNavigator
import UIKit
import WebKit

// Payload returned from JS side.
struct WordHit {
    let sentenceTokens: [String]
    let wordIndex: Int
    let rectsInWebView: [CGRect]

    var word: String? {
        guard sentenceTokens.indices.contains(wordIndex) else { return nil }
        return sentenceTokens[wordIndex]
    }

    var sentenceJoined: String {
        sentenceTokens.joined()
    }
}

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
    // TODO: make local to functions.
    private let jiebaJS: String
    private let injectJS: String
    private let injectCSS: String
    
    // Used to keep track of finger-up event from long press, and to suppress tap events due to this.
    private var longPressEndTime: CFTimeInterval = 0

    // Callbacks
    var onWordHit: ((WordHit) -> Void)?
    
    // Haptics used on magnifier start and user dragging to new word.
    private let impactFeedback = UIImpactFeedbackGenerator(style: .light)

    override init() {
        // Load the files once (fail-quiet with empty string if missing)
        self.jiebaJS =
            (try? ReaderInteractionManager.loadBundledText(
                named: "jieba_rs_wasm_combined",
                ext: "js"
            )) ?? ""
        self.injectJS =
            (try? ReaderInteractionManager.loadBundledText(
                named: "reader_inject",
                ext: "js"
            )) ?? ""
        self.injectCSS =
            (try? ReaderInteractionManager.loadBundledText(
                named: "reader_inject",
                ext: "css"
            )) ?? ""
        super.init()
    }

    static private func loadBundledText(named: String, ext: String) throws
        -> String
    {
        guard let url = Bundle.main.url(forResource: named, withExtension: ext)
        else {
            throw NSError(
                domain: "ReaderInteractionManager",
                code: 1,
                userInfo: [
                    NSLocalizedDescriptionKey:
                        "Missing \(named).\(ext) in bundle"
                ]
            )
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
        applyMode(currentMode)  // in case bind happens after a mode is already chosen
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
        evalInAllWebViews(
            """
              try { window.CR && window.CR.setSelectable(true); } catch(e) { /* noop */ }
            """
        )
    }

    private func disableSystemSelection() {
        evalInAllWebViews(
            """
              try { window.CR && window.CR.setSelectable(false); } catch(e) { /* noop */ }
            """
        )
    }

    private func injectHelpersIntoAllVisibleWebViews() {
        // 1) Inject CSS (once per doc) by appending a <style> tag
        let escapedCSS =
            injectCSS
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
        
        // 2) Inject Jieba bundle as a <script> tag in the document
        let escapedJieba =
            jiebaJS
            .replacingOccurrences(of: "\\", with: "\\\\")
            .replacingOccurrences(of: "`", with: "\\`")
            .replacingOccurrences(of: "$", with: "\\$")
            .replacingOccurrences(of: "\n", with: "\\n")

        let jiebaScriptJS = """
            (function(){
              try {
                if (!window.Jieba) {
                  const script = document.createElement('script');
                  script.type = 'text/javascript';
                  script.appendChild(document.createTextNode(`\(escapedJieba)`));
                  document.head.appendChild(script);
                }
              } catch(e) {
                console.error("Failed to inject Jieba", e);
              }
            })();
            """

        // 3) Inject the helper JS namespace (window.CR)
        let escapedJS =
            injectJS
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
        evalInAllWebViews(jiebaScriptJS)
        evalInAllWebViews(helperJS)
    }

    // MARK: helpers
    private func installLongPress(on view: UIView) {
        let lp = UILongPressGestureRecognizer(
            target: self,
            action: #selector(handleLongPress(_:))
        )
        lp.minimumPressDuration = 0.2  // keeps quick horizontal flicks for page turns
        lp.cancelsTouchesInView = false  // don't steal the pan
        lp.delegate = self
        view.addGestureRecognizer(lp)
        self.longPress = lp
        lp.isEnabled = (currentMode == .customMagnifier)
    }

    @objc private func handleLongPress(_ gr: UILongPressGestureRecognizer) {
        guard let hostView = gr.view, currentMode == .customMagnifier else { return }

        // Location in the host view (the navigator root)
        let pInHost = gr.location(in: hostView)

        // Convert to navigator root coords (just in case hostView != navigatorVC.view)
        let rootPoint: CGPoint
        if let root = navigatorVC?.view, hostView !== root {
            rootPoint = hostView.convert(pInHost, to: root)
        } else {
            rootPoint = pInHost
        }

        switch gr.state {
        case .began:
            isMagnifierActive = true
            setScrollingEnabled(false)   // freeze page swipes while magnifier is active
            highlightWord(at: rootPoint) // highlight initial word
            impactFeedback.prepare()
            impactFeedback.impactOccurred()

        case .changed:
            if isMagnifierActive {
                highlightWord(at: rootPoint) // follow finger with highlight
            }

        case .ended, .cancelled, .failed:
            if isMagnifierActive {
                isMagnifierActive = false
                setScrollingEnabled(true) // restore page swipes
                // (optional later: send JS to clear highlight when finger lifts)
                
                // Mark that a long press just finished
                longPressEndTime = CACurrentMediaTime()
            }

        default:
            break
        }
    }

    // Allow coexistence with Readium's own recognizers (paging)
    func gestureRecognizer(
        _ gestureRecognizer: UIGestureRecognizer,
        shouldRecognizeSimultaneouslyWith other: UIGestureRecognizer
    ) -> Bool {
        return true
    }
    
    /// Returns true iff a long-press ended very recently.
    func consumeSuppressedTap(threshold: CFTimeInterval = 0.1) -> Bool {
        let now = CACurrentMediaTime()
        if now - longPressEndTime < threshold {
            // Consume exactly one tap
            longPressEndTime = 0
            return true
        }
        return false
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

    //    highlight word under a given point in navigator-root coords
    private func highlightWord(at rootPoint: CGPoint) {
        guard let root = navigatorVC?.view else { return }

        for webView in findDescendantWKWebViews() {
            let local = root.convert(rootPoint, to: webView)
            if webView.bounds.contains(local) {
                let js = String(
                    format: """
                    (function() {
                      try {
                        if (window.CR && window.CR.highlightWordAtPoint) {
                          return window.CR.highlightWordAtPoint(%f, %f);
                        }
                      } catch (e) {
                        console.error("CR.highlightWordAtPoint error", e);
                      }
                      return null;
                    })();
                    """,
                    local.x,
                    local.y
                )

                webView.evaluateJavaScript(js) { result, error in
                    if let error = error {
                        print("JS error in highlightWordAtPoint: \(error)")
                        return
                    }

                    guard
                        let dict = result as? [String: Any],
                        let sentenceTokens = dict["sentenceTokens"] as? [String],
                        let wordIndexNumber = dict["wordIndex"] as? NSNumber
                    else {
                        print("No valid word info returned (result = \(String(describing: result)))")
                        return
                    }

                    let wordIndex = wordIndexNumber.intValue

                    var rects: [CGRect] = []
                    if let rectArray = dict["rects"] as? [[String: Any]] {
                        for rectDict in rectArray {
                            guard
                                let x = (rectDict["x"] as? NSNumber)?.doubleValue,
                                let y = (rectDict["y"] as? NSNumber)?.doubleValue,
                                let width = (rectDict["width"] as? NSNumber)?.doubleValue,
                                let height = (rectDict["height"] as? NSNumber)?.doubleValue
                            else { continue }
                            rects.append(CGRect(x: x, y: y, width: width, height: height))
                        }
                    }

                    let hit = WordHit(
                        sentenceTokens: sentenceTokens,
                        wordIndex: wordIndex,
                        rectsInWebView: rects
                    )
                    self.onWordHit?(hit)
                }
                break
            }
        }
    }

}
