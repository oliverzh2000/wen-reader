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

    // Cache JS/CSS payloads from bundle once at initialization for reuse across multiple injections
    // These are loaded once and reused to avoid repeated disk I/O on every chapter change
    private let injectJS: String
    private let injectCSS: String

    // Keep track of finger-up event from long press, and to suppress tap events due to this.
    private var longPressEndTime: CFTimeInterval = 0

    // Keep track of wordHit and only forward to engine once it changes.
    // This also enables haptic impact on each new highlighted word
    private var currentWordHit: WordHit?

    // Callbacks
    var onWordHit: ((WordHit?) -> Void)?

    // Haptics used on magnifier start and user dragging to new word.
    private let impactFeedback = UIImpactFeedbackGenerator(style: .medium)
    
    // Task cancellation for async operations
    private var currentLongPressTask: Task<Void, Never>?

    override init() {
        // Load required injection files from bundle
        do {
            self.injectJS = try ReaderInteractionManager.loadBundledText(
                named: "reader_inject",
                ext: "js"
            )
        } catch {
            Log.error("CRITICAL: Failed to load reader_inject.js: \(error)")
            self.injectJS = ""
        }
        
        do {
            self.injectCSS = try ReaderInteractionManager.loadBundledText(
                named: "reader_inject",
                ext: "css"
            )
        } catch {
            Log.error("CRITICAL: Failed to load reader_inject.css: \(error)")
            self.injectCSS = ""
        }
        
        super.init()
        
        // Validate that critical resources were loaded
        if injectJS.isEmpty || injectCSS.isEmpty {
            Log.error("CRITICAL: Reader interaction manager initialized without required injection files. Custom segmentation/dictionary will not function.")
        }
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

    /// Call this from engine to remove highlighted word.
    func clearHighlight() {
        let js = """
            (function() {
              try {
                if (window.CR && window.CR.clearHighlight) {
                  return window.CR.clearHighlight();
                }
              } catch (e) {
                console.error("CR.clearHighlight error", e);
              }
              return null;
            })();
            """
        evalInAllWebViews(js)

        // Reset this, so that pressing on same word again will still forward to engine.
        currentWordHit = nil
    }

    // MARK: selection toggling
    private func applyMode(_ mode: Mode) {
        switch mode {
        case .systemSelection:
            setSystemSelection(enabled: true)
            longPress?.isEnabled = false
        case .customMagnifier:
            setSystemSelection(enabled: false)
            longPress?.isEnabled = true
        }
    }

    private func setSystemSelection(enabled: Bool) {
        evalInAllWebViews(
            """
              try { window.CR && window.CR.setSelectable(\(enabled ? "true" : "false")); } catch(e) { /* noop */ }
            """
        )
    }

    private func injectHelpersIntoAllVisibleWebViews() {
        // 1) Inject CSS (once per doc) by appending a <style> tag
        guard let cssJSON = jsonEncode(injectCSS) else {
            Log.error("Failed to encode CSS for injection")
            return
        }

        let cssJS = """
            (function(){
              try {
                if (!document.getElementById('cr-nonselectable-style')) {
                  const s = document.createElement('style');
                  s.id='cr-nonselectable-style';
                  s.type='text/css';
                  s.appendChild(document.createTextNode(\(cssJSON)));
                  document.head.appendChild(s);
                }
              } catch(e) {
                console.error('CR CSS injection failed:', e);
              }
            })();
            """

        // 2) Inject the helper JS namespace (window.CR)
        guard let jsJSON = jsonEncode(injectJS) else {
            Log.error("Failed to encode JS for injection")
            return
        }

        let helperJS = """
            (function(){
              try {
                const script = document.createElement('script');
                script.type = 'text/javascript';
                script.appendChild(document.createTextNode(\(jsJSON)));
                document.head.appendChild(script);
              } catch(e) {
                console.error('CR JS injection failed:', e);
              }
            })();
            """

        evalInAllWebViews(cssJS)
        evalInAllWebViews(helperJS)
    }

    // MARK: helpers
    private func installLongPress(on view: UIView) {
        let lp = UILongPressGestureRecognizer(
            target: self,
            action: #selector(handleLongPress(_:))
        )
        lp.minimumPressDuration = ReaderConstants.Interaction.longPressDuration
        lp.cancelsTouchesInView = false  // don't steal the pan
        lp.delegate = self
        view.addGestureRecognizer(lp)
        self.longPress = lp
        lp.isEnabled = (currentMode == .customMagnifier)
    }

    @objc private func handleLongPress(_ gr: UILongPressGestureRecognizer) {
        guard let hostView = gr.view, currentMode == .customMagnifier else {
            return
        }

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
            setScrollingEnabled(false)  // freeze page swipes while magnifier is active
            handleLongPress(at: rootPoint)

        case .changed:
            handleLongPress(at: rootPoint)

        case .ended, .cancelled, .failed:
            if isMagnifierActive {
                isMagnifierActive = false
                setScrollingEnabled(true)  // restore page swipes
                
                // Only send nil word hits to engine on finger lift.
                if self.currentWordHit == nil {
                    onWordHit?(nil)
                }

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
    func consumeSuppressedTap(threshold: CFTimeInterval = ReaderConstants.Interaction.tapSuppressionWindow) -> Bool {
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
        traverseViewHierarchy(root) { view in
            if let webView = view as? WKWebView {
                result.append(webView)
            }
        }
        return result
    }

    // Toggle scrolling for any UIScrollView inside the navigator
    private func setScrollingEnabled(_ enabled: Bool) {
        guard let root = navigatorVC?.view else { return }
        traverseViewHierarchy(root) { view in
            if let scrollView = view as? UIScrollView {
                scrollView.isScrollEnabled = enabled
            }
        }
    }
    
    // Generic view hierarchy traversal
    private func traverseViewHierarchy(_ root: UIView, visitor: (UIView) -> Void) {
        visitor(root)
        for subview in root.subviews {
            traverseViewHierarchy(subview, visitor: visitor)
        }
    }
    
    // Convert point from navigator root coords to webView coords
    private func convertToWebViewCoordinates(_ point: CGPoint, webView: WKWebView) -> CGPoint {
        guard let root = navigatorVC?.view else { return point }
        return root.convert(point, to: webView)
    }
    
    // JSON encode string for safe JavaScript injection
    private func jsonEncode(_ string: String) -> String? {
        guard let data = try? JSONEncoder().encode(string),
              let json = String(data: data, encoding: .utf8) else {
            return nil
        }
        return json
    }

    private func handleLongPress(at rootPoint: CGPoint) {
        // Cancel any previous long press operation
        currentLongPressTask?.cancel()
        
        currentLongPressTask = Task {
            guard !Task.isCancelled else { return }
            
            fetchContext(at: rootPoint) { [weak self] block, sentence, run in
                guard let self else { return }
                
                Task {
                    if run == "" {
                        self.currentWordHit = nil
                        return
                    }
                    
                    let segmenter = CedictSegmentationService(
                        dict: CedictSqlService.shared,
                        maxWordLength: ReaderConstants.Segmentation.maxWordLength
                    )
                    let segmentLengths = await segmenter.segment(run).map {
                        $0.count
                    }
                    
                    guard !Task.isCancelled else { return }
                    
                    self.segmentAndHighlight(at: rootPoint, lengths: segmentLengths)
                    { [weak self] word, rects in
                        guard let self else { return }
                        if rects != self.currentWordHit?.rects {
                            let wordHit = WordHit(
                                block: block,
                                sentence: sentence,
                                run: run,
                                word: word,
                                hitPoint: rootPoint,
                                rects: rects
                            )
                            self.currentWordHit = wordHit
                            if self.currentWordHit != nil {
                                self.onWordHit?(self.currentWordHit)
                                self.impactFeedback.impactOccurred()
                            }
                        }
                    }
                }
            }
        }
    }

    private func fetchContext(
        at rootPoint: CGPoint,
        completion:
            @escaping (_ block: String, _ sentence: String, _ run: String) ->
            Void
    ) {
        var completionCalled = false
        
        for webView in findDescendantWKWebViews() {
            let local = convertToWebViewCoordinates(rootPoint, webView: webView)

            let js = """
                (function() {
                  try {
                    if (window.CR && window.CR.getContextAtPoint) {
                      return window.CR.getContextAtPoint(\(local.x), \(local.y));
                    }
                  } catch (e) {
                    console.error("CR.getContextAtPoint error", e);
                  }
                  return null;
                })();
                """

            webView.evaluateJavaScript(js) { result, error in
                guard
                    !completionCalled,
                    error == nil,
                    let dict = result as? [String: Any],
                    let block = dict["block"] as? String,
                    let sentence = dict["sentence"] as? String,
                    let run = dict["run"] as? String
                else {
                    return
                }
                completionCalled = true
                completion(block, sentence, run)
            }
        }
    }

    private func segmentAndHighlight(
        at rootPoint: CGPoint,
        lengths: [Int],
        completion: @escaping (_ word: String, _ rects: [CGRect]) -> Void
    ) {
        var completionCalled = false
        
        for webView in findDescendantWKWebViews() {
            let local = convertToWebViewCoordinates(rootPoint, webView: webView)

            // Serialize lengths to JS array
            let lengthsJSON: String
            do {
                let data = try JSONSerialization.data(withJSONObject: lengths)
                if let jsonString = String(data: data, encoding: .utf8) {
                    lengthsJSON = jsonString
                } else {
                    Log.error("Failed to convert lengths data to UTF-8 string")
                    lengthsJSON = "[]"
                }
            } catch {
                Log.error("Failed to serialize segmentation lengths to JSON: \(error)")
                lengthsJSON = "[]"
            }

            let js = """
                (function() {
                  try {
                    if (window.CR && window.CR.segmentAndHighlightAtPoint) {
                      return window.CR.segmentAndHighlightAtPoint(\(local.x), \(local.y), \(lengthsJSON));
                    }
                  } catch (e) {
                    console.error("CR.segmentAndHighlightAtPoint error", e);
                  }
                  return null;
                })();
                """

            webView.evaluateJavaScript(js) { result, error in
                guard
                    !completionCalled,
                    error == nil,
                    let dict = result as? [String: Any],
                    let word = dict["word"] as? String,
                    let rectArray = dict["rects"] as? [[String: Any]]
                else {
                    return
                }

                let rects: [CGRect] = rectArray.compactMap { rd in
                    guard
                        let x = (rd["x"] as? NSNumber)?.doubleValue,
                        let y = (rd["y"] as? NSNumber)?.doubleValue,
                        let w = (rd["width"] as? NSNumber)?.doubleValue,
                        let h = (rd["height"] as? NSNumber)?.doubleValue
                    else {
                        return nil
                    }
                    return CGRect(x: x, y: y, width: w, height: h)
                }
                completionCalled = true
                completion(word, rects)
            }
        }
    }
}
