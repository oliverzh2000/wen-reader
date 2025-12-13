// Copyright 2025 Oliver Zhang
// Licensed under the MIT License

import Foundation
import UIKit
import WebKit
import ReadiumNavigator

/// Handles long press gestures for word lookup and segmentation
@MainActor
final class LongPressGestureHandler: NSObject, UIGestureRecognizerDelegate {
    
    // MARK: - Properties
    
    private weak var navigatorVC: EPUBNavigatorViewController?
    private var longPress: UILongPressGestureRecognizer?
    private var isMagnifierActive = false
    private var longPressEndTime: CFTimeInterval = 0
    private var currentWordHit: WordHit?
    private var currentLongPressTask: Task<Void, Never>?
    
    private let impactFeedback = UIImpactFeedbackGenerator(style: .medium)
    
    // Callbacks
    var onWordHit: ((WordHit?) -> Void)?
    var onScrollingStateChange: ((Bool) -> Void)?
    
    // MARK: - Setup
    
    func install(on navigatorVC: EPUBNavigatorViewController) {
        self.navigatorVC = navigatorVC
        
        let lp = UILongPressGestureRecognizer(
            target: self,
            action: #selector(handleLongPress(_:))
        )
        lp.minimumPressDuration = ReaderConstants.Interaction.longPressDuration
        lp.cancelsTouchesInView = false
        lp.delegate = self
        navigatorVC.view.addGestureRecognizer(lp)
        
        self.longPress = lp
    }
    
    func setEnabled(_ enabled: Bool) {
        longPress?.isEnabled = enabled
    }
    
    /// Reset cached word hit state (call when dictionary is closed)
    func resetWordHitCache() {
        currentWordHit = nil
    }
    
    // MARK: - Gesture Handling
    
    @objc private func handleLongPress(_ gr: UILongPressGestureRecognizer) {
        guard let hostView = gr.view else { return }
        
        let pInHost = gr.location(in: hostView)
        
        // Convert to navigator root coords
        let rootPoint: CGPoint
        if let root = navigatorVC?.view, hostView !== root {
            rootPoint = hostView.convert(pInHost, to: root)
        } else {
            rootPoint = pInHost
        }
        
        switch gr.state {
        case .began:
            isMagnifierActive = true
            onScrollingStateChange?(false) // Disable scrolling
            handleLongPress(at: rootPoint)
            
        case .changed:
            handleLongPress(at: rootPoint)
            
        case .ended, .cancelled, .failed:
            if isMagnifierActive {
                isMagnifierActive = false
                onScrollingStateChange?(true) // Re-enable scrolling
                
                // Only send nil word hits on finger lift if no word is selected
                if currentWordHit == nil {
                    onWordHit?(nil)
                }
                
                // Mark that a long press just finished
                longPressEndTime = CACurrentMediaTime()
            }
            
        default:
            break
        }
    }
    
    // MARK: - UIGestureRecognizerDelegate
    
    func gestureRecognizer(
        _ gestureRecognizer: UIGestureRecognizer,
        shouldRecognizeSimultaneouslyWith other: UIGestureRecognizer
    ) -> Bool {
        return true
    }
    
    // MARK: - Tap Suppression
    
    /// Returns true if a long-press ended very recently and this tap should be suppressed
    func consumeSuppressedTap(
        threshold: CFTimeInterval = ReaderConstants.Interaction.tapSuppressionWindow
    ) -> Bool {
        let now = CACurrentMediaTime()
        if now - longPressEndTime < threshold {
            longPressEndTime = 0
            return true
        }
        return false
    }
    
    // MARK: - Word Hit Processing
    
    private func handleLongPress(at rootPoint: CGPoint) {
        // Cancel any previous operation
        currentLongPressTask?.cancel()
        
        currentLongPressTask = Task {
            guard !Task.isCancelled,
                  let root = navigatorVC?.view else { return }
            
            let webViews = ViewHierarchyHelper.findWebViews(in: root)
            
            fetchContext(at: rootPoint, from: webViews) { [weak self] block, sentence, run in
                guard let self else { return }
                
                Task {
                    if run == "" {
                        self.currentWordHit = nil
                        return
                    }
                    
                    // Segment the text
                    let segmenter = CedictSegmentationService(
                        dict: CedictSqlService.shared,
                        maxWordLength: ReaderConstants.Segmentation.maxWordLength
                    )
                    let segmentLengths = await segmenter.segment(run).map { $0.count }
                    
                    guard !Task.isCancelled else { return }
                    
                    // Highlight and get word
                    self.segmentAndHighlight(
                        at: rootPoint,
                        lengths: segmentLengths,
                        in: webViews
                    ) { [weak self] word, rects in
                        guard let self else { return }
                        
                        // Only notify if the word changed
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
    
    // MARK: - WebView Communication
    
    private func fetchContext(
        at rootPoint: CGPoint,
        from webViews: [WKWebView],
        completion: @escaping (_ block: String, _ sentence: String, _ run: String) -> Void
    ) {
        var completionCalled = false
        
        for webView in webViews {
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
        in webViews: [WKWebView],
        completion: @escaping (_ word: String, _ rects: [CGRect]) -> Void
    ) {
        var completionCalled = false
        
        for webView in webViews {
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
    
    private func convertToWebViewCoordinates(_ point: CGPoint, webView: WKWebView) -> CGPoint {
        guard let root = navigatorVC?.view else { return point }
        return root.convert(point, to: webView)
    }
}
