// Copyright 2025 Oliver Zhang
// Licensed under the MIT License

import Foundation
import UIKit
import WebKit
import ReadiumNavigator

/// Handles long press gestures for word lookup and segmentation.
///
/// Architecture:
/// - JS provides: block text + char index at a point, highlight a range, navigate blocks.
/// - All segmentation, sentence/run detection, and word navigation live in Swift.
/// - DOM state: at most one `id="wr-active-block"` + one `<span id="wr-highlight">`.
@MainActor
final class LongPressGestureHandler: NSObject, UIGestureRecognizerDelegate {
    
    // MARK: - Properties
    
    private weak var navigatorVC: EPUBNavigatorViewController?
    private var longPress: UILongPressGestureRecognizer?
    private var isMagnifierActive = false
    private var longPressEndTime: CFTimeInterval = 0
    private var currentLongPressTask: Task<Void, Never>?
    
    /// Current state: what block, what segments, which word is highlighted.
    private var currentBlockText: String?
    private var currentCharIndex: Int?
    private var currentRunRange: Range<Int>?
    private var currentSegments: [String]?
    private var currentWordIndex: Int?
    private var currentWordHit: WordHit?
    /// Whether the current segments have been manually adjusted (expand/shrink).
    /// If true, navigateWord will re-segment fresh before moving.
    private var segmentsManuallyAdjusted = false
    /// Set to true after a navigateWord call leaves the highlight off-screen.
    /// The caller can read this to decide whether to page-flip.
    private(set) var highlightIsOffScreen = false
    
    private let impactFeedback = UIImpactFeedbackGenerator(style: .medium)
    
    // Callbacks
    var onWordHit: ((WordHit?) -> Void)?
    var onScrollingStateChange: ((Bool) -> Void)?
    
    /// Whether ML-based segmentation (CWS) is enabled.
    var cwsEnabled: Bool = true
    
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
    
    /// Reset all cached state (call when dictionary is closed).
    func resetWordHitCache() {
        currentWordHit = nil
        currentBlockText = nil
        currentCharIndex = nil
        currentRunRange = nil
        currentSegments = nil
        currentWordIndex = nil
        segmentsManuallyAdjusted = false
    }
    
    // MARK: - Word Navigation (called from engine/view)
    
    enum Direction { case prev, next }
    
    /// Navigate to the previous or next word.
    /// Walks: within current run → adjacent run in same block → adjacent block.
    /// If segments were manually adjusted (expand/shrink), re-segments the run first.
    func navigateWord(_ direction: Direction) async -> WordHit? {
        guard var segments = currentSegments,
              var wordIdx = currentWordIndex,
              let runRange = currentRunRange,
              let blockText = currentBlockText else { return nil }
        
        // If segments were manually adjusted, re-segment from scratch
        // to get clean word boundaries before navigating
        if segmentsManuallyAdjusted {
            let chars = Array(blockText)
            let runText = String(chars[runRange])
            let service = SegmentationServiceFactory.active(cwsEnabled: cwsEnabled)
            let freshSegments = await service.segment(run: runText.toSimplified, sentence: blockText.toSimplified)
            let originalSegments = mapSegmentsToOriginal(freshSegments, originalRun: runText)
            currentSegments = originalSegments
            segments = originalSegments
            segmentsManuallyAdjusted = false
            
            // Anchor navigation on the character just past the adjusted selection's end
            // (for next) or just before its start (for prev). This makes behavior
            // consistent with sub-split: navigate moves away from the adjusted word.
            let adjustedWord = currentWordHit?.word ?? ""
            let adjustedStartInRun: Int
            if let hit = currentWordHit, let range = runText.range(of: hit.word) {
                adjustedStartInRun = runText.distance(from: runText.startIndex, to: range.lowerBound)
            } else {
                adjustedStartInRun = 0
            }
            
            let anchorCharInRun: Int
            if direction == .next {
                // Start on the word containing the char just past the adjusted selection
                anchorCharInRun = min(adjustedStartInRun + adjustedWord.count, runText.count - 1)
            } else {
                // Start on the word containing the char just before the adjusted selection
                anchorCharInRun = max(adjustedStartInRun - 1, 0)
            }
            
            wordIdx = wordIndexForChar(at: anchorCharInRun, in: originalSegments)
            currentWordIndex = wordIdx
            // Highlight this word directly (don't step further)
            return await highlightWordAtIndex(wordIdx, segments: originalSegments, runRange: runRange, blockText: blockText)
        }
        
        // Step within current run
        let nextIdx = (direction == .next) ? wordIdx + 1 : wordIdx - 1
        if nextIdx >= 0 && nextIdx < segments.count {
            return await highlightWordAtIndex(nextIdx, segments: segments, runRange: runRange, blockText: blockText)
        }
        
        // At edge of run — find adjacent run in this block
        let adjacentRun = (direction == .next)
            ? findNextRun(after: runRange.upperBound, in: blockText)
            : findPreviousRun(before: runRange.lowerBound, in: blockText)
        
        if let run = adjacentRun {
            return await segmentAndHighlightRun(range: run, in: blockText, pickLast: direction == .prev)
        }
        
        // At edge of block — navigate to adjacent block
        let jsDirection = (direction == .next) ? "next" : "prev"
        return await navigateToAdjacentBlock(direction: jsDirection)
    }
    
    /// Expand selection by one character to the right.
    /// If the expanded string isn't in the dictionary, the popup won't show,
    /// but the highlight still updates so the user sees what's selected.
    func expandRight() async -> WordHit? {
        guard var segments = currentSegments,
              let wordIdx = currentWordIndex,
              let runRange = currentRunRange,
              let blockText = currentBlockText else { return nil }
        
        let wordStartInRun = segments[0..<wordIdx].reduce(0) { $0 + $1.count }
        let wordStartInBlock = runRange.lowerBound + wordStartInRun
        let currentWord = segments[wordIdx]
        let wordEndInBlock = wordStartInBlock + currentWord.count
        
        guard wordEndInBlock < blockText.count else { return nil }
        
        let chars = Array(blockText)
        // Don't expand into non-CJK characters
        guard chars[wordEndInBlock].isCJK else { return nil }
        
        let newWord = currentWord + String(chars[wordEndInBlock])
        
        // Update segments
        segments[wordIdx] = newWord
        if wordIdx + 1 < segments.count {
            let nextSeg = segments[wordIdx + 1]
            if nextSeg.count > 1 {
                segments[wordIdx + 1] = String(nextSeg.dropFirst())
            } else {
                segments.remove(at: wordIdx + 1)
            }
        } else {
            // Expanding beyond the run
            currentRunRange = runRange.lowerBound..<(runRange.upperBound + 1)
        }
        currentSegments = segments
        segmentsManuallyAdjusted = true
        
        return await highlightWordAtIndex(wordIdx, segments: segments, runRange: currentRunRange!, blockText: blockText)
    }
    
    /// Shrink selection by one character from the right.
    /// The removed character becomes the start of the next segment.
    func shrinkRight() async -> WordHit? {
        guard var segments = currentSegments,
              let wordIdx = currentWordIndex,
              let runRange = currentRunRange,
              let blockText = currentBlockText else { return nil }
        
        let currentWord = segments[wordIdx]
        guard currentWord.count > 1 else { return nil }
        
        let newWord = String(currentWord.dropLast())
        let removedChar = String(currentWord.suffix(1))
        
        segments[wordIdx] = newWord
        if wordIdx + 1 < segments.count {
            segments[wordIdx + 1] = removedChar + segments[wordIdx + 1]
        } else {
            segments.insert(removedChar, at: wordIdx + 1)
        }
        currentSegments = segments
        segmentsManuallyAdjusted = true
        
        return await highlightWordAtIndex(wordIdx, segments: segments, runRange: runRange, blockText: blockText)
    }
    
    // MARK: - Gesture Handling
    
    @objc private func handleLongPress(_ gr: UILongPressGestureRecognizer) {
        guard let hostView = gr.view else { return }
        
        // Convert gesture point to navigator root coordinates
        let pInHost = gr.location(in: hostView)
        let rootPoint: CGPoint
        if let root = navigatorVC?.view, hostView !== root {
            rootPoint = hostView.convert(pInHost, to: root)
        } else {
            rootPoint = pInHost
        }
        
        switch gr.state {
        case .began:
            isMagnifierActive = true
            onScrollingStateChange?(false) // Disable scrolling during long press
            processLongPress(at: rootPoint, isBegan: true)
            
        case .changed:
            processLongPress(at: rootPoint, isBegan: false)
            
        case .ended, .cancelled, .failed:
            if isMagnifierActive {
                isMagnifierActive = false
                onScrollingStateChange?(true) // Re-enable scrolling
                // Only send nil word hit on finger lift if no word was selected
                if currentWordHit == nil { onWordHit?(nil) }
                // Mark time so we can suppress the spurious tap that follows a long press
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
    ) -> Bool { true }
    
    // MARK: - Tap Suppression
    
    /// Returns true if a long-press ended very recently and this tap should be suppressed.
    /// iOS delivers a spurious UITap immediately after UILongPress ends — this prevents
    /// it from toggling chrome or dismissing the dictionary.
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
    
    // MARK: - Core Processing
    
    /// Main long-press handler. Flow:
    /// 1. Ask JS for block text + char index at the finger position
    /// 2. Check if this is a sub-split (re-press on highlighted word)
    /// 3. Find the CJK run containing the pressed character
    /// 4. Segment the run (or reuse cached segmentation)
    /// 5. Highlight the word under the finger and emit a WordHit
    private func processLongPress(at rootPoint: CGPoint, isBegan: Bool) {
        currentLongPressTask?.cancel()
        
        currentLongPressTask = Task {
            guard !Task.isCancelled,
                  let root = navigatorVC?.view else { return }
            
            let webViews = ViewHierarchyHelper.findWebViews(in: root)
            
            // Phase 1: Get block text and char index from JS
            guard let result = await callJS(
                "getBlockAndCharIndexAtPoint",
                args: rootPoint,
                in: webViews
            ),
            let blockText = result["blockText"] as? String,
            let charIndex = result["charIndex"] as? Int else { return }
            
            guard !Task.isCancelled else { return }
            
            // Check sub-split: new press on already-highlighted word
            if isBegan, let _ = currentWordHit,
               currentBlockText == blockText,
               let segments = currentSegments, let wordIdx = currentWordIndex,
               let runRange = currentRunRange {
                let wordStartInRun = segments[0..<wordIdx].reduce(0) { $0 + $1.count }
                let wordStartInBlock = runRange.lowerBound + wordStartInRun
                let wordEndInBlock = wordStartInBlock + segments[wordIdx].count
                
                if charIndex >= wordStartInBlock && charIndex < wordEndInBlock {
                    performSubSplit(wordIdx: wordIdx, runRange: runRange, blockText: blockText, charIndex: charIndex)
                    return
                }
            }
            
            // Phase 2: Find the CJK run containing this character
            let chars = Array(blockText)
            guard charIndex < chars.count, chars[charIndex].isCJK else { return }
            
            var runStart = charIndex
            var runEnd = charIndex + 1
            while runStart > 0 && chars[runStart - 1].isCJK { runStart -= 1 }
            while runEnd < chars.count && chars[runEnd].isCJK { runEnd += 1 }
            let runRange = runStart..<runEnd
            let runText = String(chars[runStart..<runEnd])
            
            // Phase 3: Segment the run (reuse if same)
            if currentBlockText != blockText || currentRunRange != runRange {
                let service = SegmentationServiceFactory.active(cwsEnabled: cwsEnabled)
                // Convert to simplified for CWS/WSD (models trained on simplified)
                let segments = await service.segment(run: runText.toSimplified, sentence: blockText.toSimplified)
                guard !Task.isCancelled else { return }
                currentBlockText = blockText
                currentRunRange = runRange
                // Map segments back to original characters (trad↔simp is 1:1 char mapping)
                currentSegments = mapSegmentsToOriginal(segments, originalRun: runText)
            }
            
            guard let segments = currentSegments, !segments.isEmpty else { return }
            
            // Phase 4: Find which word the char belongs to
            let wordIdx = wordIndexForChar(at: charIndex - runStart, in: segments)
            let word = segments[wordIdx]
            
            // Skip if dragging within same word
            if !isBegan, word == currentWordHit?.word { return }
            guard SpanScorerSegmentationService.isLookupable(word) else { return }
            
            // Phase 5: Highlight and emit
            if await highlightWordAtIndex(wordIdx, segments: segments, runRange: runRange, blockText: blockText) != nil {
                impactFeedback.impactOccurred()
            }
        }
    }
    
    // MARK: - Sub-splitting
    
    /// Re-segment the currently highlighted word into smaller sub-words.
    /// Triggered when the user long-presses on an already-highlighted word.
    /// Uses the span scorer with the current word masked out, forcing DP
    /// to find the best sub-segmentation.
    private func performSubSplit(
        wordIdx: Int,
        runRange: Range<Int>,
        blockText: String,
        charIndex: Int
    ) {
        guard var segments = currentSegments else { return }
        let word = segments[wordIdx]
        guard word.count > 1, cwsEnabled,
              let spanScorer = SegmentationServiceFactory.spanScorer else { return }
        
        let chars = Array(blockText)
        let runText = String(chars[runRange])
        let wordStartInRun = segments[0..<wordIdx].reduce(0) { $0 + $1.count }
        
        guard let subSegments = spanScorer.resegment(
            originalRun: runText,
            sentence: blockText,
            wordToSplit: word,
            wordStartInRun: wordStartInRun
        ), subSegments.count > 1 || subSegments.first != word else { return }
        
        segments.replaceSubrange(wordIdx...wordIdx, with: subSegments)
        currentSegments = segments
        
        let newWordIdx = wordIndexForChar(at: charIndex - runRange.lowerBound, in: segments)
        
        Task {
            if await highlightWordAtIndex(newWordIdx, segments: segments, runRange: runRange, blockText: blockText) != nil {
                impactFeedback.impactOccurred()
            }
        }
    }
    
    // MARK: - Highlighting
    
    /// Highlight a specific word by index in the current segments array.
    /// Calls JS to render the highlight, then builds and emits a WordHit.
    /// Also checks if the highlight is visible in the viewport and sets `highlightIsOffScreen`.
    @discardableResult
    private func highlightWordAtIndex(
        _ wordIdx: Int,
        segments: [String],
        runRange: Range<Int>,
        blockText: String
    ) async -> WordHit? {
        guard wordIdx >= 0, wordIdx < segments.count else { return nil }
        
        let word = segments[wordIdx]
        let wordStartInRun = segments[0..<wordIdx].reduce(0) { $0 + $1.count }
        let wordStartInBlock = runRange.lowerBound + wordStartInRun
        
        currentWordIndex = wordIdx
        currentSegments = segments
        currentRunRange = runRange
        currentBlockText = blockText
        
        guard let root = navigatorVC?.view else { return nil }
        let webViews = ViewHierarchyHelper.findWebViews(in: root)
        
        let rects = await callJSForRects(
            "highlightRangeInLastBlock",
            args: "\(wordStartInBlock), \(word.count)",
            in: webViews
        )
        
        guard !word.isEmpty else { return nil }
        
        // Check if highlight is visible in the viewport
        highlightIsOffScreen = !(await checkHighlightVisible(in: webViews))
        
        let hitPoint = rects.first.map { CGPoint(x: $0.midX, y: $0.midY) } ?? .zero
        
        let (sentence, wordOffsetInSentence) = extractSentenceWithOffset(
            around: wordStartInBlock, wordLength: word.count, in: blockText
        )
        
        let wordHit = WordHit(
            block: blockText,
            sentence: sentence,
            run: String(Array(blockText)[runRange]),
            word: word,
            hitPoint: hitPoint,
            rects: rects,
            wordOffsetInSentence: wordOffsetInSentence
        )
        
        currentWordHit = wordHit
        onWordHit?(wordHit)
        return wordHit
    }
    
    /// Ask JS whether the highlight span is currently visible in the web view's viewport.
    private func checkHighlightVisible(in webViews: [WKWebView]) async -> Bool {
        guard let result = await callJS("isHighlightVisible", args: "", in: webViews),
              let visible = result["visible"] as? Bool else {
            return true // Assume visible if we can't check
        }
        return visible
    }
    
    // MARK: - Run Navigation
    
    /// Find the next contiguous CJK run starting at or after `offset` in the block.
    private func findNextRun(after offset: Int, in blockText: String) -> Range<Int>? {
        let chars = Array(blockText)
        var i = offset
        while i < chars.count && !chars[i].isCJK { i += 1 }
        guard i < chars.count else { return nil }
        let start = i
        while i < chars.count && chars[i].isCJK { i += 1 }
        return start..<i
    }
    
    /// Find the previous contiguous CJK run ending before `offset` in the block.
    private func findPreviousRun(before offset: Int, in blockText: String) -> Range<Int>? {
        let chars = Array(blockText)
        var i = offset - 1
        while i >= 0 && !chars[i].isCJK { i -= 1 }
        guard i >= 0 else { return nil }
        let end = i + 1
        while i > 0 && chars[i - 1].isCJK { i -= 1 }
        return i..<end
    }
    
    /// Segment a new run and highlight its first or last word.
    /// Used when navigating into an adjacent run or block.
    private func segmentAndHighlightRun(
        range runRange: Range<Int>,
        in blockText: String,
        pickLast: Bool
    ) async -> WordHit? {
        let chars = Array(blockText)
        let runText = String(chars[runRange])
        
        let service = SegmentationServiceFactory.active(cwsEnabled: cwsEnabled)
        let segments = await service.segment(run: runText.toSimplified, sentence: blockText.toSimplified)
        guard !segments.isEmpty else { return nil }
        
        // Map back to original characters
        let originalSegments = mapSegmentsToOriginal(segments, originalRun: runText)
        
        currentBlockText = blockText
        currentRunRange = runRange
        currentSegments = originalSegments
        
        let wordIdx = pickLast ? originalSegments.count - 1 : 0
        return await highlightWordAtIndex(wordIdx, segments: originalSegments, runRange: runRange, blockText: blockText)
    }
    
    /// Navigate to an adjacent block element in the DOM and highlight its first/last word.
    /// Returns nil if at the document boundary (chapter edge).
    private func navigateToAdjacentBlock(direction: String) async -> WordHit? {
        guard let root = navigatorVC?.view else { return nil }
        let webViews = ViewHierarchyHelper.findWebViews(in: root)
        
        guard let result = await callJS("getAdjacentBlock", args: "\"\(direction)\"", in: webViews),
              result["atBoundary"] == nil,
              let blockText = result["blockText"] as? String else {
            return nil // At boundary or error
        }
        
        let pickLast = (direction == "prev")
        let chars = Array(blockText)
        let runRange = pickLast
            ? findPreviousRun(before: chars.count, in: blockText)
            : findNextRun(after: 0, in: blockText)
        
        guard let range = runRange else { return nil }
        return await segmentAndHighlightRun(range: range, in: blockText, pickLast: pickLast)
    }
    
    // MARK: - Helpers
    
    /// Map simplified segments back to original (possibly traditional) characters.
    /// Since Trad↔Simp is a 1:1 character mapping, we just slice the original
    /// run using segment lengths from the simplified segmentation.
    private func mapSegmentsToOriginal(_ simplifiedSegments: [String], originalRun: String) -> [String] {
        let chars = Array(originalRun)
        var result: [String] = []
        var offset = 0
        for seg in simplifiedSegments {
            let len = seg.count
            let end = min(offset + len, chars.count)
            result.append(String(chars[offset..<end]))
            offset = end
        }
        return result
    }
    
    /// Given a character offset within a run, find which segment (word) it belongs to.
    private func wordIndexForChar(at charOffsetInRun: Int, in segments: [String]) -> Int {
        var offset = 0
        for (i, seg) in segments.enumerated() {
            if charOffsetInRun >= offset && charOffsetInRun < offset + seg.count {
                return i
            }
            offset += seg.count
        }
        return segments.count - 1
    }
    
    /// Extract the sentence containing `charIndex` by walking to sentence boundary punctuation.
    /// Returns (sentence, wordOffsetInSentence) tuple.
    private func extractSentenceWithOffset(around charIndex: Int, wordLength: Int, in blockText: String) -> (String, Int) {
        let chars = Array(blockText)
        let boundaries: Set<Character> = ["。", "！", "？", "!", "?", "\n"]
        
        var start = charIndex
        while start > 0 && !boundaries.contains(chars[start - 1]) { start -= 1 }
        
        var end = charIndex
        while end < chars.count && !boundaries.contains(chars[end]) { end += 1 }
        if end < chars.count && boundaries.contains(chars[end]) { end += 1 }
        
        let sentence = String(chars[start..<end])
        let wordOffsetInSentence = charIndex - start
        return (sentence, wordOffsetInSentence)
    }
    
    /// Extract the sentence containing `charIndex` by walking to sentence boundary punctuation.
    private func extractSentence(around charIndex: Int, in blockText: String) -> String {
        extractSentenceWithOffset(around: charIndex, wordLength: 0, in: blockText).0
    }
    
    // MARK: - JS Communication (unified)
    
    /// Call a WR function by name with pre-formatted arguments string.
    /// For point-based calls, pass `rootPoint` and it will be converted per web view.
    private func callJS(
        _ functionName: String,
        args rootPoint: CGPoint,
        in webViews: [WKWebView]
    ) async -> [String: Any]? {
        for webView in webViews {
            let local = convertToWebViewCoordinates(rootPoint, webView: webView)
            let argsStr = "\(local.x), \(local.y)"
            if let result = await evaluateWR(functionName, args: argsStr, in: webView) {
                return result
            }
        }
        return nil
    }
    
    /// Call a WR function with a pre-formatted args string (non-point-based).
    private func callJS(
        _ functionName: String,
        args: String,
        in webViews: [WKWebView]
    ) async -> [String: Any]? {
        for webView in webViews {
            if let result = await evaluateWR(functionName, args: args, in: webView) {
                return result
            }
        }
        return nil
    }
    
    /// Call a WR function that returns `{ rects: [...] }` and parse into [CGRect].
    private func callJSForRects(
        _ functionName: String,
        args: String,
        in webViews: [WKWebView]
    ) async -> [CGRect] {
        guard let result = await callJS(functionName, args: args, in: webViews),
              let rectArray = result["rects"] as? [[String: Any]] else { return [] }
        
        return rectArray.compactMap { rd in
            guard let x = (rd["x"] as? NSNumber)?.doubleValue,
                  let y = (rd["y"] as? NSNumber)?.doubleValue,
                  let w = (rd["width"] as? NSNumber)?.doubleValue,
                  let h = (rd["height"] as? NSNumber)?.doubleValue
            else { return nil }
            return CGRect(x: x, y: y, width: w, height: h)
        }
    }
    
    /// Core JS evaluation helper — wraps the call in try/catch and null checks.
    private func evaluateWR(
        _ functionName: String,
        args: String,
        in webView: WKWebView
    ) async -> [String: Any]? {
        let js = "(function(){try{if(window.WR&&window.WR.\(functionName))return window.WR.\(functionName)(\(args));} catch(e){}return null;})();"
        return try? await webView.evaluateJavaScript(js) as? [String: Any]
    }
    
    private func convertToWebViewCoordinates(_ point: CGPoint, webView: WKWebView) -> CGPoint {
        guard let root = navigatorVC?.view else { return point }
        return root.convert(point, to: webView)
    }
}

// MARK: - Character.isCJK

extension Character {
    /// Whether this character is a CJK ideograph (Han script).
    var isCJK: Bool {
        // Uses Unicode script property via regex — delegates to ICU,
        // so it stays correct as new Unicode versions add blocks.
        return String(self).range(of: "\\p{Script=Han}", options: .regularExpression) != nil
    }
}

// MARK: - String Trad→Simp conversion

extension String {
    /// Convert Traditional Chinese to Simplified using ICU transforms.
    /// Returns the original string if conversion fails or produces no change.
    var toSimplified: String {
        self.applyingTransform(StringTransform("Hant-Hans"), reverse: false) ?? self
    }
}
