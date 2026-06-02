// Copyright 2025 Oliver Zhang
// Licensed under the MIT License

import Foundation

/// Provides segmentation services for the app.
///
/// Initializes eagerly at app startup. The active service depends on the
/// `cwsEnabled` setting — when disabled, falls back to dictionary-based segmentation.
enum SegmentationServiceFactory {

    /// Shared CEDICT trie, built once at startup.
    static let sharedTrie: CedictTrie? = {
        let start = CFAbsoluteTimeGetCurrent()
        let trie = CedictTrie.fromDatabase()
        if let trie {
            let ms = (CFAbsoluteTimeGetCurrent() - start) * 1000
            Log.info("SegmentationServiceFactory: Built CEDICT trie (\(trie.wordCount) words) in \(String(format: "%.0f", ms))ms")
        }
        return trie
    }()

    /// The span scorer service, if model loaded successfully.
    static let spanScorer: SpanScorerSegmentationService? = {
        guard let trie = sharedTrie else { return nil }
        do {
            let service = try SpanScorerSegmentationService(trie: trie)
            Log.info("SegmentationServiceFactory: Span scorer loaded")
            return service
        } catch {
            Log.error("SegmentationServiceFactory: Span scorer failed (\(error.localizedDescription))")
            return nil
        }
    }()

    /// The dictionary-based fallback segmentation service.
    static let cedictService: SegmentationService = CedictSegmentationService(
        dict: CedictSqlService.shared,
        maxWordLength: ReaderConstants.Segmentation.maxWordLength
    )

    /// Returns the active segmentation service based on settings.
    /// When CWS is enabled and span scorer is available, uses ML segmentation.
    /// Otherwise falls back to dictionary-based.
    static func active(cwsEnabled: Bool) -> SegmentationService {
        if cwsEnabled, let scorer = spanScorer {
            return scorer
        }
        return cedictService
    }

    /// Eagerly initialize all services. Call at app startup.
    static func initialize() {
        _ = sharedTrie
        _ = spanScorer
        _ = cedictService
    }
}
