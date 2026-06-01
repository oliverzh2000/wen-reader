// Copyright 2025 Oliver Zhang
// Licensed under the MIT License

import Foundation

/// Provides a shared `SegmentationService` instance for the app.
///
/// Tries span scorer first, falls back to dictionary-based segmentation.
enum SegmentationServiceFactory {

    static let sharedTrie: CedictTrie? = {
        let start = CFAbsoluteTimeGetCurrent()
        let trie = CedictTrie.fromDatabase()
        if let trie {
            let ms = (CFAbsoluteTimeGetCurrent() - start) * 1000
            Log.info("SegmentationServiceFactory: Built CEDICT trie (\(trie.wordCount) words) in \(String(format: "%.0f", ms))ms")
        }
        return trie
    }()

    static let shared: SegmentationService = {
        WSDServiceFactory.initialize()

        if let trie = sharedTrie {
            do {
                let service = try SpanScorerSegmentationService(trie: trie)
                Log.info("SegmentationServiceFactory: Using span scorer segmentation")
                return service
            } catch {
                Log.error("SegmentationServiceFactory: Span scorer failed (\(error.localizedDescription)), falling back to CEDICT")
            }
        }

        return CedictSegmentationService(
            dict: CedictSqlService.shared,
            maxWordLength: ReaderConstants.Segmentation.maxWordLength
        )
    }()

    static var spanScorer: SpanScorerSegmentationService? {
        shared as? SpanScorerSegmentationService
    }
}
