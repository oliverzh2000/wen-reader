// Copyright 2025 Oliver Zhang
// Licensed under the MIT License

import CoreGraphics
import Foundation

/// Centralized constants for the reader feature
enum ReaderConstants {
    /// Dictionary popover display settings
    enum Dictionary {
        /// Maximum height for dictionary popover
        static let popoverMaxHeight: CGFloat = 300
        
        /// Animation response time for dictionary transitions
        static let animationResponse: Double = 0.25
        
        /// Animation damping factor for dictionary transitions
        static let animationDamping: Double = 1.0
    }
    
    /// Location persistence settings
    enum Persistence {
        /// Minimum interval between location saves to throttle writes
        static let saveThrottleInterval: TimeInterval = 0.5
    }
    
    /// User interaction timing constants
    enum Interaction {
        /// Duration for long press gesture recognition
        static let longPressDuration: TimeInterval = 0.2
        
        /// Time window to suppress taps after long press ends
        static let tapSuppressionWindow: TimeInterval = 0.1
    }
    
    /// Segmentation settings
    enum Segmentation {
        /// Maximum word length for dictionary-based segmentation
        static let maxWordLength: Int = 6
        
        /// Maximum cache size for word lookup results
        static let maxCacheSize: Int = 1000

        /// Maximum cache size for CoreML sentence segmentation results (FIFO eviction)
        static let maxSentenceCacheSize: Int = 500
    }

    /// Word Sense Disambiguation settings
    enum WSD {
        /// Maximum cache size for WSD rank results (FIFO eviction)
        static let maxCacheSize: Int = 200
    }
    
    /// Image file settings
    enum Image {
        /// JPEG compression quality for book covers.
        static let coverCompressionQuality: CGFloat = 0.1
    }
    
    enum Book {
        static let maxTitleLength = 200
    }
}
