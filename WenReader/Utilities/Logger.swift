// Copyright 2025 Oliver Zhang
// Licensed under the MIT License

import Foundation
import OSLog

/// Logging utility using Apple's unified logging system (OSLog)
///
/// Usage:
/// ```swift
/// Log.debug("Detailed debug info")
/// Log.info("General information")
/// Log.error("Error occurred: \(error)")
/// ```
enum Log {
    /// Logger instance for the app
    /// Subsystem should be your bundle identifier
    private static let logger = Logger(
        subsystem: Bundle.main.bundleIdentifier ?? "com.wenreader",
        category: "app"
    )
    
    /// Debug-level logging (verbose, development only)
    /// Automatically stripped from Release builds
    nonisolated static func debug(_ message: String) {
        logger.debug("\(message)")
    }
    
    /// Info-level logging (helpful but not essential)
    nonisolated static func info(_ message: String) {
        logger.info("\(message)")
    }
    
    /// Error-level logging (recoverable errors)
    nonisolated static func error(_ message: String) {
        logger.error("\(message)")
    }
    
    /// Fault-level logging (serious issues)
    nonisolated static func fault(_ message: String) {
        logger.fault("\(message)")
    }
}
