// Copyright 2025 Oliver Zhang
// Licensed under the MIT License

import Foundation

/// Application-wide error types with user-friendly messages
enum AppError: LocalizedError {
    case bookImportFailed(reason: String)
    case bookOpenFailed(reason: String)
    case databaseError(message: String)
    case fileOperationFailed(operation: String, underlying: Error)
    case invalidEPUB(reason: String)
    case metadataLoadFailed(underlying: Error)
    
    var errorDescription: String? {
        switch self {
        case .bookImportFailed(let reason):
            return "Failed to import book: \(reason)"
        case .bookOpenFailed(let reason):
            return "Failed to open book: \(reason)"
        case .databaseError(let message):
            return "Dictionary error: \(message)"
        case .fileOperationFailed(let operation, let underlying):
            return "Failed to \(operation): \(underlying.localizedDescription)"
        case .invalidEPUB(let reason):
            return "Invalid EPUB file: \(reason)"
        case .metadataLoadFailed(let underlying):
            return "Failed to read book information: \(underlying.localizedDescription)"
        }
    }
    
    var recoverySuggestion: String? {
        switch self {
        case .bookImportFailed:
            return "Please make sure the file is a valid EPUB format and try again."
        case .bookOpenFailed:
            return "The book may be corrupted. Try re-importing it."
        case .databaseError:
            return "Please restart the app. If the problem persists, reinstall the app."
        case .fileOperationFailed:
            return "Please check if you have enough storage space and try again."
        case .invalidEPUB:
            return "This file does not appear to be a valid EPUB. Please select a different file."
        case .metadataLoadFailed:
            return "The book file may be incomplete or corrupted."
        }
    }
}
