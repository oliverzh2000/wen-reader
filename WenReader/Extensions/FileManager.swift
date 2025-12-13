// Copyright 2025 Oliver Zhang
// Licensed under the MIT License

import Foundation

extension FileManager {
    /// Returns the directory for storing book files in Application Support
    nonisolated static var appSupportBooksDir: URL {
        let base = FileManager.default.urls(
            for: .applicationSupportDirectory,
            in: .userDomainMask
        )[0]
        let dir = base.appendingPathComponent("Books", isDirectory: true)
        
        // Create directory if it doesn't exist
        do {
            try FileManager.default.createDirectory(
                at: dir,
                withIntermediateDirectories: true
            )
        } catch {
            // Log but don't crash - directory may already exist
            Log.error("Failed to create Books directory at \(dir.path): \(error)")
        }
        
        return dir
    }
}
