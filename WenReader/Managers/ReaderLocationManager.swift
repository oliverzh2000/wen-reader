// Copyright 2025 Oliver Zhang
// Licensed under the MIT License

import Foundation
import ReadiumShared

/// Manages reading location persistence and navigation
@MainActor
final class ReaderLocationManager {
    private var bookId: UUID?
    private var lastSavedAt = Date.distantPast
    
    // MARK: - Public API
    
    func setBookId(_ id: UUID) {
        self.bookId = id
    }
    
    func saveLocation(_ locator: Locator) {
        guard let id = bookId else { return }
        
        // Throttle writes to avoid excessive I/O
        let now = Date()
        guard now.timeIntervalSince(lastSavedAt) > ReaderConstants.Persistence.saveThrottleInterval else {
            return
        }
        lastSavedAt = now
        
        let key = "lastLocation.\(id.uuidString)"
        UserDefaults.standard.set(locator.jsonString, forKey: key)
    }
    
    func loadLocation() -> Locator? {
        guard let id = bookId else { return nil }
        
        let key = "lastLocation.\(id.uuidString)"
        guard let jsonString = UserDefaults.standard.string(forKey: key) else {
            return nil
        }
        
        return try? Locator(jsonString: jsonString)
    }
}
