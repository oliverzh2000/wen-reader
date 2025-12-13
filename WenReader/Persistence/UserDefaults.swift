// Copyright 2025 Oliver Zhang
// Licensed under the MIT License

import Foundation

/// Extension to UserDefaults for convenient Codable persistence
extension UserDefaults {
    /// Save a Codable value to UserDefaults
    /// - Parameters:
    ///   - value: The value to save
    ///   - key: The key to store it under
    func setCodable<T: Codable>(_ value: T, forKey key: String) {
        do {
            let data = try JSONEncoder().encode(value)
            set(data, forKey: key)
        } catch {
            Log.error("Failed to encode data for key '\(key)': \(error)")
        }
    }
    
    /// Load a Codable value from UserDefaults
    /// - Parameters:
    ///   - type: The type to decode
    ///   - key: The key to load from
    ///   - defaultValue: The default value if loading fails
    /// - Returns: The loaded value or the default
    func codable<T: Codable>(
        _ type: T.Type,
        forKey key: String,
        default defaultValue: T
    ) -> T {
        guard let data = data(forKey: key) else {
            return defaultValue
        }
        
        do {
            return try JSONDecoder().decode(type, from: data)
        } catch {
            Log.error("Failed to decode \(type) for key '\(key)': \(error)")
            return defaultValue
        }
    }
    
    /// Load an optional Codable value from UserDefaults
    /// - Parameters:
    ///   - type: The type to decode
    ///   - key: The key to load from
    /// - Returns: The loaded value or nil
    func codable<T: Codable>(
        _ type: T.Type,
        forKey key: String
    ) -> T? {
        guard let data = data(forKey: key) else {
            return nil
        }
        
        do {
            return try JSONDecoder().decode(type, from: data)
        } catch {
            Log.error("Failed to decode \(type) for key '\(key)': \(error)")
            return nil
        }
    }
}
