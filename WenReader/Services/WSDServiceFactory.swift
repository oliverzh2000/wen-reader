// Copyright 2025 Oliver Zhang
// Licensed under the MIT License

import Foundation

/// Provides a shared `WSDService` instance for the app.
///
/// Initialized eagerly at app startup. When WSD is disabled via settings,
/// callers should skip calling into the service — dictionary results will
/// be returned in database insertion order.
enum WSDServiceFactory {

    /// The shared WSD service, or `nil` if the model failed to load.
    static let shared: WSDService? = {
        do {
            let service = try WSDService()
            Log.info("WSDServiceFactory: WSD service loaded")
            return service
        } catch {
            Log.error("WSDServiceFactory: WSD model failed to load (\(error.localizedDescription)), sense ranking disabled")
            return nil
        }
    }()

    /// Eagerly initialize the WSD service and wire it into the dictionary service.
    /// Call at app startup. Safe to call multiple times.
    static func initialize(wsdEnabled: Bool) {
        if wsdEnabled, let service = shared {
            CedictSqlService.shared.wsdService = service
        } else {
            CedictSqlService.shared.wsdService = nil
        }
    }
}
