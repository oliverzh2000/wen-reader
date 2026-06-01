// Copyright 2025 Oliver Zhang
// Licensed under the MIT License

import Foundation

/// Provides a shared `WSDService` instance for the app and wires it into `DictionaryService`.
///
/// Attempts to create a `WSDService` backed by the on-device WSD CoreML model.
/// If the model fails to load (missing from bundle, corrupt, etc.), WSD is disabled
/// and dictionary results are returned in default insertion order (graceful degradation).
///
/// On successful initialization, the service is automatically injected into
/// `CedictSqlService.shared.wsdService` so that callers of `DictionaryService.lookup`
/// receive WSD-sorted results without any direct WSD interaction.
enum WSDServiceFactory {

    /// The shared WSD service, or `nil` if the model failed to load.
    /// Lazily initialized on first access; also sets `CedictSqlService.shared.wsdService`.
    static let shared: WSDService? = {
        do {
            let service = try WSDService()
            CedictSqlService.shared.wsdService = service
            Log.info("WSDServiceFactory: WSD service loaded and injected into DictionaryService")
            return service
        } catch {
            Log.error("WSDServiceFactory: WSD model failed to load (\(error.localizedDescription)), sense ranking disabled")
            return nil
        }
    }()

    /// Ensure the WSD service is initialized and wired into the dictionary service.
    /// Call this at app startup or lazily before the first dictionary lookup that needs WSD.
    /// Safe to call multiple times — the underlying `shared` property is initialized only once.
    static func initialize() {
        _ = shared
    }
}
