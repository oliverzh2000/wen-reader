// Copyright 2025 Oliver Zhang
// Licensed under the MIT License

import SwiftUI

@main
struct WenReaderApp: App {
    @StateObject private var catalog = CatalogStore()
    @StateObject private var globalUiState = UiState()
    @StateObject private var settingsStore = SettingsStore()

    init() {
        // Eagerly initialize ML services at app startup (off main thread work
        // is deferred internally, but this triggers model loading early).
        let settings = UserDefaults.standard.codable(
            ReaderSettings.self,
            forKey: "reader.settings",
            default: ReaderSettings()
        )
        SegmentationServiceFactory.initialize()
        WSDServiceFactory.initialize(wsdEnabled: settings.wsdEnabled)
    }

    var body: some Scene {
        WindowGroup {
            TabView {
                // Library tab
                NavigationStack {
                    LibraryView()
                }
                .tabItem {
                    Label("Library", systemImage: "books.vertical")
                }

                // Settings tab
                NavigationStack {
                    AboutView()
                }
                .tabItem {
                    Label("About", systemImage: "info.circle")
                }
            }
            // Environment objects apply to both tabs
            .environmentObject(catalog)
            .environmentObject(globalUiState)
            .environmentObject(settingsStore)
            .statusBarHidden(globalUiState.hideStatusBar)
            .preferredColorScheme(preferredColorScheme)
        }
    }
    
    // Compute color scheme from settings
    private var preferredColorScheme: ColorScheme? {
        switch settingsStore.settings.theme {
        case .light:
            return .light
        case .dark:
            return .dark
        case .system:
            return nil  // nil = follow system appearance
        }
    }
}
