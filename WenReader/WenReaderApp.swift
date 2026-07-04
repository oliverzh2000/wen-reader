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

    @State private var showAbout = false

    var body: some Scene {
        WindowGroup {
            NavigationStack {
                LibraryView()
            }
            .environmentObject(catalog)
            .environmentObject(globalUiState)
            .environmentObject(settingsStore)
            .statusBarHidden(globalUiState.hideStatusBar)
            .preferredColorScheme(preferredColorScheme)
            .sheet(isPresented: $showAbout) {
                AboutView()
                    .presentationDetents([.medium, .large])
            }
        }
        .commands {
            CommandGroup(replacing: .appInfo) {
                Button("About Wen Reader") {
                    showAbout = true
                }
            }
            CommandMenu("Word") {
                Text("Previous Word\t\t←")
                Text("Next Word\t\t→")
                Divider()
                Text("Shrink Selection\t\t↑")
                Text("Expand Selection\t\t↓")
                Divider()
                Text("Toggle Auto-Advance\t\tSpace")
            }
        }
    }
}

extension WenReaderApp {
    // Compute color scheme from settings
    private var preferredColorScheme: ColorScheme? {
        switch settingsStore.settings.theme {
        case .light:
            return .light
        case .dark:
            return .dark
        case .system:
            return nil
        }
    }
}
