//
//  WenReaderApp.swift
//  ChineseReader
//
//  Created by Oliver Zhang on 2025-11-08.
//

import SwiftUI

@main
struct WenReaderApp: App {
    @StateObject private var catalog = CatalogStore()
    @StateObject private var globalUiState = UiState()
    @StateObject private var settingsStore = SettingsStore()

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
        }
    }
}
