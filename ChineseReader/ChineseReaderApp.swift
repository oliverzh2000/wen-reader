//
//  SkeletonApp.swift
//  ChineseReader
//
//  Created by Oliver Zhang on 2025-11-08.
//

import SwiftUI

@main
struct ChineseReaderApp: App {
    @StateObject private var catalog = CatalogStore()
    @StateObject private var globalUiState = UiState()
    @StateObject private var settingsStore = SettingsStore()

    var body: some Scene {
        WindowGroup {
            NavigationStack { LibraryView() }
                .environmentObject(catalog)
                .environmentObject(globalUiState)
                .environmentObject(settingsStore)
                .statusBarHidden(globalUiState.hideStatusBar)
        }
    }
}
