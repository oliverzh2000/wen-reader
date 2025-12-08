//
//  SettingsStore.swift
//  ChineseReader
//
//  Created by Oliver Zhang on 2025-11-09.
//

import Combine
import Foundation

// MARK: - SettingsStore
@MainActor
final class SettingsStore: ObservableObject {
    @Published var settings: ReaderSettings {
        didSet { save() }
    }

    private let key = "reader.settings"

    init() {
        self.settings = UserDefaults.standard.codable(
            ReaderSettings.self,
            forKey: key,
            default: ReaderSettings()
        )
    }

    private func save() {
        UserDefaults.standard.setCodable(settings, forKey: key)
    }
}
