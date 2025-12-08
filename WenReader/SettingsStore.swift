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
        if let data = UserDefaults.standard.data(forKey: key),
            let s = try? JSONDecoder().decode(ReaderSettings.self, from: data)
        {
            self.settings = s
        } else {
            self.settings = ReaderSettings()
        }
    }

    private func save() {
        if let data = try? JSONEncoder().encode(settings) {
            UserDefaults.standard.set(data, forKey: key)
        }
    }
}
