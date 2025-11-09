//
//  SettingsStore.swift
//  ChineseReader
//
//  Created by Oliver Zhang on 2025-11-09.
//

import Combine
import Foundation

// MARK: - Enums
enum ReaderFont: String, CaseIterable, Codable, Identifiable {
    case songti
    case pingfang

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .songti: return "Songti (宋体)"
        case .pingfang: return "PingFang (苹方)"
        }
    }
}

enum ReaderTheme: String, CaseIterable, Codable, Identifiable {
    case light, dark, sepia

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .light: return "Light"
        case .dark: return "Dark"
        case .sepia: return "Sepia"
        }
    }
}

// MARK: - Default settings
struct ReaderSettings: Codable, Equatable {
    var font: ReaderFont = .songti
    var fontSize: Double = 1.0
    var lineHeight: Double = 1.5
    var margins: Double = 1.0
    var justify: Bool = true
    var theme: ReaderTheme = .light
}

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
