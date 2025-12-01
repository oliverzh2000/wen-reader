//
//  SettingsStore.swift
//  ChineseReader
//
//  Created by Oliver Zhang on 2025-11-09.
//

import Combine
import Foundation

// MARK: - Enums
enum ReaderInteractionMode: String, CaseIterable, Codable, Identifiable {
    case customMagnifier  // with inline pinyin/definition under magnifier
    case systemSelection  // default iOS selection

    var id: String { rawValue }

    mutating func toggle() {
        if self == .customMagnifier {
            self = .systemSelection
        } else {
            self = .customMagnifier
        }
    }
}

enum ReaderFont: String, CaseIterable, Codable, Identifiable {
    case notoSerifSC
    case pingFangSC

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .notoSerifSC: return "Noto Serif SC"
        case .pingFangSC: return "PingFang SC"
        }
    }
}

enum ReaderTheme: String, CaseIterable, Codable, Identifiable {
    case light, dark, sepia, system

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .light: return "Light"
        case .dark: return "Dark"
        case .sepia: return "Sepia"
        case .system: return "System"
        }
    }
}

enum PromptStyle: String, CaseIterable, Codable, Identifiable {
    case quick, full
    
    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .quick: return "Concise Translation"
        case .full: return "Detailed Explanation"
        }
    }
}

// MARK: - Default settings
struct ReaderSettings: Codable, Equatable {
    var interactionMode: ReaderInteractionMode = .systemSelection
    var font: ReaderFont = .notoSerifSC
    var fontSize: Double = 1.5
    var lineHeight: Double = 1.5
    var margins: Double = 1.0
    var justify: Bool = true
    var theme: ReaderTheme = .system
    var promptStyle: PromptStyle = .quick
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
