// Copyright 2025 Oliver Zhang
// Licensed under the MIT License

import Foundation
import SwiftUI

struct SettingsSheet: View {
    @EnvironmentObject var settingsStore: SettingsStore
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            Form {
                Section("Typography") {
                    Picker("Font", selection: $settingsStore.settings.font) {
                        ForEach(ReaderFont.allCases) { font in
                            Text(font.displayName).tag(font)
                        }
                    }
                    .pickerStyle(.automatic)

                    VStack(alignment: .leading) {
                        Text("Font Size")
                        Slider(
                            value: $settingsStore.settings.fontSize,
                            in: 1.0...2.0,
                            step: 0.1
                        ) {
                        } minimumValueLabel: {
                            Text("")
                        } maximumValueLabel: {
                            Text(
                                "\(String(format: "%.1f", settingsStore.settings.fontSize))"
                            )
                            .foregroundStyle(.secondary)
                        }
                    }

                    VStack(alignment: .leading) {
                        Text("Line Height")
                        Slider(
                            value: $settingsStore.settings.lineHeight,
                            in: 1.0...2.0,
                            step: 0.1
                        ) {
                        } minimumValueLabel: {
                            Text("")
                        } maximumValueLabel: {
                            Text(
                                "\(String(format: "%.1f", settingsStore.settings.lineHeight))"
                            )
                            .foregroundStyle(.secondary)
                        }
                    }

                    VStack(alignment: .leading) {
                        Text("Margins")
                        Slider(
                            value: $settingsStore.settings.margins,
                            in: 0.0...2.0,
                            step: 0.1
                        ) {
                        } minimumValueLabel: {
                            Text("")
                        } maximumValueLabel: {
                            Text(
                                "\(String(format: "%.1f", settingsStore.settings.margins))"
                            )
                            .foregroundStyle(.secondary)
                        }
                    }

                    Toggle(
                        "Justify Text",
                        isOn: $settingsStore.settings.justify
                    )
                }

                Section("Appearance") {
                    Picker("Theme", selection: $settingsStore.settings.theme) {
                        ForEach(ReaderTheme.allCases) { theme in
                            Text(theme.displayName).tag(theme)
                        }
                    }
                    .pickerStyle(.automatic)
                }

                Section("Interaction") {
                    Picker("Long Press", selection: $settingsStore.settings.interactionMode) {
                        Text("Word Lookup").tag(ReaderInteractionMode.custom)
                        Text("Text Selection").tag(ReaderInteractionMode.system)
                    }
                    .pickerStyle(.automatic)
                }

                Section {
                    Toggle(
                        "Word Segmentation",
                        isOn: $settingsStore.settings.cwsEnabled
                    )
                    Toggle(
                        "Definition Ranking",
                        isOn: $settingsStore.settings.wsdEnabled
                    )
                } header: {
                    Text("On-Device ML")
                } footer: {
                    Text("Enable for context-aware word segmentation and definition ranking. Disable to save battery.")
                }

                Section {
                    Button("Reset to Defaults", role: .destructive) {
                        settingsStore.settings = ReaderSettings()
                    }
                }
            }
            .navigationTitle("Settings")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button { dismiss() } label: {
                        Image(systemName: "xmark")
                            .foregroundStyle(.secondary)
                    }
                }
            }
            .preferredColorScheme(preferredColorScheme)
        }
    }
    
    // Compute color scheme from settings to update sheet immediately
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
