// Copyright 2025 Oliver Zhang
// Licensed under the MIT License

import SwiftUI

/// Menu for exporting text to external apps (Pleco, Clipboard, LLM prompt)
///
/// Scope and prompt style are controlled in Settings.
/// This menu is just action buttons — one tap to do the thing.
struct TextExportMenu: View {
    let wordHit: WordHit
    @EnvironmentObject var settingsStore: SettingsStore
    
    private var scope: ShareScope { settingsStore.settings.shareScope }
    private var promptStyle: PromptStyle { settingsStore.settings.promptStyle }
    
    var body: some View {
        Menu {
            Section("Actions") {
                Button("Open in Pleco") {
                    TextExport.openInPleco(scope: scope, text: textForScope)
                }
                
                Button("Copy Text") {
                    TextExport.copyToClipboard(textForScope)
                }
                
                Button("Copy LLM Prompt") {
                    let prompt = TextExport.buildPrompt(
                        scope: scope,
                        promptStyle: promptStyle,
                        text: textForScope,
                        context: scope == .word ? wordHit.sentence : nil
                    )
                    TextExport.copyToClipboard(prompt)
                }
            }
            Section("Options") {
                Picker("Text Scope", selection: $settingsStore.settings.shareScope) {
                    ForEach(ShareScope.allCases) { scope in
                        Text(scope.displayName).tag(scope)
                    }
                }
                .pickerStyle(.menu)
                Picker("Prompt Style", selection: $settingsStore.settings.promptStyle) {
                    ForEach(PromptStyle.allCases) { style in
                        Text(style.displayName).tag(style)
                    }
                }
                .pickerStyle(.menu)
            }
        } label: {
            Label("Share", systemImage: "square.and.arrow.up")
                .labelStyle(.iconOnly)
                .tint(.secondary)
        }
    }
    
    /// Resolve the text for the current scope from the WordHit.
    private var textForScope: String {
        switch scope {
        case .word: return wordHit.word
        case .sentence: return wordHit.sentence
        case .paragraph: return wordHit.block
        }
    }
}
