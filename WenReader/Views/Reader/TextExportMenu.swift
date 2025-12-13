// Copyright 2025 Oliver Zhang
// Licensed under the MIT License

import SwiftUI

/// Menu for exporting text to external apps (Pleco, ChatGPT, Clipboard)
struct TextExportMenu: View {
    let wordHit: WordHit
    @Binding var promptStyle: PromptStyle
    
    var body: some View {
        Menu {
            Section("Send/Share") {
                // Pleco submenu
                Menu {
                    Button("Word") {
                        TextExport.openInPleco(scope: .word, text: wordHit.word)
                    }
                    Button("Sentence") {
                        TextExport.openInPleco(scope: .sentence, text: wordHit.sentence)
                    }
                    Button("Paragraph") {
                        TextExport.openInPleco(scope: .paragraph, text: wordHit.block)
                    }
                } label: {
                    Label("Pleco", systemImage: "book.pages")
                }

                // ChatGPT submenu
                Menu {
                    Section("Select Prompt Style") {
                        Picker("Prompt Style", selection: $promptStyle) {
                            ForEach(PromptStyle.allCases) { style in
                                Text(style.displayName).tag(style)
                            }
                        }
                        .pickerStyle(.inline)
                    }
                    Button("Word") {
                        TextExport.openInChatGPT(
                            scope: .word,
                            promptStyle: promptStyle,
                            text: wordHit.word,
                            context: wordHit.sentence
                        )
                    }
                    Button("Sentence") {
                        TextExport.openInChatGPT(
                            scope: .sentence,
                            promptStyle: promptStyle,
                            text: wordHit.sentence
                        )
                    }
                    Button("Paragraph") {
                        TextExport.openInChatGPT(
                            scope: .paragraph,
                            promptStyle: promptStyle,
                            text: wordHit.block
                        )
                    }
                } label: {
                    Label("ChatGPT", systemImage: "sparkles")
                }

                // Copy submenu
                Menu {
                    Button("Word") {
                        TextExport.copyToClipboard(wordHit.word)
                    }
                    Button("Sentence") {
                        TextExport.copyToClipboard(wordHit.sentence)
                    }
                    Button("Paragraph") {
                        TextExport.copyToClipboard(wordHit.block)
                    }
                } label: {
                    Label("Copy", systemImage: "doc.on.doc")
                }
            }
        } label: {
            Label("Share", systemImage: "square.and.arrow.up")
                .labelStyle(.iconOnly)
                .tint(.secondary)
        }
    }
}
