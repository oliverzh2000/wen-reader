//
//  TextExport.swift
//  WenReader
//
//  Created by Oliver Zhang on 2025-12-07.
//

import Foundation
import UIKit

/// Scope for share/send operations
enum ShareScope {
    case word
    case sentence
    case paragraph
}

/// Helper for sharing/sending text to external apps
struct TextExport {
    
    // MARK: - Pleco Integration
    
    /// Open text in Pleco dictionary app
    static func openInPleco(scope: ShareScope, text: String) {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }

        switch scope {
        case .word:
            // Direct word lookup in dictionary
            guard let encoded = trimmed.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed) else {
                return
            }

            let urlString = "plecoapi://x-callback-url/s?q=\(encoded)"
            guard let url = URL(string: urlString) else { return }

            if UIApplication.shared.canOpenURL(url) {
                UIApplication.shared.open(url, options: [:])
            } else {
                // Fallback: copy word to clipboard
                UIPasteboard.general.string = trimmed
            }

        case .sentence, .paragraph:
            // For longer text, send to Clipboard Reader
            UIPasteboard.general.string = trimmed

            // URL to open Clipboard Reader
            let urlString = "plecoapi://x-callback-url/clipboard"
            guard let url = URL(string: urlString) else { return }

            if UIApplication.shared.canOpenURL(url) {
                UIApplication.shared.open(url, options: [:])
            }
        }
    }
    
    // MARK: - ChatGPT Integration
    
    /// Open text in ChatGPT with contextual prompt
    static func openInChatGPT(
        scope: ShareScope,
        promptStyle: PromptStyle,
        text: String,
        context: String? = nil
    ) {
        let trimmedText = text.trimmingCharacters(in: .whitespacesAndNewlines)
        let trimmedContext = context?.trimmingCharacters(in: .whitespacesAndNewlines)

        let prompt = makeChatGPTPrompt(
            scope: scope,
            style: promptStyle,
            text: trimmedText,
            context: trimmedContext
        )

        guard let encodedPrompt = prompt.addingPercentEncoding(
            withAllowedCharacters: CharacterSet.urlQueryAllowed
        ) else {
            return
        }

        let urlString = "https://chat.openai.com/?q=\(encodedPrompt)"
        guard let url = URL(string: urlString) else { return }

        UIApplication.shared.open(url, options: [:])
    }
    
    // MARK: - Clipboard
    
    /// Copy text to clipboard
    static func copyToClipboard(_ text: String) {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        UIPasteboard.general.string = trimmed
    }
    
    // MARK: - Private Helpers
    
    private static func makeChatGPTPrompt(
        scope: ShareScope,
        style: PromptStyle,
        text: String,
        context: String?
    ) -> String {
        switch (style, scope) {
        case (.quick, .word):
            return """
            Give a concise 2–3 word English gloss for the following Chinese word, suitable for a popup dictionary. Do not explain, just give the gloss.

            Word: \(text)
            Sentence (for context): \(context ?? "")
            """

        case (.quick, .sentence):
            return """
            Translate the following Chinese sentence into natural, concise English. Only output the translation.

            Sentence: \(text)
            """

        case (.quick, .paragraph):
            return """
            Translate the following Chinese paragraph into natural, concise English. Only output the translation.

            Paragraph:
            \(text)
            """

        case (.full, .word):
            return """
            Explain the following Chinese word in English. Include:
            - A natural English translation
            - Brief nuance and register (formal/informal, written/spoken)
            - Any nuance or implied tone that is not obvious from a direct translation

            Word: \(text)
            Sentence (for context): \(context ?? "")
            """

        case (.full, .sentence):
            return """
            Explain the following Chinese sentence in English. Include:
            - A natural English translation
            - A brief breakdown of any key words or grammar points
            - Any nuance or implied tone that is not obvious from a direct translation

            Sentence: \(text)
            """

        case (.full, .paragraph):
            return """
            Explain the following Chinese paragraph in English. Include:
            - A natural English translation
            - A brief summary of the main idea
            - Any important nuances, tone, or context implied by the wording

            Paragraph:
            \(text)
            """
        }
    }
}
