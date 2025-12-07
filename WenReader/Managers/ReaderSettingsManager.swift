//
//  ReaderSettingsManager.swift
//  WenReader
//
//  Created by architectural refactoring
//

import ReadiumNavigator
import ReadiumShared
import SwiftUI

extension FontFamily {
    static let notoSerifSC: FontFamily = "Noto Serif SC"
    static let pingFangSC: FontFamily = "PingFang SC"
}

/// Manages application of reader settings to the navigator
@MainActor
final class ReaderSettingsManager {
    
    // MARK: - Public API
    
    /// Apply reader settings to the navigator
    func apply(
        _ settings: ReaderSettings,
        systemColorScheme: ColorScheme,
        to navigator: EPUBNavigatorViewController,
        interactionManager: ReaderInteractionManager
    ) {
        // Create preferences editor
        let preferences = EPUBPreferences(publisherStyles: false)
        let editor = navigator.editor(of: preferences)
        
        applyFont(settings.font, to: editor)
        applyTheme(settings.theme, systemColorScheme: systemColorScheme, to: editor)
        applyInteractionMode(settings.interactionMode, to: interactionManager)
        applyTypography(settings, to: editor)
        
        navigator.submitPreferences(editor.preferences)
    }
    
    // MARK: - Private Helpers
    
    private func applyFont(_ font: ReaderFont, to editor: EPUBPreferencesEditor) {
        switch font {
        case .notoSerifSC:
            editor.fontFamily.set(.notoSerifSC)
        case .pingFangSC:
            editor.fontFamily.set(.pingFangSC)
        }
    }
    
    private func applyTheme(
        _ theme: ReaderTheme,
        systemColorScheme: ColorScheme,
        to editor: EPUBPreferencesEditor
    ) {
        switch theme {
        case .light:
            editor.theme.set(.light)
        case .dark:
            editor.theme.set(.dark)
        case .system:
            editor.theme.set(systemColorScheme == .light ? .light : .dark)
        }
    }
    
    private func applyInteractionMode(
        _ mode: ReaderInteractionMode,
        to interactionManager: ReaderInteractionManager
    ) {
        switch mode {
        case .system:
            interactionManager.setMode(.systemSelection)
        case .custom:
            interactionManager.setMode(.customMagnifier)
        }
    }
    
    private func applyTypography(_ settings: ReaderSettings, to editor: EPUBPreferencesEditor) {
        editor.fontSize.set(settings.fontSize)
        editor.lineHeight.set(settings.lineHeight)
        editor.pageMargins.set(settings.margins)
        editor.textAlign.set(
            settings.justify ? TextAlignment.justify : TextAlignment.start
        )
    }
}
