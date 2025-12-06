//
//  ErrorAlertModifier.swift
//  WenReader
//
//  Created by architectural refactoring
//

import SwiftUI

/// Reusable view modifier for displaying AppError alerts
struct ErrorAlertModifier: ViewModifier {
    @Binding var error: AppError?
    let title: String
    
    func body(content: Content) -> some View {
        content.alert(
            title,
            isPresented: Binding(
                get: { error != nil },
                set: { if !$0 { error = nil } }
            ),
            presenting: error
        ) { _ in
            Button("OK", role: .cancel) {
                error = nil
            }
        } message: { error in
            VStack(alignment: .leading, spacing: 8) {
                Text(error.localizedDescription)
                if let suggestion = error.recoverySuggestion {
                    Text(suggestion)
                        .font(.caption)
                }
            }
        }
    }
}

extension View {
    /// Displays an error alert with consistent styling
    /// - Parameters:
    ///   - title: The title of the alert
    ///   - error: Binding to an optional AppError
    func errorAlert(title: String, error: Binding<AppError?>) -> some View {
        modifier(ErrorAlertModifier(error: error, title: title))
    }
}
