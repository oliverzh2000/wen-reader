//
//  UiState.swift
//  WenReader
//
//  Created by refactoring on 2025-12-10.
//

import Foundation
import Combine

/// Global UI state for app-wide UI coordination
@MainActor
final class UiState: ObservableObject {
    @Published var hideStatusBar = false
}
