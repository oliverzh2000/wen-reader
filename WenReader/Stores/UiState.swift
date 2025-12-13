// Copyright 2025 Oliver Zhang
// Licensed under the MIT License

import Foundation
import Combine

/// Global UI state for app-wide UI coordination
@MainActor
final class UiState: ObservableObject {
    @Published var hideStatusBar = false
}
