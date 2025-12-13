// Copyright 2025 Oliver Zhang
// Licensed under the MIT License

import Foundation
import Combine

/// Manages dictionary lookups and navigation stack
/// This manager owns dictionary lookup state (results, navigation stack)
/// WordHit state is owned by ReadiumEngine (from user interaction)
@MainActor
final class ReaderDictionaryManager: ObservableObject {
    @Published private(set) var currentResult: DictionaryResult?
    
    private var stack: [DictionaryResult] = []
    private let service: DictionaryService
    
    init(service: DictionaryService = CedictSqlService.shared) {
        self.service = service
    }
    
    // MARK: - Public API
    
    /// Look up a word and reset the stack
    func lookup(_ word: String?) async {
        guard let word = word else {
            clear()
            return
        }
        
        guard let result = await service.lookup(word) else {
            return
        }
        
        stack = [result]
        currentResult = result
    }
    
    /// Push a new word onto the dictionary stack (for cross-references)
    func push(_ word: String) async {
        guard let result = await service.lookup(word) else {
            return
        }
        
        stack.append(result)
        currentResult = result
    }
    
    /// Go back to previous dictionary entry
    func pop() {
        guard stack.count > 1 else { return }
        stack.removeLast()
        currentResult = stack.last
    }
    
    /// Clear dictionary and word hit
    func clear() {
        stack.removeAll()
        currentResult = nil
    }
    
    /// Whether we can go back in the dictionary stack
    var canGoBack: Bool {
        stack.count > 1
    }
}
