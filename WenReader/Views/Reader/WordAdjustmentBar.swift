// Copyright 2025 Oliver Zhang
// Licensed under the MIT License

import SwiftUI

/// Floating toolbar for adjusting word selection boundaries.
/// Sits in the bottom safe area inset region.
///
/// Layout: [Prev] [Shrink] [Play/Pause] [Grow] [Next]
///
/// - Prev/Next: navigate between segmented words
/// - Shrink/Grow: adjust selection boundary by one character
/// - Play/Pause (center): toggle auto-advance mode with animated symbol transition
struct WordAdjustmentBar: View {
    var onPrev: () -> Void
    var onShrink: () -> Void
    var onGrow: () -> Void
    var onNext: () -> Void
    
    /// Auto-advance interval in seconds. Passed from settings.
    var autoAdvanceInterval: Double = 1.0
    
    @State private var isAutoAdvancing = false
    @State private var autoAdvanceTask: Task<Void, Never>?

    var body: some View {
        HStack {
            Button(action: { stopAutoAdvance(); onPrev() }) {
                Image(systemName: "chevron.left")
                    .frame(maxWidth: .infinity)
            }
            Button(action: { stopAutoAdvance(); onShrink() }) {
                Image(systemName: "chevron.left.2")
                    .frame(maxWidth: .infinity)
            }
            
            // Center: play/pause toggle for auto-advance
            autoAdvanceToggle
            
            Button(action: { stopAutoAdvance(); onGrow() }) {
                Image(systemName: "chevron.right.2")
                    .frame(maxWidth: .infinity)
            }
            Button(action: { stopAutoAdvance(); onNext() }) {
                Image(systemName: "chevron.right")
                    .frame(maxWidth: .infinity)
            }
        }
        .foregroundStyle(.secondary)
        .padding(.horizontal)
        .onDisappear {
            // Ensure timer is killed when bar is removed from view hierarchy
            stopAutoAdvance()
        }
    }
    
    // MARK: - Auto-Advance Toggle (center button)
    
    /// Animated play/pause button that toggles auto-advance mode.
    /// Uses `.contentTransition(.symbolEffect(.replace))` for a smooth morph between icons.
    private var autoAdvanceToggle: some View {
        Button {
            UIImpactFeedbackGenerator(style: .light).impactOccurred()
            if isAutoAdvancing {
                stopAutoAdvance()
            } else {
                startAutoAdvance()
            }
        } label: {
            Image(systemName: isAutoAdvancing ? "pause.fill" : "play.fill")
                .contentTransition(.symbolEffect(.replace))
                .foregroundStyle(isAutoAdvancing ? Color.accentColor : .secondary)
                .frame(maxWidth: .infinity)
        }
    }
    
    // MARK: - Auto-Advance
    
    private func startAutoAdvance() {
        guard !isAutoAdvancing else { return }
        isAutoAdvancing = true
        
        autoAdvanceTask = Task {
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: UInt64(autoAdvanceInterval * 1_000_000_000))
                guard !Task.isCancelled else { break }
                await MainActor.run { onNext() }
            }
            await MainActor.run { isAutoAdvancing = false }
        }
    }
    
    private func stopAutoAdvance() {
        autoAdvanceTask?.cancel()
        autoAdvanceTask = nil
        isAutoAdvancing = false
    }
}

#Preview {
    ZStack {
        Color(.systemBackground)

        VStack {
            Spacer()
            WordAdjustmentBar(
                onPrev: { print("prev") },
                onShrink: { print("shrink") },
                onGrow: { print("grow") },
                onNext: { print("next") },
                autoAdvanceInterval: 0.5
            )
        }
    }
}
