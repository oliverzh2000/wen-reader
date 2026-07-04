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
    var onToggleAutoAdvance: () -> Void
    var isAutoAdvancing: Bool

    var body: some View {
        HStack {
            Button(action: onPrev) {
                Image(systemName: "chevron.left")
                    .frame(maxWidth: .infinity)
            }
            Button(action: onShrink) {
                Image(systemName: "chevron.left.2")
                    .frame(maxWidth: .infinity)
            }

            // Center: play/pause toggle for auto-advance
            Button {
                UIImpactFeedbackGenerator(style: .light).impactOccurred()
                onToggleAutoAdvance()
            } label: {
                Image(systemName: isAutoAdvancing ? "pause.fill" : "play.fill")
                    .contentTransition(.symbolEffect(.replace))
                    .foregroundStyle(isAutoAdvancing ? Color.accentColor : .secondary)
                    .frame(maxWidth: .infinity)
            }

            Button(action: onGrow) {
                Image(systemName: "chevron.right.2")
                    .frame(maxWidth: .infinity)
            }
            Button(action: onNext) {
                Image(systemName: "chevron.right")
                    .frame(maxWidth: .infinity)
            }
        }
        .foregroundStyle(.secondary)
        .padding(.horizontal)
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
                onToggleAutoAdvance: { print("toggle") },
                isAutoAdvancing: false
            )
        }
    }
}
