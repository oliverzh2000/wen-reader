// Copyright 2025 Oliver Zhang
// Licensed under the MIT License

import SwiftUI

/// Floating toolbar for adjusting word selection boundaries.
/// Sits just below the safe area in the unsafe region.
struct WordAdjustmentBar: View {
    var onPrev: () -> Void
    var onShrink: () -> Void
    var onGrow: () -> Void
    var onNext: () -> Void

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
            Button(action: onGrow) {
                Image(systemName: "chevron.right.2")
                    .frame(maxWidth: .infinity)
            }
            Button(action: onNext) {
                Image(systemName: "chevron.right")
                    .frame(maxWidth: .infinity)
            }
        }
        .buttonStyle(.plain)
        .foregroundStyle(.secondary)
        .padding(.horizontal)
        .ignoresSafeArea(edges: .bottom)
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
                onNext: { print("next") }
            )
        }
    }
}
