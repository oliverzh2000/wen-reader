//
//  AboutView.swift
//  WenReader
//
//  Created by Oliver Zhang on 2025-12-01.
//

import Foundation
import SwiftUI

/// One third-party library / asset that has a license file in the bundle.
struct LicenseItem: Identifiable, Hashable {
    let id = UUID()
    let name: String              // e.g. "CC-CEDICT"
    let subtitle: String          // e.g. "CC BY-SA 3.0"
    let resourceName: String      // e.g. "LICENSE"
    let resourceExtension: String // e.g. "txt"
}

/// Configure your libraries here.
private let thirdPartyLicenses: [LicenseItem] = [
    LicenseItem(
        name: "CC-CEDICT",
        subtitle: "CC BY-SA 3.0",
        resourceName: "cedict-license",
        resourceExtension: "txt"
    ),
    LicenseItem(
        name: "Readium",
        subtitle: "BSD-3-Clause",
        resourceName: "readium-license",
        resourceExtension: "txt"
    ),
    LicenseItem(
        name: "Noto Serif SC",
        subtitle: "SIL Open Font License 1.1",
        resourceName: "noto-serif-sc-license",
        resourceExtension: "txt"
    )
]

struct LicensesView: View {
    @State private var selectedLicense: LicenseItem?

    var body: some View {
        List(thirdPartyLicenses) { item in
            Button {
                selectedLicense = item
            } label: {
                VStack(alignment: .leading, spacing: 4) {
                    Text(item.name)
                        .font(.headline)
                    Text(item.subtitle)
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                }
                .padding(.vertical, 4)
            }
        }
        .navigationTitle("Licenses")
        .sheet(item: $selectedLicense) { item in
            LicenseDetailView(item: item)
        }
    }
}

struct LicenseDetailView: View {
    let item: LicenseItem

    @Environment(\.dismiss) private var dismiss
    @State private var licenseText: String = "Loading…"

    var body: some View {
        NavigationStack {
            ScrollView {
                Text(licenseText)
                    .font(.caption)
                    .monospaced()
                    .padding()
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .navigationTitle(item.name)
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Done") {
                        dismiss()
                    }
                }
            }
            .onAppear {
                loadLicenseText()
            }
        }
    }

    private func loadLicenseText() {
        if let url = Bundle.main.url(
            forResource: item.resourceName,
            withExtension: item.resourceExtension
        ) {
            do {
                licenseText = try String(contentsOf: url, encoding: .utf8)
            } catch {
                licenseText = """
                Failed to read license file.

                Error: \(error.localizedDescription)
                """
            }
        } else {
            let pathDescription = "\(item.resourceName).\(item.resourceExtension)"

            licenseText = """
            License file not found in app bundle.

            Expected path:
            \(pathDescription)
            """
        }
    }
}

struct AboutView: View {
    var body: some View {
        NavigationStack {
            List {
                NavigationLink("Licenses") {
                    LicensesView()
                }
            }
            .navigationTitle("About")
        }
    }
}
