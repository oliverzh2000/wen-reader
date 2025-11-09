//
//  ReadiumEngine.swift
//  ChineseReader
//
//  Created by Oliver Zhang on 2025-11-08.
//

import Combine
import ReadiumAdapterGCDWebServer
import ReadiumNavigator
import ReadiumShared
import ReadiumStreamer
import SwiftUI

typealias RLink = ReadiumShared.Link

@MainActor
final class ReadiumEngine: ObservableObject {
    // MARK: Outputs for the UI
    @Published var publication: Publication?
    @Published var navigatorVC: EPUBNavigatorViewController?
    @Published var openError: Error?
    @Published var isOpening: Bool = false
    @Published var currentLocation: Locator?

    // MARK: Core components (kept for the app lifetime of this engine)
    private lazy var httpClient = DefaultHTTPClient()
    // Use Readium’s GCD-based HTTP server; this serves resources to the navigator
    private lazy var httpServer = GCDHTTPServer(assetRetriever: assetRetriever)
    private lazy var assetRetriever = AssetRetriever(httpClient: httpClient)
    private lazy var publicationOpener = PublicationOpener(
        parser: DefaultPublicationParser(
            httpClient: httpClient,
            assetRetriever: assetRetriever,
            pdfFactory: DefaultPDFDocumentFactory()
        )
    )

    // Book identity for persistence
    private var bookId: UUID?
    private var lastSavedAt = Date.distantPast

    // MARK: Open
    func open(bookId: UUID, fileURL: URL, sender: UIView?) async {
        guard !isOpening else { return }
        self.bookId = bookId
        isOpening = true
        openError = nil

        do {
            let assetResult = await assetRetriever.retrieve(
                url: fileURL.anyURL.absoluteURL!
            )
            guard case .success(let asset) = assetResult else {
                if case .failure(let e) = assetResult { throw e }
                return
            }

            let openResult = await publicationOpener.open(
                asset: asset,
                // If you add LCP later, set to true so the toolkit may prompt
                allowUserInteraction: true,
                sender: sender
            )
            guard case .success(let pub) = openResult else {
                if case .failure(let e) = openResult { throw e }
                return
            }

            self.publication = pub

            // Decide profile; we’re targeting EPUB here
            guard pub.conforms(to: .epub) else {
                throw NSError(
                    domain: "Reader",
                    code: 1,
                    userInfo: [
                        NSLocalizedDescriptionKey:
                            "Unsupported format (not EPUB)."
                    ]
                )
            }

            let initial = loadLastLocation()
            let navigator = try EPUBNavigatorViewController(
                publication: pub,
                initialLocation: initial,
                httpServer: httpServer
            )
            navigator.delegate = self
            self.navigatorVC = navigator

        } catch {
            self.openError = error
        }

        isOpening = false
    }

    // MARK: Navigation helpers you can call from SwiftUI buttons
    func go(to link: RLink) async {
        await navigatorVC?.go(to: link)
    }

    func go(to locator: Locator) async {
        await navigatorVC?.go(to: locator)
    }

    // MARK: Persistence for last location
    private func saveLastLocation(_ locator: Locator) {
        guard let id = bookId else { return }
        // Throttle writes a bit (avoid writing several times per second)
        let now = Date()
        guard now.timeIntervalSince(lastSavedAt) > 0.5 else { return }
        lastSavedAt = now

        let key = "lastLocation.\(id.uuidString)"
        UserDefaults.standard.set(locator.jsonString, forKey: key)
    }

    private func loadLastLocation() -> Locator? {
        guard let id = bookId else { return nil }
        let key = "lastLocation.\(id.uuidString)"
        guard let s = UserDefaults.standard.string(forKey: key) else {
            return nil
        }
        return try? Locator(jsonString: s)
    }
}

// MARK: - NavigatorDelegate
extension ReadiumEngine: EPUBNavigatorDelegate {
    func navigator(
        _ navigator: any ReadiumNavigator.Navigator,
        presentError error: ReadiumNavigator.NavigatorError
    ) {
    }

    func navigator(_ navigator: Navigator, locationDidChange locator: Locator) {
        currentLocation = locator
        saveLastLocation(locator)
    }
}
