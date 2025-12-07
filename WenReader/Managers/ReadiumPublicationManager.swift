//
//  ReadiumPublicationManager.swift
//  WenReader
//
//  Created by architectural refactoring
//

import Foundation
import ReadiumAdapterGCDWebServer
import ReadiumNavigator
import ReadiumShared
import ReadiumStreamer
import SwiftUI

typealias RLink = ReadiumShared.Link

/// Manages Readium publication lifecycle (opening, navigator creation)
@MainActor
final class ReadiumPublicationManager {
    
    // Core Readium components
    private lazy var httpClient = DefaultHTTPClient()
    private lazy var httpServer = GCDHTTPServer(assetRetriever: assetRetriever)
    private lazy var assetRetriever = AssetRetriever(httpClient: httpClient)
    private lazy var publicationOpener = PublicationOpener(
        parser: DefaultPublicationParser(
            httpClient: httpClient,
            assetRetriever: assetRetriever,
            pdfFactory: DefaultPDFDocumentFactory()
        )
    )
    
    // MARK: - Public API
    
    /// Open an EPUB file and create a navigator
    func open(
        fileURL: URL,
        initialLocation: Locator?,
        sender: UIView?
    ) async -> Result<(Publication, EPUBNavigatorViewController), AppError> {
        
        guard let absoluteURL = fileURL.anyURL.absoluteURL else {
            return .failure(.bookOpenFailed(reason: "Invalid file URL"))
        }
        
        // Retrieve asset
        let assetResult = await assetRetriever.retrieve(url: absoluteURL)
        guard case .success(let asset) = assetResult else {
            let error = (assetResult as? Result<Asset, Error>)
                .flatMap { result -> Error? in
                    if case .failure(let e) = result { return e }
                    return nil
                }
            return .failure(.bookOpenFailed(
                reason: error?.localizedDescription ?? "Failed to retrieve asset"
            ))
        }
        
        // Open publication
        let openResult = await publicationOpener.open(
            asset: asset,
            allowUserInteraction: true,
            sender: sender
        )
        guard case .success(let publication) = openResult else {
            let error = (openResult as? Result<Publication, Error>)
                .flatMap { result -> Error? in
                    if case .failure(let e) = result { return e }
                    return nil
                }
            return .failure(.bookOpenFailed(
                reason: error?.localizedDescription ?? "Failed to open publication"
            ))
        }
        
        // Ensure it's an EPUB
        guard publication.conforms(to: .epub) else {
            return .failure(.invalidEPUB(reason: "Only EPUB format is supported"))
        }
        
        // Get resources for font loading
        guard let resources = Bundle.main.resourceURL else {
            return .failure(.bookOpenFailed(reason: "Failed to access app resources"))
        }
        
        // Create navigator
        do {
            let navigator = try EPUBNavigatorViewController(
                publication: publication,
                initialLocation: initialLocation,
                config: .init(
                    fontFamilyDeclarations: [
                        CSSFontFamilyDeclaration(
                            fontFamily: .notoSerifSC,
                            fontFaces: [
                                CSSFontFace(
                                    file: FileURL(
                                        url: resources.appendingPathComponent(
                                            "NotoSerifSC-VariableFont_wght.ttf"
                                        )
                                    )!,
                                    style: .normal,
                                    weight: .variable(200...900)
                                )
                            ]
                        ).eraseToAnyHTMLFontFamilyDeclaration()
                    ]
                ),
                httpServer: httpServer
            )
            
            return .success((publication, navigator))
        } catch {
            return .failure(.bookOpenFailed(reason: error.localizedDescription))
        }
    }
}
