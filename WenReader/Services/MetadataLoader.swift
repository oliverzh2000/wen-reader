// Copyright 2025 Oliver Zhang
// Licensed under the MIT License

import Foundation
import ReadiumShared
import ReadiumStreamer
import UIKit
enum EpubMetadataLoader {
    static func load(from url: URL) async -> BookMetadata? {
        // Instantiate the required components.
        let httpClient = DefaultHTTPClient()
        let assetRetriever = AssetRetriever(
            httpClient: httpClient
        )
        let publicationOpener = PublicationOpener(
            parser: DefaultPublicationParser(
                httpClient: httpClient,
                assetRetriever: assetRetriever,
                pdfFactory: DefaultPDFDocumentFactory()
            )
        )
        
        // Retrieve an `Asset` to access the file content.
        guard let absoluteURL = url.anyURL.absoluteURL else {
            Log.error("Failed to get absolute URL for: \(url)")
            return nil
        }
        
        let assetResult = await assetRetriever.retrieve(url: absoluteURL)
        guard case .success(let asset) = assetResult else {
            if case .failure(let error) = assetResult {
                Log.error("Failed to retrieve asset: \(error)")
            }
            return nil
        }
        
        // Open a `Publication` from the `Asset`.
        let publicationResult = await publicationOpener.open(
            asset: asset,
            allowUserInteraction: true,
            sender: nil
        )
        guard case .success(let publication) = publicationResult else {
            if case .failure(let error) = publicationResult {
                Log.error("Failed to open publication: \(error)")
            }
            return nil
        }
        
        // Extract cover image (non-fatal if missing)
        let coverResult = await publication.cover()
        let cover: UIImage?
        switch coverResult {
        case .success(let image):
            cover = image
        case .failure(let error):
            Log.info("No cover image available: \(error)")
            cover = nil
        }
        
        let title = publication.metadata.title
        let authors = publication.metadata.authors.map(\.name)
        let canonicalID = publication.metadata.identifier
        
        // Compute page count from positions list
        let pageCount: Int?
        switch await publication.positions() {
        case .success(let positions):
            pageCount = positions.count
        case .failure:
            pageCount = nil
        }
        
        return BookMetadata(
            title: title,
            authors: authors,
            canonicalID: canonicalID,
            cover: cover,
            pageCount: pageCount
        )
    }
}
