//
//  MetadataLoader.swift
//  ChineseReader
//
//  Created by Oliver Zhang on 2025-11-30.
//

import Foundation
import ReadiumShared
import ReadiumStreamer
import UIKit


struct BookMetadata {
    var title: String?
    var authors: [String]
    var canonicalID: String?
    var cover: UIImage?
}

enum EpubMetadataLoader {
    static func load(from url: URL) async -> BookMetadata? {
        do {
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
            switch await assetRetriever.retrieve(url: url.anyURL.absoluteURL!) {
            case .success(let asset):
                // Open a `Publication` from the `Asset`.
                switch await publicationOpener.open(asset: asset, allowUserInteraction: true, sender: nil) {
                case .success(let publication):
                    
                    let coverResult = await publication.cover()
                    let cover: UIImage?
                    switch coverResult {
                    case .success(let image):
                        cover = image
                    case .failure:
                        cover = nil
                    }
                    let title = publication.metadata.title
                    let authors = publication.metadata.authors.map(\.name)
                    let canonicalID = publication.metadata.identifier
                    
                    return BookMetadata(title: title, authors: authors, canonicalID: canonicalID, cover: cover)
                    
                case .failure(let error):
                    // Failed to access or parse the publication
                    print("pase fail")
                    return nil
                }
                
            case .failure(let error):
                // Failed to retrieve the asset
                print(error)
                return nil
            }
        }
    }
}
