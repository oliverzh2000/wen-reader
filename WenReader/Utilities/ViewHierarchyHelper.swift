//
//  ViewHierarchyHelper.swift
//  WenReader
//
//  Created by refactoring on 2025-12-10.
//

import UIKit
import WebKit

/// Utility for traversing and finding views in the hierarchy
enum ViewHierarchyHelper {
    /// Recursively traverse view hierarchy and apply visitor to each view
    static func traverse(_ root: UIView, visitor: (UIView) -> Void) {
        visitor(root)
        for subview in root.subviews {
            traverse(subview, visitor: visitor)
        }
    }
    
    /// Find all WKWebView instances in the view hierarchy
    static func findWebViews(in root: UIView) -> [WKWebView] {
        var result: [WKWebView] = []
        traverse(root) { view in
            if let webView = view as? WKWebView {
                result.append(webView)
            }
        }
        return result
    }
    
    /// Enable or disable scrolling for all UIScrollView instances in the hierarchy
    static func setScrollingEnabled(_ enabled: Bool, in root: UIView) {
        traverse(root) { view in
            if let scrollView = view as? UIScrollView {
                scrollView.isScrollEnabled = enabled
            }
        }
    }
}
