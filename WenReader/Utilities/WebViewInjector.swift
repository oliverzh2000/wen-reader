// Copyright 2025 Oliver Zhang
// Licensed under the MIT License

import Foundation
import UIKit
import WebKit

/// Handles JavaScript and CSS injection into WebViews
@MainActor
final class WebViewInjector {
    private let injectJS: String
    private let injectCSS: String
    
    /// Initialize with JS and CSS content to inject
    init(javascript: String, css: String) {
        self.injectJS = javascript
        self.injectCSS = css
        
        // Validate that resources were provided
        if javascript.isEmpty || css.isEmpty {
            Log.error("CRITICAL: WebViewInjector initialized with empty JS or CSS")
        }
    }
    
    /// Convenience initializer that loads from bundle
    convenience init?(javascriptResource: String, cssResource: String) {
        guard let jsURL = Bundle.main.url(forResource: javascriptResource, withExtension: "js"),
              let cssURL = Bundle.main.url(forResource: cssResource, withExtension: "css"),
              let js = try? String(contentsOf: jsURL, encoding: .utf8),
              let css = try? String(contentsOf: cssURL, encoding: .utf8) else {
            Log.error("Failed to load injection resources from bundle")
            return nil
        }
        
        self.init(javascript: js, css: css)
    }
    
    /// Inject helpers (JS + CSS) into all provided webviews
    func injectHelpers(into webViews: [WKWebView]) {
        // 1) Inject CSS by appending a <style> tag
        guard let cssJSON = jsonEncode(injectCSS) else {
            Log.error("Failed to encode CSS for injection")
            return
        }
        
        let cssJS = """
            (function(){
              try {
                if (!document.getElementById('cr-nonselectable-style')) {
                  const s = document.createElement('style');
                  s.id='cr-nonselectable-style';
                  s.type='text/css';
                  s.appendChild(document.createTextNode(\(cssJSON)));
                  document.head.appendChild(s);
                }
              } catch(e) {
                console.error('CR CSS injection failed:', e);
              }
            })();
            """
        
        // 2) Inject the helper JS namespace (window.CR)
        guard let jsJSON = jsonEncode(injectJS) else {
            Log.error("Failed to encode JS for injection")
            return
        }
        
        let helperJS = """
            (function(){
              try {
                const script = document.createElement('script');
                script.type = 'text/javascript';
                script.appendChild(document.createTextNode(\(jsJSON)));
                document.head.appendChild(script);
              } catch(e) {
                console.error('CR JS injection failed:', e);
              }
            })();
            """
        
        evaluateInAll(webViews: webViews, javascript: cssJS)
        evaluateInAll(webViews: webViews, javascript: helperJS)
    }
    
    /// Evaluate JavaScript in all provided webviews
    func evaluateInAll(webViews: [WKWebView], javascript: String) {
        for webView in webViews {
            webView.evaluateJavaScript(javascript, completionHandler: nil)
        }
    }
    
    /// Enable or disable text selection in all webviews
    func setSelectionEnabled(_ enabled: Bool, in webViews: [WKWebView]) {
        let js = """
            try { window.CR && window.CR.setSelectable(\(enabled ? "true" : "false")); } catch(e) { /* noop */ }
            """
        evaluateInAll(webViews: webViews, javascript: js)
    }
    
    /// Clear highlight in all webviews
    func clearHighlight(in webViews: [WKWebView]) {
        let js = """
            (function() {
              try {
                if (window.CR && window.CR.clearHighlight) {
                  return window.CR.clearHighlight();
                }
              } catch (e) {
                console.error("CR.clearHighlight error", e);
              }
              return null;
            })();
            """
        evaluateInAll(webViews: webViews, javascript: js)
    }
    
    // MARK: - Private Helpers
    
    /// JSON encode string for safe JavaScript injection
    private func jsonEncode(_ string: String) -> String? {
        guard let data = try? JSONEncoder().encode(string),
              let json = String(data: data, encoding: .utf8) else {
            return nil
        }
        return json
    }
}
