//
//  NavigatorHost.swift
//  ChineseReader
//
//  Created by Oliver Zhang on 2025-11-08.
//

import SwiftUI
import ReadiumNavigator

/// UIKit container that hosts the EPUB navigator and calls `onLayout`
/// every time its view lays out subviews.
final class NavigatorHostController: UIViewController {
    let navigatorVC: EPUBNavigatorViewController
    let onLayout: (() -> Void)?

    init(
        navigatorVC: EPUBNavigatorViewController,
        onLayout: (() -> Void)? = nil
    ) {
        self.navigatorVC = navigatorVC
        self.onLayout = onLayout
        super.init(nibName: nil, bundle: nil)
    }

    @available(*, unavailable)
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    override func viewDidLoad() {
        super.viewDidLoad()

        // Embed the Readium navigator as a child VC
        addChild(navigatorVC)
        navigatorVC.view.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(navigatorVC.view)

        NSLayoutConstraint.activate([
            navigatorVC.view.leadingAnchor.constraint(
                equalTo: view.leadingAnchor
            ),
            navigatorVC.view.trailingAnchor.constraint(
                equalTo: view.trailingAnchor
            ),
            navigatorVC.view.topAnchor.constraint(
                equalTo: view.topAnchor
            ),
            navigatorVC.view.bottomAnchor.constraint(
                equalTo: view.bottomAnchor
            ),
        ])

        navigatorVC.didMove(toParent: self)
    }

    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        // Called on every layout pass (rotation, sheet appear, etc.)
        onLayout?()
    }
}

/// SwiftUI wrapper that exposes the host controller.
struct NavigatorHost: UIViewControllerRepresentable {
    typealias UIViewControllerType = NavigatorHostController

    let navigatorVC: EPUBNavigatorViewController
    let onLayout: (() -> Void)?

    init(
        navigatorVC: EPUBNavigatorViewController,
        onLayout: (() -> Void)? = nil
    ) {
        self.navigatorVC = navigatorVC
        self.onLayout = onLayout
    }

    func makeUIViewController(
        context: Context
    ) -> NavigatorHostController {
        NavigatorHostController(
            navigatorVC: navigatorVC,
            onLayout: onLayout
        )
    }

    func updateUIViewController(
        _ uiViewController: NavigatorHostController,
        context: Context
    ) {
        // Nothing for now
    }
}
