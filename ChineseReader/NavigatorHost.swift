//
//  NavigatorHost.swift
//  ChineseReader
//
//  Created by Oliver Zhang on 2025-11-08.
//

import ReadiumNavigator
import SwiftUI

struct NavigatorHost: UIViewControllerRepresentable {
    typealias UIViewControllerType = UIViewController

    /// We pass the already-created EPUBNavigatorViewController from ReadiumEngine.
    let navigatorVC: EPUBNavigatorViewController

    func makeUIViewController(context: Context) -> UIViewController {
        // Container vc to host child VC cleanly
        let host = UIViewController()
        host.addChild(navigatorVC)
        navigatorVC.view.translatesAutoresizingMaskIntoConstraints = false
        host.view.addSubview(navigatorVC.view)

        NSLayoutConstraint.activate([
            navigatorVC.view.leadingAnchor.constraint(
                equalTo: host.view.leadingAnchor
            ),
            navigatorVC.view.trailingAnchor.constraint(
                equalTo: host.view.trailingAnchor
            ),
            navigatorVC.view.topAnchor.constraint(equalTo: host.view.topAnchor),
            navigatorVC.view.bottomAnchor.constraint(
                equalTo: host.view.bottomAnchor
            ),
        ])

        navigatorVC.didMove(toParent: host)
        return host
    }

    func updateUIViewController(
        _ uiViewController: UIViewController,
        context: Context
    ) {
        // Nothing needed for now
    }
}
