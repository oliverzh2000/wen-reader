// Copyright 2025 Oliver Zhang
// Licensed under the MIT License

import SwiftUI
import ReadiumNavigator

/// Key action callbacks for Mac Catalyst keyboard handling.
struct KeyActions {
    var onArrowLeft: (() -> Void)?
    var onArrowRight: (() -> Void)?
    var onArrowUp: (() -> Void)?
    var onArrowDown: (() -> Void)?
    var onSpace: (() -> Void)?
    var onEscape: (() -> Void)?
}

/// UIKit container that hosts the EPUB navigator and calls `onLayout`
/// every time its view lays out subviews.
final class NavigatorHostController: UIViewController {
    let navigatorVC: EPUBNavigatorViewController
    let onLayout: (() -> Void)?
    var keyActions = KeyActions()

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

    override var canBecomeFirstResponder: Bool { true }

    override var keyCommands: [UIKeyCommand]? {
        guard ProcessInfo.processInfo.isMacCatalystApp else { return nil }
        let cmds = [
            UIKeyCommand(input: UIKeyCommand.inputLeftArrow, modifierFlags: [], action: #selector(handleKey(_:))),
            UIKeyCommand(input: UIKeyCommand.inputRightArrow, modifierFlags: [], action: #selector(handleKey(_:))),
            UIKeyCommand(input: UIKeyCommand.inputUpArrow, modifierFlags: [], action: #selector(handleKey(_:))),
            UIKeyCommand(input: UIKeyCommand.inputDownArrow, modifierFlags: [], action: #selector(handleKey(_:))),
            UIKeyCommand(input: " ", modifierFlags: [], action: #selector(handleKey(_:))),
            UIKeyCommand(input: UIKeyCommand.inputEscape, modifierFlags: [], action: #selector(handleKey(_:))),
        ]
        for cmd in cmds { cmd.wantsPriorityOverSystemBehavior = true }
        return cmds
    }

    @objc private func handleKey(_ sender: UIKeyCommand) {
        switch sender.input {
        case UIKeyCommand.inputLeftArrow:  keyActions.onArrowLeft?()
        case UIKeyCommand.inputRightArrow: keyActions.onArrowRight?()
        case UIKeyCommand.inputUpArrow:    keyActions.onArrowUp?()
        case UIKeyCommand.inputDownArrow:  keyActions.onArrowDown?()
        case " ":                          keyActions.onSpace?()
        case UIKeyCommand.inputEscape:     keyActions.onEscape?()
        default: break
        }
    }

    override func viewDidLoad() {
        super.viewDidLoad()

        // Embed the Readium navigator as a child VC
        addChild(navigatorVC)
        navigatorVC.view.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(navigatorVC.view)

        NSLayoutConstraint.activate([
            navigatorVC.view.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            navigatorVC.view.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            navigatorVC.view.topAnchor.constraint(equalTo: view.topAnchor),
            navigatorVC.view.bottomAnchor.constraint(equalTo: view.bottomAnchor),
        ])

        navigatorVC.didMove(toParent: self)
    }

    override func viewDidAppear(_ animated: Bool) {
        super.viewDidAppear(animated)
        becomeFirstResponder()
        // Readium's internal web views may set up constraints after the first layout pass.
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) { [weak self] in
            self?.onLayout?()
        }
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
    var keyActions: KeyActions

    init(
        navigatorVC: EPUBNavigatorViewController,
        onLayout: (() -> Void)? = nil,
        keyActions: KeyActions = KeyActions()
    ) {
        self.navigatorVC = navigatorVC
        self.onLayout = onLayout
        self.keyActions = keyActions
    }

    func makeUIViewController(context: Context) -> NavigatorHostController {
        let controller = NavigatorHostController(navigatorVC: navigatorVC, onLayout: onLayout)
        controller.keyActions = keyActions
        return controller
    }

    func updateUIViewController(_ uiViewController: NavigatorHostController, context: Context) {
        uiViewController.keyActions = keyActions
    }
}
