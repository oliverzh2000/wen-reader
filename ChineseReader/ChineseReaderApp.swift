import Combine
import SwiftUI
import UniformTypeIdentifiers

// MARK: - Models
struct BookItem: Identifiable, Codable, Hashable {
    let id: UUID
    var displayName: String
    /// File name of the local copy inside Application Support/Books (sandbox)
    var relativePath: String
}

// MARK: - Catalog (sandboxed copies; no security-scoped bookmarks needed on iOS)
final class CatalogStore: ObservableObject {
    @Published private(set) var books: [BookItem] = [] { didSet { persist() } }

    private let storageKey = "catalog.books.v1"

    init() { restore() }

    // Where we keep imported copies so we can reopen them later without extra permissions
    private func booksDirectory() -> URL {
        let base = FileManager.default.urls(
            for: .applicationSupportDirectory,
            in: .userDomainMask
        )[0]
        let dir = base.appendingPathComponent("Books", isDirectory: true)
        try? FileManager.default.createDirectory(
            at: dir,
            withIntermediateDirectories: true
        )
        return dir
    }

    /// Import by copying the selected EPUB into our sandbox. This avoids security-scope persistence.
    func add(url: URL) {
        let accessed = url.startAccessingSecurityScopedResource()
        defer { if accessed { url.stopAccessingSecurityScopedResource() } }
        do {
            // De-dupe by filename or title-ish display name
            let filename = url.lastPathComponent
            if let idx = books.firstIndex(where: {
                $0.displayName + ".epub" == filename
                    || $0.displayName
                        == url.deletingPathExtension().lastPathComponent
            }) {
                withAnimation {
                    books.move(fromOffsets: IndexSet(integer: idx), toOffset: 0)
                }
                return
            }

            let id = UUID()
            let dest = booksDirectory().appendingPathComponent(
                "\(id.uuidString).epub"
            )
            if !FileManager.default.fileExists(atPath: dest.path) {
                try FileManager.default.copyItem(at: url, to: dest)
            }

            let item = BookItem(
                id: id,
                displayName: url.deletingPathExtension().lastPathComponent,
                relativePath: dest.lastPathComponent
            )
            withAnimation { books.insert(item, at: 0) }
        } catch {
            print("Failed to import: \(error)")
        }
    }

    /// Remove from catalog only (does not delete the sandboxed copy to honor "don’t delete from disk").
    func remove(_ book: BookItem) {
        if let idx = books.firstIndex(of: book) {
            withAnimation {
                books.remove(at: idx)
                // Ensue Void return to silence compiler warning.
                ()
            }
        }
    }

    func localURL(for book: BookItem) -> URL {
        booksDirectory().appendingPathComponent(book.relativePath)
    }

    private func persist() {
        do {
            UserDefaults.standard.set(
                try JSONEncoder().encode(books),
                forKey: storageKey
            )
        } catch { print("Persist error: \(error)") }
    }

    private func restore() {
        guard let data = UserDefaults.standard.data(forKey: storageKey) else {
            return
        }
        do {
            books = try JSONDecoder().decode([BookItem].self, from: data)
        } catch { print("Restore error: \(error)") }
    }
}

final class ChromeState: ObservableObject {
    @Published var hideStatusBar = false
}

// MARK: - App
@main
struct ReaderSkeletonApp: App {
    @StateObject private var catalog = CatalogStore()
    @StateObject private var chrome = ChromeState()

    var body: some Scene {
        WindowGroup {
            NavigationStack { LibraryView() }
                .environmentObject(catalog)
                .environmentObject(chrome)
                .statusBarHidden(chrome.hideStatusBar)
        }
    }
}

// MARK: - Views
struct LibraryView: View {
    @EnvironmentObject private var catalog: CatalogStore
    @State private var showImporter = false
    @State private var importError: String?

    var body: some View {
        List {
            Section("Imported Books") {
                if catalog.books.isEmpty {
                    ContentUnavailableView(
                        "No books yet",
                        systemImage: "book",
                        description: Text(
                            "Tap the '+' button to add books."
                        )
                    )
                } else {
                    ForEach(catalog.books) { book in
                        NavigationLink(value: book) {
                            HStack(spacing: 12) {
                                Image(systemName: "book.closed")
                                VStack(alignment: .leading) {
                                    Text(book.displayName).font(.headline)
                                    Text("EPUB").font(.caption).foregroundStyle(
                                        .secondary
                                    )
                                }
                                Spacer()
                                Button(role: .destructive) {
                                    catalog.remove(book)
                                } label: {
                                    Image(systemName: "trash")
                                }.buttonStyle(.borderless)
                            }
                        }
                    }
                }
            }
        }
        .navigationTitle("Library")
        .toolbar {
            ToolbarItem(placement: .bottomBar) {
                Button {
                    showImporter = true
                } label: {
                    Label("Add", systemImage: "plus")
                }
            }
        }
        .navigationDestination(for: BookItem.self) { book in
            ReaderView(book: book)
        }
        .fileImporter(
            isPresented: $showImporter,
            allowedContentTypes: [.epub],
            allowsMultipleSelection: false
        ) { result in
            switch result {
            case .success(let urls):
                catalog.add(url: urls.first!)
            case .failure(let err):
                importError = err.localizedDescription
            }
        }
        .alert(
            "Import failed",
            isPresented: Binding(
                get: { importError != nil },
                set: { _ in importError = nil }
            )
        ) {
            Button("OK", role: .cancel) {}
        } message: {
            Text(importError ?? "Unknown error")
        }
        .scrollBounceBehavior(.basedOnSize)
    }
}

extension UTType {
    static let epub = UTType(importedAs: "org.idpf.epub-container")
}

// MARK: - Reader
private struct ReaderSurface: View {
    var body: some View {
        Color.red
            .ignoresSafeArea(edges: [.horizontal])  // or remove entirely, per above
            .contentShape(Rectangle())
    }
}

struct ReaderView: View {
    @EnvironmentObject private var catalog: CatalogStore
    @EnvironmentObject private var chrome: ChromeState

    let book: BookItem

    @State private var showChrome = false
    @State private var showChapters = false
    @State private var showSettings = false
    @State private var completedInitialSync = false

    var body: some View {
        ZStack {
            ReaderSurface()
        }
        .statusBarHidden(!showChrome)
        .navigationTitle(book.displayName)
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            // Tappable title
            ToolbarItem(placement: .principal) {
                Button(book.displayName) {
                    withAnimation(.easeInOut) { showChrome.toggle() }
                    chrome.hideStatusBar.toggle()
                }
                .buttonStyle(.plain)  // keeps it looking like a title
            }
            // Trailing toolbar buttons (only when chrome is visible)
            ToolbarItemGroup(placement: .topBarTrailing) {
                if showChrome {
                    // Chapters
                    Button {
                        showChapters = true
                    } label: {
                        Image(systemName: "list.bullet")
                    }
                    .labelStyle(.iconOnly)

                    // Reader display settings ("aA" style icon)
                    Button {
                        showSettings = true
                    } label: {
                        Image(systemName: "textformat.size")  // aA-style symbol
                    }
                    .labelStyle(.iconOnly)
                }
            }
        }
        .onAppear {
            guard !completedInitialSync else { return }

            // Hide by default on entry.
            chrome.hideStatusBar = !showChrome
            completedInitialSync = true
        }
        .sheet(isPresented: $showChapters) {
            PanelChaptersPlaceholder()
                .presentationDetents([.medium, .large])
        }
        .sheet(isPresented: $showSettings) {
            PanelSettingsPlaceholder()
                .presentationDetents([.medium, .large])
        }
        .navigationBarBackButtonHidden(!showChrome)
    }
}

// MARK: - Contents/Settings Sheets
struct PanelChaptersPlaceholder: View {
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            Form {
            }
            .navigationTitle("Contents")
            .navigationBarTitleDisplayMode(.inline)  // “small” title like Settings
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Done") { dismiss() }
                }
            }
        }
    }
}

struct PanelSettingsPlaceholder: View {
    @State private var showSettings = false
    @State private var fontName = "Songti SC"
    @State private var fontSize: Double = 18
    @State private var lineHeight: Double = 1.5
    @State private var margins: Double = 1.0
    @State private var justify = true
    @State private var theme = "System"

    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            Form {
                Section("Typography") {
                    Picker("Font", selection: $fontName) {
                        Text("Songti SC").tag("Songti SC")
                        Text("PingFang SC").tag("PingFang SC")
                    }
                    .pickerStyle(.automatic)

                    VStack(alignment: .leading) {
                        Text("Font Size")
                        Slider(value: $fontSize, in: 8.0...32, step: 1.0) {
                        } minimumValueLabel: {
                            Text("")
                        } maximumValueLabel: {
                            Text("\(Int(fontSize)) pt")
                                .foregroundStyle(.secondary)
                        }
                    }

                    VStack(alignment: .leading) {
                        Text("Line Height")
                        Slider(value: $lineHeight, in: 1.0...2.0, step: 0.05) {
                        } minimumValueLabel: {
                            Text("")
                        } maximumValueLabel: {
                            Text("\(String(format: "%.1f", lineHeight))")
                                .foregroundStyle(.secondary)
                        }
                    }

                    VStack(alignment: .leading) {
                        Text("Margins")
                        Slider(value: $margins, in: 0.0...1.5, step: 0.1) {
                        } minimumValueLabel: {
                            Text("")
                        } maximumValueLabel: {
                            Text("\(String(format: "%.1f", margins))")
                                .foregroundStyle(.secondary)
                        }
                    }

                    Toggle("Justify Text", isOn: $justify)
                }

                Section("Appearance") {
                    Picker("Theme", selection: $theme) {
                        Text("Light").tag("Light")
                        Text("Dark").tag("Dark")
                        Text("System").tag("System")
                    }
                    .pickerStyle(.automatic)
                }

                Section {
                    Button("Reset to Defaults", role: .destructive) {
                        // reset values
                        fontName = "Songti SC"
                        fontSize = 18
                        justify = true
                        lineHeight = 1.5
                        margins = 1.0
                        theme = "System"
                    }
                }
            }
            .navigationTitle("Reading Settings")
            .navigationBarTitleDisplayMode(.inline)  // “small” title like Settings
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Done") { dismiss() }
                }
            }
        }
    }
}

#Preview {
    NavigationStack { LibraryView() }
        .environmentObject(CatalogStore())
}
