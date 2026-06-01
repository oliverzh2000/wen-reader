// Copyright 2025 Oliver Zhang
// Licensed under the MIT License

/// All dictionary entries returned for a lookup.
/// Typically: all possible readings (pronunciations) and meanings for a given written form.
struct DictionaryResult {
    let entries: [Entry]
}

/// One pronunciation of a specific written form (unique trad/simp/pinyin triple in CC-CEDICT).
struct Entry {
    let traditional: String
    let simplified: String
    let accentedPinyin: [String]
    let senses: [Sense]
}

/// One logically distinct meaning for this pronunciation.
struct Sense {
    // Each gloss can be made up of plain text and clickable links.
    let glosses: [Gloss]
    let isClassifier: Bool
    /// True when WSD ranked this sense in the top cluster for its entry.
    var isPrimary: Bool = true
}

/// A single gloss, made up of fragments.
struct Gloss: Hashable {
    let fragments: [GlossFragment]
}

/// A piece of a gloss: either plain text, pinyin, or link to another headword.
enum GlossFragment: Hashable {
    case text(String)
    case accentedPinyin([String])
    case link(LinkedHeadword)
}

/// A cross-reference like '個|个'.
/// Note that the pinyin fragment is parsed separately and will not form part of the clickable link.
struct LinkedHeadword: Hashable {
    let traditional: String
    let simplified: String
}
