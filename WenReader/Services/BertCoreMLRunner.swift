// Copyright 2025 Oliver Zhang
// Licensed under the MIT License

import CoreML
import Foundation

/// Shared infrastructure for CoreML BERT-style model inference.
///
/// Handles model loading, tokenization, padding to the fixed sequence length,
/// and running predictions. Domain-specific logic (span scoring, WSD)
/// lives in the respective service classes.
final class BertCoreMLRunner {

    let model: MLModel
    let tokenizer: BertTokenizer
    let sequenceLength: Int

    /// Overhead tokens added by the tokenizer: [CLS] + [SEP].
    static let specialTokenCount = 2

    /// Maximum characters of text that fit in the model input.
    var maxInputChars: Int {
        sequenceLength - Self.specialTokenCount
    }

    /// Initialize with a CoreML model and tokenizer from the app bundle.
    ///
    /// - Parameters:
    ///   - modelName: CoreML model resource name (without `.mlmodelc` extension).
    ///   - vocabName: Vocab file resource name for the BertTokenizer.
    ///   - sequenceLength: Fixed sequence length the model expects.
    ///   - computeUnits: Compute units for model execution. Defaults to `.all`.
    /// - Throws: If the model or vocab file cannot be loaded.
    init(
        modelName: String,
        vocabName: String,
        sequenceLength: Int,
        computeUnits: MLComputeUnits = .all
    ) throws {
        guard let modelURL = Bundle.main.url(forResource: modelName, withExtension: "mlmodelc") else {
            throw BertCoreMLRunnerError.modelNotFound(modelName)
        }
        let config = MLModelConfiguration()
        config.computeUnits = computeUnits
        self.model = try MLModel(contentsOf: modelURL, configuration: config)
        self.tokenizer = try BertTokenizer(resourceName: vocabName)
        self.sequenceLength = sequenceLength
    }

    /// Internal initializer for testing with pre-built components.
    init(model: MLModel, tokenizer: BertTokenizer, sequenceLength: Int) {
        self.model = model
        self.tokenizer = tokenizer
        self.sequenceLength = sequenceLength
    }

    // MARK: - Inference

    /// Tokenize text, pad to fixed sequence length, and run inference.
    ///
    /// - Parameters:
    ///   - text: Input text to tokenize and encode.
    ///   - outputKey: The key name for the output feature in the CoreML model.
    /// - Returns: The output `MLMultiArray`, or nil on failure.
    func predict(text: String, outputKey: String) -> MLMultiArray? {
        let encoded = tokenizer.encode(text)
        return predict(inputIds: encoded.inputIds, outputKey: outputKey)
    }

    /// Pad pre-tokenized input to fixed sequence length and run inference.
    /// The model computes attention_mask internally from input_ids (non-zero → 1).
    ///
    /// - Parameters:
    ///   - inputIds: Token IDs including [CLS] and [SEP]. Padding positions must be 0.
    ///   - outputKey: The key name for the output feature in the CoreML model.
    /// - Returns: The output `MLMultiArray`, or nil on failure.
    func predict(inputIds: [Int32], outputKey: String) -> MLMultiArray? {
        let seqLen = inputIds.count

        guard seqLen <= sequenceLength else {
            Log.error("BertCoreMLRunner: token count \(seqLen) exceeds model sequence length \(sequenceLength)")
            return nil
        }

        do {
            let idsArr = try MLMultiArray(shape: [1, NSNumber(value: sequenceLength)], dataType: .int32)

            for i in 0..<seqLen {
                idsArr[[0, NSNumber(value: i)] as [NSNumber]] = NSNumber(value: inputIds[i])
            }
            // Pad with 0 (model treats 0 as padding via internal attention_mask)
            for i in seqLen..<sequenceLength {
                idsArr[[0, NSNumber(value: i)] as [NSNumber]] = 0
            }

            let provider = try MLDictionaryFeatureProvider(dictionary: [
                "input_ids": MLFeatureValue(multiArray: idsArr),
            ])

            let output = try model.prediction(from: provider)

            guard let result = output.featureValue(for: outputKey)?.multiArrayValue else {
                Log.error("BertCoreMLRunner: output missing key '\(outputKey)'")
                return nil
            }
            return result
        } catch {
            Log.error("BertCoreMLRunner: prediction failed: \(error.localizedDescription)")
            return nil
        }
    }

    /// Run inference and return output as a Float array.
    ///
    /// - Parameters:
    ///   - inputIds: Token IDs including [CLS] and [SEP]. Padding positions must be 0.
    ///   - outputKey: The key name for the output feature in the CoreML model.
    ///   - maxFloats: If set, only read this many floats from the output (useful for
    ///     ignoring padding positions in sequence outputs).
    /// - Returns: Float array, or nil on failure.
    func predictFloats(inputIds: [Int32], outputKey: String, maxFloats: Int? = nil) -> [Float]? {
        guard let arr = predict(inputIds: inputIds, outputKey: outputKey) else {
            return nil
        }
        let count = maxFloats ?? arr.count
        guard count <= arr.count else { return nil }
        var floats = [Float](repeating: 0, count: count)
        for i in 0..<count { floats[i] = arr[i].floatValue }
        return floats
    }
}

// MARK: - Errors

enum BertCoreMLRunnerError: Error, LocalizedError {
    case modelNotFound(String)

    var errorDescription: String? {
        switch self {
        case .modelNotFound(let name):
            return "CoreML model '\(name).mlmodelc' not found in app bundle"
        }
    }
}
