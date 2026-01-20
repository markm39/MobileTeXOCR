import Foundation
import CoreML
import Vision
import UIKit

/// Result from LaTeX OCR recognition
public struct LaTeXOCRResult {
    /// Recognized LaTeX string
    public let latex: String

    /// Bounding box in normalized coordinates (0-1)
    public let boundingBox: CGRect?

    /// Confidence score (0-1)
    public let confidence: Float
}

/// Handwritten LaTeX OCR using CoreML
public class LaTeXOCR {

    // MARK: - Properties

    private let encoder: MLModel
    private let decoder: MLModel
    private let tokenizer: LaTeXTokenizer

    private let imageSize: Int = 384
    private let maxSequenceLength: Int = 256

    // MARK: - Initialization

    /// Initialize with CoreML model paths
    /// - Parameters:
    ///   - encoderPath: Path to encoder .mlpackage
    ///   - decoderPath: Path to decoder .mlpackage
    public init(encoderPath: URL, decoderPath: URL) throws {
        let config = MLModelConfiguration()
        config.computeUnits = .all  // Use Neural Engine when available

        self.encoder = try MLModel(contentsOf: encoderPath, configuration: config)
        self.decoder = try MLModel(contentsOf: decoderPath, configuration: config)
        self.tokenizer = LaTeXTokenizer()
    }

    /// Initialize with bundled models
    public convenience init() throws {
        guard let encoderURL = Bundle.main.url(forResource: "latex_ocr_encoder", withExtension: "mlpackage"),
              let decoderURL = Bundle.main.url(forResource: "latex_ocr_decoder", withExtension: "mlpackage") else {
            throw LaTeXOCRError.modelNotFound
        }
        try self.init(encoderPath: encoderURL, decoderPath: decoderURL)
    }

    // MARK: - Public Methods

    /// Recognize LaTeX from an image
    /// - Parameter image: Input UIImage
    /// - Returns: Recognition result with LaTeX and bounding box
    public func recognize(image: UIImage) throws -> LaTeXOCRResult {
        // Preprocess image
        guard let pixelBuffer = preprocessImage(image) else {
            throw LaTeXOCRError.preprocessingFailed
        }

        // Run encoder
        let encoderFeatures = try runEncoder(pixelBuffer: pixelBuffer)

        // Run decoder (autoregressive)
        let (tokenIds, confidence) = try runDecoder(encoderFeatures: encoderFeatures)

        // Decode tokens to LaTeX
        let result = tokenizer.decode(tokenIds: tokenIds)

        return LaTeXOCRResult(
            latex: result.latex,
            boundingBox: result.boundingBox,
            confidence: confidence
        )
    }

    /// Recognize LaTeX from a CVPixelBuffer (for camera integration)
    /// - Parameter pixelBuffer: Camera frame
    /// - Returns: Recognition result
    public func recognize(pixelBuffer: CVPixelBuffer) throws -> LaTeXOCRResult {
        let resizedBuffer = resizePixelBuffer(pixelBuffer, to: CGSize(width: imageSize, height: imageSize))

        let encoderFeatures = try runEncoder(pixelBuffer: resizedBuffer ?? pixelBuffer)
        let (tokenIds, confidence) = try runDecoder(encoderFeatures: encoderFeatures)
        let result = tokenizer.decode(tokenIds: tokenIds)

        return LaTeXOCRResult(
            latex: result.latex,
            boundingBox: result.boundingBox,
            confidence: confidence
        )
    }

    // MARK: - Private Methods

    private func preprocessImage(_ image: UIImage) -> CVPixelBuffer? {
        // Resize to target size
        let targetSize = CGSize(width: imageSize, height: imageSize)

        UIGraphicsBeginImageContextWithOptions(targetSize, true, 1.0)
        defer { UIGraphicsEndImageContext() }

        // Draw with white background (for handwriting on white paper)
        UIColor.white.setFill()
        UIRectFill(CGRect(origin: .zero, size: targetSize))

        // Calculate aspect-fit rect
        let aspectRatio = image.size.width / image.size.height
        var drawRect: CGRect

        if aspectRatio > 1 {
            let height = targetSize.width / aspectRatio
            drawRect = CGRect(x: 0, y: (targetSize.height - height) / 2, width: targetSize.width, height: height)
        } else {
            let width = targetSize.height * aspectRatio
            drawRect = CGRect(x: (targetSize.width - width) / 2, y: 0, width: width, height: targetSize.height)
        }

        image.draw(in: drawRect)

        guard let resizedImage = UIGraphicsGetImageFromCurrentImageContext(),
              let cgImage = resizedImage.cgImage else {
            return nil
        }

        // Convert to pixel buffer
        return createPixelBuffer(from: cgImage)
    }

    private func createPixelBuffer(from cgImage: CGImage) -> CVPixelBuffer? {
        let attributes: [String: Any] = [
            kCVPixelBufferCGImageCompatibilityKey as String: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey as String: true
        ]

        var pixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            cgImage.width,
            cgImage.height,
            kCVPixelFormatType_32ARGB,
            attributes as CFDictionary,
            &pixelBuffer
        )

        guard status == kCVReturnSuccess, let buffer = pixelBuffer else {
            return nil
        }

        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }

        guard let context = CGContext(
            data: CVPixelBufferGetBaseAddress(buffer),
            width: cgImage.width,
            height: cgImage.height,
            bitsPerComponent: 8,
            bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue
        ) else {
            return nil
        }

        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: cgImage.width, height: cgImage.height))

        return buffer
    }

    private func resizePixelBuffer(_ pixelBuffer: CVPixelBuffer, to size: CGSize) -> CVPixelBuffer? {
        // Use Vision for efficient resizing
        var resizedBuffer: CVPixelBuffer?

        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        let scale = min(size.width / ciImage.extent.width, size.height / ciImage.extent.height)
        let scaledImage = ciImage.transformed(by: CGAffineTransform(scaleX: scale, y: scale))

        // Create output buffer
        let attributes: [String: Any] = [
            kCVPixelBufferCGImageCompatibilityKey as String: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey as String: true
        ]

        CVPixelBufferCreate(
            kCFAllocatorDefault,
            Int(size.width),
            Int(size.height),
            kCVPixelFormatType_32BGRA,
            attributes as CFDictionary,
            &resizedBuffer
        )

        if let buffer = resizedBuffer {
            let context = CIContext()
            context.render(scaledImage, to: buffer)
        }

        return resizedBuffer
    }

    private func runEncoder(pixelBuffer: CVPixelBuffer) throws -> MLMultiArray {
        // Create input
        guard let input = try? MLDictionaryFeatureProvider(dictionary: ["image": pixelBuffer]) else {
            throw LaTeXOCRError.encoderFailed
        }

        // Run inference
        let output = try encoder.prediction(from: input)

        guard let features = output.featureValue(for: "features")?.multiArrayValue else {
            throw LaTeXOCRError.encoderFailed
        }

        return features
    }

    private func runDecoder(encoderFeatures: MLMultiArray) throws -> ([Int], Float) {
        var tokenIds: [Int] = [tokenizer.bosTokenId]
        var totalLogProb: Float = 0.0

        for _ in 0..<maxSequenceLength {
            // Create input arrays
            let inputIds = try createInputArray(from: tokenIds)

            let input = try MLDictionaryFeatureProvider(dictionary: [
                "encoder_features": encoderFeatures,
                "input_ids": inputIds
            ])

            // Run decoder step
            let output = try decoder.prediction(from: input)

            guard let logits = output.featureValue(for: "logits")?.multiArrayValue else {
                throw LaTeXOCRError.decoderFailed
            }

            // Get next token (greedy decoding)
            let (nextToken, logProb) = argmaxWithProb(logits)
            tokenIds.append(nextToken)
            totalLogProb += logProb

            // Check for end of sequence
            if nextToken == tokenizer.eosTokenId {
                break
            }
        }

        let avgConfidence = exp(totalLogProb / Float(tokenIds.count))
        return (tokenIds, avgConfidence)
    }

    private func createInputArray(from tokenIds: [Int]) throws -> MLMultiArray {
        let array = try MLMultiArray(shape: [1, NSNumber(value: tokenIds.count)], dataType: .int32)

        for (i, tokenId) in tokenIds.enumerated() {
            array[[0, NSNumber(value: i)]] = NSNumber(value: tokenId)
        }

        return array
    }

    private func argmaxWithProb(_ array: MLMultiArray) -> (Int, Float) {
        let count = array.count
        var maxIdx = 0
        var maxVal: Float = -Float.infinity

        for i in 0..<count {
            let val = array[i].floatValue
            if val > maxVal {
                maxVal = val
                maxIdx = i
            }
        }

        // Compute softmax probability for confidence
        var sumExp: Float = 0
        for i in 0..<count {
            sumExp += exp(array[i].floatValue - maxVal)
        }
        let prob = 1.0 / sumExp  // exp(maxVal - maxVal) / sumExp

        return (maxIdx, log(prob))
    }
}

// MARK: - Errors

public enum LaTeXOCRError: Error {
    case modelNotFound
    case preprocessingFailed
    case encoderFailed
    case decoderFailed
}
