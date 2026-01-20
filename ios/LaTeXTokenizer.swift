import Foundation

/// Decoded result from tokenizer
public struct DecodedResult {
    public let latex: String
    public let boundingBox: CGRect?
}

/// LaTeX tokenizer for iOS
/// Handles conversion between token IDs and LaTeX strings
public class LaTeXTokenizer {

    // MARK: - Properties

    /// Number of location bins for coordinate discretization
    public let numLocationBins: Int = 1000

    /// Special token IDs (relative to latex vocab start)
    public let padTokenId: Int
    public let bosTokenId: Int
    public let eosTokenId: Int
    public let unkTokenId: Int
    public let sepTokenId: Int
    public let locStartTokenId: Int
    public let locEndTokenId: Int

    /// Vocabulary
    private var idToToken: [Int: String]
    private var tokenToId: [String: Int]

    // MARK: - Initialization

    public init() {
        // Build core vocabulary
        let coreVocab = LaTeXTokenizer.buildCoreVocab()

        self.idToToken = [:]
        self.tokenToId = [:]

        for (i, token) in coreVocab.enumerated() {
            let id = numLocationBins + i
            idToToken[id] = token
            tokenToId[token] = id
        }

        // Set special token IDs
        self.padTokenId = numLocationBins + 0
        self.bosTokenId = numLocationBins + 1
        self.eosTokenId = numLocationBins + 2
        self.unkTokenId = numLocationBins + 3
        self.sepTokenId = numLocationBins + 4
        self.locStartTokenId = numLocationBins + 5
        self.locEndTokenId = numLocationBins + 6
    }

    // MARK: - Public Methods

    /// Decode token IDs to LaTeX and bounding box
    public func decode(tokenIds: [Int]) -> DecodedResult {
        var latex = ""
        var boundingBox: CGRect?

        var i = 0
        while i < tokenIds.count {
            let tokenId = tokenIds[i]

            // Skip special tokens
            if tokenId == padTokenId || tokenId == bosTokenId ||
               tokenId == eosTokenId || tokenId == sepTokenId {
                i += 1
                continue
            }

            // Handle bounding box
            if tokenId == locStartTokenId {
                // Extract 4 coordinates after <loc>
                if i + 5 < tokenIds.count && tokenIds[i + 5] == locEndTokenId {
                    let x1 = coordinateFromToken(tokenIds[i + 1])
                    let y1 = coordinateFromToken(tokenIds[i + 2])
                    let x2 = coordinateFromToken(tokenIds[i + 3])
                    let y2 = coordinateFromToken(tokenIds[i + 4])

                    boundingBox = CGRect(
                        x: CGFloat(x1),
                        y: CGFloat(y1),
                        width: CGFloat(x2 - x1),
                        height: CGFloat(y2 - y1)
                    )

                    i += 6  // Skip <loc> + 4 coords + </loc>
                    continue
                }
            }

            // Handle location end (skip if standalone)
            if tokenId == locEndTokenId {
                i += 1
                continue
            }

            // Handle location tokens (coordinate values)
            if isLocationToken(tokenId) {
                i += 1
                continue
            }

            // Handle LaTeX tokens
            if let token = idToToken[tokenId] {
                latex += token
            }

            i += 1
        }

        return DecodedResult(latex: latex.trimmingCharacters(in: .whitespaces), boundingBox: boundingBox)
    }

    /// Check if token ID is a location coordinate
    public func isLocationToken(_ tokenId: Int) -> Bool {
        return tokenId >= 0 && tokenId < numLocationBins
    }

    /// Convert coordinate (0-1) to token ID
    public func tokenFromCoordinate(_ coord: Float) -> Int {
        let clamped = max(0, min(1, coord))
        return Int(clamped * Float(numLocationBins - 1))
    }

    /// Convert token ID to coordinate (0-1)
    public func coordinateFromToken(_ tokenId: Int) -> Float {
        guard isLocationToken(tokenId) else { return 0 }
        return Float(tokenId) / Float(numLocationBins - 1)
    }

    /// Vocabulary size
    public var vocabSize: Int {
        return numLocationBins + idToToken.count
    }

    // MARK: - Private Methods

    private static func buildCoreVocab() -> [String] {
        var vocab: [String] = []

        // Special tokens (must match Python tokenizer order)
        vocab.append(contentsOf: ["<pad>", "<bos>", "<eos>", "<unk>", "<sep>", "<loc>", "</loc>"])

        // Digits
        vocab.append(contentsOf: (0...9).map { String($0) })

        // Lowercase letters
        vocab.append(contentsOf: "abcdefghijklmnopqrstuvwxyz".map { String($0) })

        // Uppercase letters
        vocab.append(contentsOf: "ABCDEFGHIJKLMNOPQRSTUVWXYZ".map { String($0) })

        // Basic operators and punctuation
        vocab.append(contentsOf: [
            "+", "-", "=", "<", ">", "(", ")", "[", "]", "{", "}",
            ",", ".", ":", ";", "!", "?", "'", "\"", "/", "\\", "|",
            "*", "^", "_", "&", "#", "@", "%", "~", "`"
        ])

        // Greek letters
        let greekLower = [
            "\\alpha", "\\beta", "\\gamma", "\\delta", "\\epsilon", "\\zeta",
            "\\eta", "\\theta", "\\iota", "\\kappa", "\\lambda", "\\mu",
            "\\nu", "\\xi", "\\pi", "\\rho", "\\sigma", "\\tau",
            "\\upsilon", "\\phi", "\\chi", "\\psi", "\\omega"
        ]
        let greekUpper = [
            "\\Alpha", "\\Beta", "\\Gamma", "\\Delta", "\\Epsilon", "\\Zeta",
            "\\Eta", "\\Theta", "\\Iota", "\\Kappa", "\\Lambda", "\\Mu",
            "\\Nu", "\\Xi", "\\Pi", "\\Rho", "\\Sigma", "\\Tau",
            "\\Upsilon", "\\Phi", "\\Chi", "\\Psi", "\\Omega"
        ]
        let greekVar = [
            "\\varepsilon", "\\vartheta", "\\varpi", "\\varrho", "\\varsigma", "\\varphi"
        ]
        vocab.append(contentsOf: greekLower)
        vocab.append(contentsOf: greekUpper)
        vocab.append(contentsOf: greekVar)

        // Common math commands
        vocab.append(contentsOf: [
            "\\frac", "\\sqrt", "\\sum", "\\prod", "\\int", "\\oint",
            "\\partial", "\\nabla", "\\infty", "\\pm", "\\mp", "\\times", "\\div",
            "\\cdot", "\\circ", "\\bullet", "\\star", "\\dagger", "\\ddagger",
            "\\leq", "\\geq", "\\neq", "\\approx", "\\equiv", "\\sim", "\\simeq",
            "\\subset", "\\supset", "\\subseteq", "\\supseteq", "\\in", "\\notin",
            "\\cup", "\\cap", "\\setminus", "\\emptyset",
            "\\forall", "\\exists", "\\neg", "\\wedge", "\\vee", "\\rightarrow",
            "\\leftarrow", "\\Rightarrow", "\\Leftarrow", "\\leftrightarrow",
            "\\to", "\\mapsto", "\\implies", "\\iff"
        ])

        // Delimiters
        vocab.append(contentsOf: [
            "\\left", "\\right", "\\langle", "\\rangle", "\\lfloor", "\\rfloor",
            "\\lceil", "\\rceil", "\\lvert", "\\rvert", "\\lVert", "\\rVert"
        ])

        // Accents and modifiers
        vocab.append(contentsOf: [
            "\\hat", "\\bar", "\\dot", "\\ddot", "\\tilde", "\\vec", "\\overline",
            "\\underline", "\\overbrace", "\\underbrace", "\\overrightarrow"
        ])

        // Spacing
        vocab.append(contentsOf: ["\\quad", "\\qquad", "\\,", "\\:", "\\;", "\\ ", "\\!"])

        // Text and formatting
        vocab.append(contentsOf: [
            "\\text", "\\textbf", "\\textit", "\\mathrm", "\\mathbf", "\\mathit",
            "\\mathcal", "\\mathbb", "\\mathfrak", "\\boldsymbol"
        ])

        // Environments
        vocab.append(contentsOf: [
            "\\begin", "\\end", "matrix", "pmatrix", "bmatrix", "vmatrix",
            "cases", "align", "aligned", "array", "equation"
        ])

        // Matrix commands
        vocab.append(contentsOf: ["\\\\", "\\hline", "\\cline", "\\multicolumn", "\\multirow"])

        // Functions
        vocab.append(contentsOf: [
            "\\lim", "\\limsup", "\\liminf", "\\max", "\\min", "\\sup", "\\inf",
            "\\log", "\\ln", "\\exp", "\\sin", "\\cos", "\\tan", "\\cot",
            "\\sec", "\\csc", "\\arcsin", "\\arccos", "\\arctan",
            "\\sinh", "\\cosh", "\\tanh", "\\coth"
        ])

        // Other symbols
        vocab.append(contentsOf: [
            "\\ldots", "\\cdots", "\\vdots", "\\ddots",
            "\\prime", "\\angle", "\\triangle", "\\square", "\\diamond",
            "\\perp", "\\parallel", "\\cong", "\\propto"
        ])

        return vocab
    }
}
