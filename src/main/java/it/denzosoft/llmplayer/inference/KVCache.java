package it.denzosoft.llmplayer.inference;

import it.denzosoft.llmplayer.tensor.VectorOps;
import it.denzosoft.llmplayer.tensor.VectorOpsFactory;


/**
 * Key-Value cache for autoregressive generation. Pre-allocated per layer, stores projected K and V
 * vectors for all past positions.
 *
 * <p>Supports three storage modes:
 * <ul>
 *   <li>{@link Mode#FLOAT32} — plain {@code float[]} (4 bytes/elem, default).
 *   <li>{@link Mode#Q8_0} — block-quantized int8 with FP32 scales (1.125 bytes/elem, ~3.56× smaller).
 *       Uses the same Q8_0 block layout as llama.cpp: 32 elements per block, one FP32 scale per
 *       block (2 bytes in llama.cpp's FP16 scale; we use FP32 for Java ergonomics — still saves
 *       3.56×). Enabled via {@code -Dkv.q8=true} JVM flag (see {@link InferenceState}).
 *   <li>{@link Mode#Q4_1} — block-quantized 4-bit unsigned with per-block FP32 scale and min
 *       (0.75 bytes/elem, ~5.33× smaller). Reconstruction is {@code val = q * d + m} with
 *       {@code q ∈ [0, 15]}. Best fit for KV cache because K/V activations are not zero-mean.
 *       Enabled via {@code -Dkv.q4=true} JVM flag. Larger savings, higher quant noise — bandwidth-bound
 *       attention paths (DeepSeek2 MLA, long-context dense models) benefit the most.
 * </ul>
 *
 * <p>When in Q8 mode, {@link #keyLayer(int)} and {@link #valueLayer(int)} throw — callers must
 * use {@link #storeK}, {@link #storeV}, {@link #dotK}, {@link #saxpyV} which transparently
 * quantize on write and dequantize on read. {@link Attention} has been refactored to use only
 * these methods.
 *
 * <p>Scope: this is the cache used by the standard {@code InferenceEngine} (Llama, Qwen2/3,
 * Mistral, Phi, Gemma 2/3, Granite, etc.). The MoE/hybrid engines (DeepSeek2, Qwen3MoE, Qwen3.5,
 * Nemotron-H, Gemma4) have their own state classes with independent KV storage and are not yet
 * covered by this mode.
 */
public class KVCache {

    /** Block size for Q8_0 quantization (same as llama.cpp). */
    public static final int Q8_BLOCK = 32;

    /** Block size for Q4_1 quantization (same as llama.cpp). */
    public static final int Q4_BLOCK = 32;

    /** Strategy interface used internally for Q8 dequant ops; SIMD impl plugs in via reflection. */
    public interface Q8Ops {
        float dot(float[] q, int qOff, byte[] kQuants, int kOff,
                  float[] kScales, int kScalesOff, int n);
        void  saxpy(float weight, byte[] vQuants, int vOff,
                    float[] vScales, int vScalesOff, float[] out, int outOff, int n);
        void  quantize(float[] src, int srcOff, int n,
                       byte[] dst, int dstOff, float[] scales, int scalesOff);
    }

    private static final Q8Ops Q8 = loadQ8Ops();

    /**
     * Probe for the Java 21 SIMD impl ({@code SimdQ8KvOps}); fall back to scalar.
     * {@link SimdQ8KvOps} implements {@link Q8Ops} so that the per-call dispatch goes
     * through a direct interface invocation (JIT-inlinable), not reflection. The
     * Java 8 build excludes {@code java21/} so {@code Class.forName} returns null and
     * we use {@link #SCALAR_Q8}.
     */
    private static Q8Ops loadQ8Ops() {
        try {
            Class<?> cls = Class.forName("it.denzosoft.llmplayer.inference.SimdQ8KvOps");
            Q8Ops impl = (Q8Ops) cls.getDeclaredConstructor().newInstance();
            return impl;
        } catch (Throwable t) {
            return SCALAR_Q8;
        }
    }

    private static final Q8Ops SCALAR_Q8 = new Q8Ops() {
        public float dot(float[] q, int qOff, byte[] kQ, int kOff,
                         float[] kS, int kSOff, int n) {
            return dotQ8BlockScalar(q, qOff, kQ, kOff, kS, kSOff, n);
        }
        public void saxpy(float w, byte[] vQ, int vOff, float[] vS, int vSOff,
                          float[] out, int outOff, int n) {
            saxpyQ8BlockScalar(w, vQ, vOff, vS, vSOff, out, outOff, n);
        }
        public void quantize(float[] src, int srcOff, int n,
                             byte[] dst, int dstOff, float[] scales, int sOff) {
            quantizeBlocksScalar(src, srcOff, n, dst, dstOff, scales, sOff);
        }
    };

    public enum Mode { FLOAT32, Q8_0, Q4_1 }

    private final Mode mode;
    private final int kvDim;  // K stride (and V stride when symmetric; == kDim)
    private final int vDim;   // V stride (== kvDim when symmetric, or different for MLA)
    private final int maxSeqLen;

    // FLOAT32 mode
    private final float[][] keyCache;   // [layer][position * kvDim]
    private final float[][] valueCache; // [layer][position * kvDim]

    // Q8_0 mode
    private final byte[][] keyQuants;    // [layer][position * kvDim]
    private final float[][] keyScales;   // [layer][position * (kvDim / 32)]
    private final byte[][] valueQuants;
    private final float[][] valueScales;

    // Q4_1 mode: nibbles[pos * kvDim / 2] (2 elements per byte), per-block FP32 d (scale) and m (min)
    private final byte[][] keyNibbles;
    private final float[][] keyQ4Scales;
    private final float[][] keyQ4Mins;
    private final byte[][] valueNibbles;
    private final float[][] valueQ4Scales;
    private final float[][] valueQ4Mins;

    public KVCache(int blockCount, int kvDim, int maxSeqLen) {
        this(blockCount, kvDim, maxSeqLen, Mode.FLOAT32);
    }

    public KVCache(int blockCount, int kvDim, int maxSeqLen, Mode mode) {
        this(blockCount, kvDim, kvDim, maxSeqLen, mode);
    }

    /**
     * Asymmetric-dimension constructor for MLA (DeepSeek2), where keyLength ≠ valueLength.
     * Standard GQA callers should use the symmetric overload which sets kDim == vDim.
     */
    public KVCache(int blockCount, int kDim, int vDim, int maxSeqLen, Mode mode) {
        this.mode = mode;
        this.kvDim = kDim;
        this.vDim = vDim;
        this.maxSeqLen = maxSeqLen;
        if (mode == Mode.FLOAT32) {
            this.keyCache = new float[blockCount][maxSeqLen * kDim];
            this.valueCache = new float[blockCount][maxSeqLen * vDim];
            this.keyQuants = null; this.keyScales = null;
            this.valueQuants = null; this.valueScales = null;
            this.keyNibbles = null; this.keyQ4Scales = null; this.keyQ4Mins = null;
            this.valueNibbles = null; this.valueQ4Scales = null; this.valueQ4Mins = null;
        } else if (mode == Mode.Q8_0) {
            if (kDim % Q8_BLOCK != 0 || vDim % Q8_BLOCK != 0) {
                throw new IllegalArgumentException(
                    "Q8_0 KV cache requires kDim and vDim divisible by " + Q8_BLOCK
                        + ", got kDim=" + kDim + ", vDim=" + vDim);
            }
            int kScalesPerPos = kDim / Q8_BLOCK;
            int vScalesPerPos = vDim / Q8_BLOCK;
            this.keyCache = null; this.valueCache = null;
            this.keyQuants = new byte[blockCount][maxSeqLen * kDim];
            this.keyScales = new float[blockCount][maxSeqLen * kScalesPerPos];
            this.valueQuants = new byte[blockCount][maxSeqLen * vDim];
            this.valueScales = new float[blockCount][maxSeqLen * vScalesPerPos];
            this.keyNibbles = null; this.keyQ4Scales = null; this.keyQ4Mins = null;
            this.valueNibbles = null; this.valueQ4Scales = null; this.valueQ4Mins = null;
        } else { // Q4_1
            if (kDim % Q4_BLOCK != 0 || vDim % Q4_BLOCK != 0) {
                throw new IllegalArgumentException(
                    "Q4_1 KV cache requires kDim and vDim divisible by " + Q4_BLOCK
                        + ", got kDim=" + kDim + ", vDim=" + vDim);
            }
            int kBlocksPerPos = kDim / Q4_BLOCK;
            int vBlocksPerPos = vDim / Q4_BLOCK;
            this.keyCache = null; this.valueCache = null;
            this.keyQuants = null; this.keyScales = null;
            this.valueQuants = null; this.valueScales = null;
            // Nibbles: 2 elements per byte → kDim/2 bytes per position
            this.keyNibbles = new byte[blockCount][maxSeqLen * (kDim / 2)];
            this.keyQ4Scales = new float[blockCount][maxSeqLen * kBlocksPerPos];
            this.keyQ4Mins = new float[blockCount][maxSeqLen * kBlocksPerPos];
            this.valueNibbles = new byte[blockCount][maxSeqLen * (vDim / 2)];
            this.valueQ4Scales = new float[blockCount][maxSeqLen * vBlocksPerPos];
            this.valueQ4Mins = new float[blockCount][maxSeqLen * vBlocksPerPos];
        }
    }

    public Mode getMode() { return mode; }
    public int getKvDim() { return kvDim; }
    public int getMaxSeqLen() { return maxSeqLen; }

    /** Byte offset into the cache for a given position (F32 mode: in floats; Q8 mode: in bytes). */
    public int offset(int position) { return position * kvDim; }

    /**
     * Direct float[] accessor — ONLY valid in FLOAT32 mode. Throws in Q8 mode.
     * Prefer {@link #dotK} / {@link #saxpyV} for mode-agnostic code.
     */
    public float[] keyLayer(int layer) {
        if (mode != Mode.FLOAT32) {
            throw new UnsupportedOperationException(
                "keyLayer() not available in " + mode + " mode; use dotK()");
        }
        return keyCache[layer];
    }

    public float[] valueLayer(int layer) {
        if (mode != Mode.FLOAT32) {
            throw new UnsupportedOperationException(
                "valueLayer() not available in " + mode + " mode; use saxpyV()");
        }
        return valueCache[layer];
    }

    /** Store the K projection for a single token at position {@code pos}. Quantizes if in Q8/Q4 mode. */
    public void storeK(int layer, int pos, float[] k, int len) {
        if (mode == Mode.FLOAT32) {
            System.arraycopy(k, 0, keyCache[layer], pos * kvDim, len);
        } else if (mode == Mode.Q8_0) {
            Q8.quantize(k, 0, len, keyQuants[layer], pos * kvDim, keyScales[layer], pos * (kvDim / Q8_BLOCK));
        } else { // Q4_1
            quantizeQ4_1(k, 0, len,
                keyNibbles[layer], pos * (kvDim / 2),
                keyQ4Scales[layer], pos * (kvDim / Q4_BLOCK),
                keyQ4Mins[layer], pos * (kvDim / Q4_BLOCK));
        }
    }

    /** Store the V projection for a single token at position {@code pos}. Quantizes if in Q8/Q4 mode. */
    public void storeV(int layer, int pos, float[] v, int len) {
        if (mode == Mode.FLOAT32) {
            System.arraycopy(v, 0, valueCache[layer], pos * vDim, len);
        } else if (mode == Mode.Q8_0) {
            Q8.quantize(v, 0, len, valueQuants[layer], pos * vDim, valueScales[layer], pos * (vDim / Q8_BLOCK));
        } else { // Q4_1
            quantizeQ4_1(v, 0, len,
                valueNibbles[layer], pos * (vDim / 2),
                valueQ4Scales[layer], pos * (vDim / Q4_BLOCK),
                valueQ4Mins[layer], pos * (vDim / Q4_BLOCK));
        }
    }

    /**
     * Q·K dot product for one head at one past position:
     * {@code result = query[qOff..qOff+headSize] · K[layer][pos][kvHeadOff..kvHeadOff+headSize]}.
     *
     * <p>F32 → SIMD dot. Q8/Q4 → dequant inline block-by-block.
     */
    public float dotK(int layer, int pos, int kvHeadOff, int headSize, float[] query, int qOff) {
        if (mode == Mode.FLOAT32) {
            return VectorOpsFactory.get().dot(query, qOff, keyCache[layer], pos * kvDim + kvHeadOff, headSize);
        } else if (mode == Mode.Q8_0) {
            int baseOff = pos * kvDim + kvHeadOff;
            int baseScales = baseOff / Q8_BLOCK; // safe: kvHeadOff is multiple of Q8_BLOCK because headSize∈{64,128,256}
            return Q8.dot(query, qOff, keyQuants[layer], baseOff, keyScales[layer], baseScales, headSize);
        } else { // Q4_1
            int byteOff = (pos * kvDim + kvHeadOff) / 2;
            int blockOff = (pos * kvDim + kvHeadOff) / Q4_BLOCK;
            return dotQ4_1Block(query, qOff,
                keyNibbles[layer], byteOff,
                keyQ4Scales[layer], blockOff,
                keyQ4Mins[layer], blockOff, headSize);
        }
    }

    /**
     * Weighted add:
     * {@code out[outOff..outOff+headSize] += weight * V[layer][pos][kvHeadOff..kvHeadOff+headSize]}.
     */
    public void saxpyV(int layer, int pos, int kvHeadOff, int headSize, float weight,
                        float[] out, int outOff) {
        if (mode == Mode.FLOAT32) {
            VectorOpsFactory.get().saxpy(weight, valueCache[layer], pos * vDim + kvHeadOff, out, outOff, headSize);
        } else if (mode == Mode.Q8_0) {
            int baseOff = pos * vDim + kvHeadOff;
            int baseScales = baseOff / Q8_BLOCK;
            Q8.saxpy(weight, valueQuants[layer], baseOff, valueScales[layer], baseScales, out, outOff, headSize);
        } else { // Q4_1
            int byteOff = (pos * vDim + kvHeadOff) / 2;
            int blockOff = (pos * vDim + kvHeadOff) / Q4_BLOCK;
            saxpyQ4_1Block(weight,
                valueNibbles[layer], byteOff,
                valueQ4Scales[layer], blockOff,
                valueQ4Mins[layer], blockOff,
                out, outOff, headSize);
        }
    }

    // ---------------- Q8_0 helpers ----------------

    /**
     * Quantize {@code n} floats from {@code src[srcOff..]} into Q8_0 blocks of 32. Writes
     * {@code n} bytes to {@code dst[dstOff..]} and {@code n/32} scales to {@code scales[scalesOff..]}.
     */
    private static void quantizeBlocksScalar(float[] src, int srcOff, int n,
                                        byte[] dst, int dstOff, float[] scales, int scalesOff) {
        int blocks = n / Q8_BLOCK;
        for (int b = 0; b < blocks; b++) {
            int base = srcOff + b * Q8_BLOCK;
            // 1. max |x| in block
            float maxAbs = 0f;
            for (int i = 0; i < Q8_BLOCK; i++) {
                float v = src[base + i];
                float a = v >= 0 ? v : -v;
                if (a > maxAbs) maxAbs = a;
            }
            // 2. scale = maxAbs / 127 (so values map into [-127, 127])
            float scale = maxAbs / 127.0f;
            float invScale = scale != 0f ? 1.0f / scale : 0f;
            scales[scalesOff + b] = scale;
            // 3. quantize
            int dstBase = dstOff + b * Q8_BLOCK;
            for (int i = 0; i < Q8_BLOCK; i++) {
                int q = Math.round(src[base + i] * invScale);
                if (q > 127) q = 127;
                else if (q < -128) q = -128;
                dst[dstBase + i] = (byte) q;
            }
        }
    }

    // Inline scalar Q8 dequant. Tried delegating to VectorOps.dotQ8Block (SIMD with FMA) but
    // the per-call dispatch through VectorOpsFactory.get() costs more than the SIMD savings:
    // for a 40K-call-per-token attention pass, the virtual dispatch dominates over the FMA
    // win. The HotSpot JIT inlines and partly auto-vectorizes this static method better.
    // Future work: a localized SIMD class wired in directly without VectorOpsFactory dispatch
    // could deliver real speedup, especially with ByteVector-based byte→float conversion.

    private static float dotQ8BlockScalar(float[] q, int qOff, byte[] kQuants, int kOff,
                                     float[] kScales, int kScalesOff, int n) {
        int blocks = n / Q8_BLOCK;
        float total = 0f;
        for (int b = 0; b < blocks; b++) {
            float scale = kScales[kScalesOff + b];
            int qBase = qOff + b * Q8_BLOCK;
            int kBase = kOff + b * Q8_BLOCK;
            float acc = 0f;
            for (int i = 0; i < Q8_BLOCK; i++) {
                acc += q[qBase + i] * kQuants[kBase + i];
            }
            total += acc * scale;
        }
        return total;
    }

    private static void saxpyQ8BlockScalar(float weight, byte[] vQuants, int vOff,
                                      float[] vScales, int vScalesOff,
                                      float[] out, int outOff, int n) {
        int blocks = n / Q8_BLOCK;
        for (int b = 0; b < blocks; b++) {
            float scale = vScales[vScalesOff + b];
            float w = weight * scale;
            int vBase = vOff + b * Q8_BLOCK;
            int outBase = outOff + b * Q8_BLOCK;
            for (int i = 0; i < Q8_BLOCK; i++) {
                out[outBase + i] += w * vQuants[vBase + i];
            }
        }
    }

    /**
     * Approximate memory usage in bytes, accounting for asymmetric K/V dims.
     */
    public long memoryBytes() {
        if (mode == Mode.FLOAT32) {
            long k = (long) keyCache.length * keyCache[0].length * 4;
            long v = (long) valueCache.length * valueCache[0].length * 4;
            return k + v;
        } else if (mode == Mode.Q8_0) {
            long kq = (long) keyQuants.length * keyQuants[0].length;
            long ks = (long) keyScales.length * keyScales[0].length * 4;
            long vq = (long) valueQuants.length * valueQuants[0].length;
            long vs = (long) valueScales.length * valueScales[0].length * 4;
            return kq + ks + vq + vs;
        } else { // Q4_1
            long kn = (long) keyNibbles.length * keyNibbles[0].length;
            long ks = (long) keyQ4Scales.length * keyQ4Scales[0].length * 4;
            long km = (long) keyQ4Mins.length * keyQ4Mins[0].length * 4;
            long vn = (long) valueNibbles.length * valueNibbles[0].length;
            long vs = (long) valueQ4Scales.length * valueQ4Scales[0].length * 4;
            long vm = (long) valueQ4Mins.length * valueQ4Mins[0].length * 4;
            return kn + ks + km + vn + vs + vm;
        }
    }

    // ---------------- Q4_1 helpers (scalar; SIMD plug can come later, mirrors Q8 pattern) ----------------

    /**
     * Quantize {@code n} floats from {@code src[srcOff..]} into Q4_1 blocks of 32.
     * Writes {@code n/2} bytes (2 elements per byte: low nibble = element 2i, high = element 2i+1)
     * to {@code dst[dstOff..]} and {@code n/32} blocks of (scale, min) FP32 pairs.
     *
     * <p>Standard Q4_1 reconstruction: {@code val = q * d + m} where {@code q ∈ [0, 15]}.
     */
    private static void quantizeQ4_1(float[] src, int srcOff, int n,
                                     byte[] dst, int dstOff,
                                     float[] scales, int scalesOff,
                                     float[] mins, int minsOff) {
        int blocks = n / Q4_BLOCK;
        for (int b = 0; b < blocks; b++) {
            int base = srcOff + b * Q4_BLOCK;
            float vMin = Float.POSITIVE_INFINITY;
            float vMax = Float.NEGATIVE_INFINITY;
            for (int i = 0; i < Q4_BLOCK; i++) {
                float v = src[base + i];
                if (v < vMin) vMin = v;
                if (v > vMax) vMax = v;
            }
            float d = (vMax - vMin) / 15.0f;
            float invD = (d != 0f) ? (1.0f / d) : 0f;
            scales[scalesOff + b] = d;
            mins[minsOff + b] = vMin;
            int dstBase = dstOff + b * (Q4_BLOCK / 2);
            // Interleaved pack: byte i = (q[2i] | (q[2i+1] << 4))
            for (int i = 0; i < Q4_BLOCK / 2; i++) {
                int lo = (int) ((src[base + 2 * i]     - vMin) * invD + 0.5f);
                int hi = (int) ((src[base + 2 * i + 1] - vMin) * invD + 0.5f);
                if (lo < 0) lo = 0; else if (lo > 15) lo = 15;
                if (hi < 0) hi = 0; else if (hi > 15) hi = 15;
                dst[dstBase + i] = (byte) ((lo & 0x0F) | ((hi & 0x0F) << 4));
            }
        }
    }

    private static float dotQ4_1Block(float[] q, int qOff,
                                      byte[] nibbles, int nibOff,
                                      float[] scales, int scalesOff,
                                      float[] mins, int minsOff,
                                      int n) {
        int blocks = n / Q4_BLOCK;
        float total = 0f;
        for (int b = 0; b < blocks; b++) {
            float d = scales[scalesOff + b];
            float m = mins[minsOff + b];
            int qBase = qOff + b * Q4_BLOCK;
            int nBase = nibOff + b * (Q4_BLOCK / 2);
            float accQ = 0f;  // sum(q_i * x_i) where q_i is the int nibble
            float accSum = 0f; // sum(x_i)   needed for the m * sum(x) term
            for (int i = 0; i < Q4_BLOCK / 2; i++) {
                byte packed = nibbles[nBase + i];
                int lo = packed & 0x0F;
                int hi = (packed >> 4) & 0x0F;
                float x0 = q[qBase + 2 * i];
                float x1 = q[qBase + 2 * i + 1];
                accQ += lo * x0 + hi * x1;
                accSum += x0 + x1;
            }
            total += d * accQ + m * accSum;
        }
        return total;
    }

    private static void saxpyQ4_1Block(float weight,
                                       byte[] nibbles, int nibOff,
                                       float[] scales, int scalesOff,
                                       float[] mins, int minsOff,
                                       float[] out, int outOff, int n) {
        int blocks = n / Q4_BLOCK;
        for (int b = 0; b < blocks; b++) {
            float d = scales[scalesOff + b];
            float m = mins[minsOff + b];
            float wd = weight * d;
            float wm = weight * m;
            int nBase = nibOff + b * (Q4_BLOCK / 2);
            int outBase = outOff + b * Q4_BLOCK;
            for (int i = 0; i < Q4_BLOCK / 2; i++) {
                byte packed = nibbles[nBase + i];
                int lo = packed & 0x0F;
                int hi = (packed >> 4) & 0x0F;
                out[outBase + 2 * i]     += wd * lo + wm;
                out[outBase + 2 * i + 1] += wd * hi + wm;
            }
        }
    }
}
