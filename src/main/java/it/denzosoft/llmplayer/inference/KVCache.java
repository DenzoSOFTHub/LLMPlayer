package it.denzosoft.llmplayer.inference;

import it.denzosoft.llmplayer.tensor.VectorOps;
import it.denzosoft.llmplayer.tensor.VectorOpsFactory;

/**
 * Key-Value cache for autoregressive generation. Pre-allocated per layer, stores projected K and V
 * vectors for all past positions.
 *
 * <p>Supports two storage modes:
 * <ul>
 *   <li>{@link Mode#FLOAT32} — plain {@code float[]} (4 bytes/elem, default).
 *   <li>{@link Mode#Q8_0} — block-quantized int8 with FP32 scales (1.125 bytes/elem, ~3.56× smaller).
 *       Uses the same Q8_0 block layout as llama.cpp: 32 elements per block, one FP32 scale per
 *       block (2 bytes in llama.cpp's FP16 scale; we use FP32 for Java ergonomics — still saves
 *       3.56×). Enabled via {@code -Dkv.q8=true} JVM flag (see {@link InferenceState}).
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

    public enum Mode { FLOAT32, Q8_0 }

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
            this.keyQuants = null;
            this.keyScales = null;
            this.valueQuants = null;
            this.valueScales = null;
        } else {
            // Q8_0 requires both dims divisible by block size.
            if (kDim % Q8_BLOCK != 0 || vDim % Q8_BLOCK != 0) {
                throw new IllegalArgumentException(
                    "Q8_0 KV cache requires kDim and vDim divisible by " + Q8_BLOCK
                        + ", got kDim=" + kDim + ", vDim=" + vDim);
            }
            int kScalesPerPos = kDim / Q8_BLOCK;
            int vScalesPerPos = vDim / Q8_BLOCK;
            this.keyCache = null;
            this.valueCache = null;
            this.keyQuants = new byte[blockCount][maxSeqLen * kDim];
            this.keyScales = new float[blockCount][maxSeqLen * kScalesPerPos];
            this.valueQuants = new byte[blockCount][maxSeqLen * vDim];
            this.valueScales = new float[blockCount][maxSeqLen * vScalesPerPos];
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
            throw new UnsupportedOperationException("keyLayer() not available in Q8 mode; use dotK()");
        }
        return keyCache[layer];
    }

    public float[] valueLayer(int layer) {
        if (mode != Mode.FLOAT32) {
            throw new UnsupportedOperationException("valueLayer() not available in Q8 mode; use saxpyV()");
        }
        return valueCache[layer];
    }

    /** Store the K projection for a single token at position {@code pos}. Quantizes if in Q8 mode. */
    public void storeK(int layer, int pos, float[] k, int len) {
        if (mode == Mode.FLOAT32) {
            System.arraycopy(k, 0, keyCache[layer], pos * kvDim, len);
        } else {
            quantizeBlocks(k, 0, len, keyQuants[layer], pos * kvDim, keyScales[layer], pos * (kvDim / Q8_BLOCK));
        }
    }

    /** Store the V projection for a single token at position {@code pos}. Quantizes if in Q8 mode. */
    public void storeV(int layer, int pos, float[] v, int len) {
        if (mode == Mode.FLOAT32) {
            System.arraycopy(v, 0, valueCache[layer], pos * vDim, len);
        } else {
            quantizeBlocks(v, 0, len, valueQuants[layer], pos * vDim, valueScales[layer], pos * (vDim / Q8_BLOCK));
        }
    }

    /**
     * Q·K dot product for one head at one past position:
     * {@code result = query[qOff..qOff+headSize] · K[layer][pos][kvHeadOff..kvHeadOff+headSize]}.
     *
     * <p>In F32 mode delegates to SIMD {@code dot}. In Q8 mode dequantizes inline block-by-block.
     */
    public float dotK(int layer, int pos, int kvHeadOff, int headSize, float[] query, int qOff) {
        if (mode == Mode.FLOAT32) {
            return VectorOpsFactory.get().dot(query, qOff, keyCache[layer], pos * kvDim + kvHeadOff, headSize);
        } else {
            int baseOff = pos * kvDim + kvHeadOff;
            int baseScales = baseOff / Q8_BLOCK; // safe: kvHeadOff is multiple of Q8_BLOCK because headSize∈{64,128,256}
            return dotQ8Block(query, qOff, keyQuants[layer], baseOff, keyScales[layer], baseScales, headSize);
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
        } else {
            int baseOff = pos * vDim + kvHeadOff;
            int baseScales = baseOff / Q8_BLOCK;
            saxpyQ8Block(weight, valueQuants[layer], baseOff, valueScales[layer], baseScales, out, outOff, headSize);
        }
    }

    // ---------------- Q8_0 helpers ----------------

    /**
     * Quantize {@code n} floats from {@code src[srcOff..]} into Q8_0 blocks of 32. Writes
     * {@code n} bytes to {@code dst[dstOff..]} and {@code n/32} scales to {@code scales[scalesOff..]}.
     */
    private static void quantizeBlocks(float[] src, int srcOff, int n,
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

    private static float dotQ8Block(float[] q, int qOff, byte[] kQuants, int kOff,
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

    private static void saxpyQ8Block(float weight, byte[] vQuants, int vOff,
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
        } else {
            long kq = (long) keyQuants.length * keyQuants[0].length;
            long ks = (long) keyScales.length * keyScales[0].length * 4;
            long vq = (long) valueQuants.length * valueQuants[0].length;
            long vs = (long) valueScales.length * valueScales[0].length * 4;
            return kq + ks + vq + vs;
        }
    }
}
