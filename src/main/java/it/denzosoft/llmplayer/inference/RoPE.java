package it.denzosoft.llmplayer.inference;

/**
 * Rotary Position Embedding.
 * Pre-computes cos/sin tables and applies rotation to Q and K vectors.
 *
 * Supports two modes:
 * - NORMAL (Llama, DeepSeek2): consecutive pairs (vec[2i], vec[2i+1])
 * - NEOX (Qwen, Falcon, GLM4): split-half pairs (vec[i], vec[halfRope+i])
 *
 * Supports partial RoPE: only the first ropeDimCount dimensions are rotated,
 * the rest pass through unchanged (used by GLM4 with ropeDimCount=64, headSize=128).
 *
 * Supports YaRN (NTK-by-parts) scaling for extended context models.
 */
public class RoPE {

    public static final int ROPE_TYPE_NORMAL = 0;
    public static final int ROPE_TYPE_NEOX = 2;

    private final float[] cosTable;
    private final float[] sinTable;
    private final int headSize;
    private final int ropeDimCount; // how many dims to rotate per head (may be < headSize)
    private final int ropeType;
    private final float mscale;    // YaRN attention magnitude correction (1.0 if no YaRN)

    public RoPE(int headSize, int ropeDimCount, int maxSeqLen, float theta, int ropeType, float[] freqFactors) {
        this(headSize, ropeDimCount, maxSeqLen, theta, ropeType, freqFactors, null);
    }

    /**
     * Full constructor with optional YaRN parameters.
     * @param yarnParams YaRN scaling parameters, or null for standard RoPE.
     */
    public RoPE(int headSize, int ropeDimCount, int maxSeqLen, float theta, int ropeType,
                float[] freqFactors, YarnParams yarnParams) {
        this.headSize = headSize;
        this.ropeDimCount = ropeDimCount;
        this.ropeType = ropeType;
        int halfRope = ropeDimCount / 2;
        this.cosTable = new float[maxSeqLen * halfRope];
        this.sinTable = new float[maxSeqLen * halfRope];

        // Compute YaRN correction dimensions if applicable
        float corrLow = 0, corrHigh = 0;
        float freqScale = 1.0f;
        float computedMscale = 1.0f;
        boolean useYarn = yarnParams != null && yarnParams.scalingFactor > 1.0f;

        if (useYarn) {
            freqScale = 1.0f / yarnParams.scalingFactor;
            // Compute YaRN correction dimension boundaries
            corrLow = yarnCorrDim(ropeDimCount, yarnParams.origContextLength, yarnParams.betaFast, theta);
            corrHigh = yarnCorrDim(ropeDimCount, yarnParams.origContextLength, yarnParams.betaSlow, theta);
            corrLow = Math.max(0, (float) Math.floor(corrLow));
            corrHigh = Math.min(ropeDimCount / 2.0f - 1, (float) Math.ceil(corrHigh));

            // mscale: attention magnitude correction
            // From llama.cpp: attn_factor_org = attn_factor * (1 + 0.1 * log(1/freq_scale))
            //                  mscale = attn_factor_org * (1 + 0.1 * yarn_log_mul * log(1/freq_scale))
            // mscale is NOT baked into cos/sin tables - it's applied as attention scale: mscale^2/sqrt(d)
            float attnFactor = 1.0f;
            if (yarnParams.yarnLogMul > 0) {
                attnFactor = 0.1f * yarnParams.yarnLogMul * (float) Math.log(yarnParams.scalingFactor) + 1.0f;
            }
            computedMscale = attnFactor * (1.0f + 0.1f * (float) Math.log(1.0f / freqScale));
        }
        this.mscale = computedMscale;

        for (int pos = 0; pos < maxSeqLen; pos++) {
            for (int i = 0; i < halfRope; i++) {
                // Base frequency for dimension pair i
                float baseFreq = (float) (1.0 / Math.pow(theta, 2.0 * i / ropeDimCount));

                if (freqFactors != null && i < freqFactors.length) {
                    baseFreq /= freqFactors[i];
                }

                float freq;
                if (useYarn) {
                    // YaRN NTK-by-parts blending
                    float thetaExtrap = baseFreq;                  // original frequency
                    float thetaInterp = freqScale * baseFreq;     // interpolated frequency
                    // ramp: 1.0 for high-freq (keep original), 0.0 for low-freq (interpolate)
                    float rampMix = yarnRamp(corrLow, corrHigh, i);
                    freq = thetaInterp * (1.0f - rampMix) + thetaExtrap * rampMix;
                } else {
                    freq = baseFreq;
                }

                float angle = pos * freq;
                cosTable[pos * halfRope + i] = (float) Math.cos(angle);
                sinTable[pos * halfRope + i] = (float) Math.sin(angle);
            }
        }
    }

    /** YaRN correction dimension: maps a beta value to the boundary dimension index. */
    private static float yarnCorrDim(int nDims, int nCtxOrig, float beta, float freqBase) {
        return nDims * (float) Math.log(nCtxOrig / (beta * 2.0 * Math.PI)) / (2.0f * (float) Math.log(freqBase));
    }

    /** YaRN ramp function: returns 1.0 for high-freq dims, 0.0 for low-freq, interpolated in between. */
    private static float yarnRamp(float low, float high, int i) {
        float y = (i - low) / Math.max(0.001f, high - low);
        return 1.0f - Math.min(1.0f, Math.max(0.0f, y));
    }

    /** YaRN scaling parameters. */
    public static class YarnParams {
        public final float scalingFactor;       // e.g., 40.0
        public final int origContextLength;     // e.g., 4096
        public final float betaFast;            // default 32.0
        public final float betaSlow;            // default 1.0
        public final float yarnLogMul;          // e.g., 0.0707

        public YarnParams(float scalingFactor, int origContextLength, float yarnLogMul) {
            this(scalingFactor, origContextLength, 32.0f, 1.0f, yarnLogMul);
        }

        public YarnParams(float scalingFactor, int origContextLength,
                          float betaFast, float betaSlow, float yarnLogMul) {
            this.scalingFactor = scalingFactor;
            this.origContextLength = origContextLength;
            this.betaFast = betaFast;
            this.betaSlow = betaSlow;
            this.yarnLogMul = yarnLogMul;
        }
    }

    /**
     * Apply RoPE to a single head's Q or K vector in-place.
     * vec[offset..offset+headSize-1] is modified (only first ropeDimCount dims rotated).
     */
    public void apply(float[] vec, int offset, int position) {
        int halfRope = ropeDimCount / 2;
        int tableOffset = position * halfRope;
        if (ropeType == ROPE_TYPE_NORMAL) {
            // Consecutive pairs: (vec[2i], vec[2i+1])
            for (int i = 0; i < halfRope; i++) {
                float cos = cosTable[tableOffset + i];
                float sin = sinTable[tableOffset + i];
                float v0 = vec[offset + 2 * i];
                float v1 = vec[offset + 2 * i + 1];
                vec[offset + 2 * i] = v0 * cos - v1 * sin;
                vec[offset + 2 * i + 1] = v0 * sin + v1 * cos;
            }
        } else {
            // Split-half pairs: (vec[i], vec[halfRope+i])
            for (int i = 0; i < halfRope; i++) {
                float cos = cosTable[tableOffset + i];
                float sin = sinTable[tableOffset + i];
                float v0 = vec[offset + i];
                float v1 = vec[offset + halfRope + i];
                vec[offset + i] = v0 * cos - v1 * sin;
                vec[offset + halfRope + i] = v0 * sin + v1 * cos;
            }
        }
        // Dimensions from ropeDimCount..headSize-1 are left unchanged (partial RoPE)
    }

    /**
     * Apply RoPE to all heads in a Q or K vector.
     */
    public void applyAllHeads(float[] vec, int nHeads, int position) {
        for (int h = 0; h < nHeads; h++) {
            apply(vec, h * headSize, position);
        }
    }

    /**
     * Get the YaRN mscale value for attention score scaling.
     * For standard RoPE (no YaRN), returns 1.0.
     * Caller should use mscale^2 / sqrt(headDim) as the attention scale factor.
     */
    public float getMscale() {
        return mscale;
    }
}
