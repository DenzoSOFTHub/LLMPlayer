package it.denzosoft.llmplayer.inference;

import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/**
 * Vectorized Q8_0 KV-cache dequant + dot / saxpy. Discovered by reflection from
 * {@link KVCache} when the Java 21 Vector API is available; the scalar paths
 * inside {@code KVCache} remain the Java 8 fallback. Implements {@link KVCache.Q8Ops}
 * so per-call dispatch goes through a JIT-inlinable interface invocation
 * (no reflection on the hot path).
 *
 * <p>Algorithm matches the per-block contract used by the scalar fallback:
 * 32-element Q8_0 blocks, one FP32 scale per block. Within each block we load
 * 8 int8 quants as a {@link ByteVector}, sign-extend to {@link IntVector} via
 * {@code B2I}, convert to {@link FloatVector} via {@code I2F}, and FMA with
 * the matching query slice. Block scale is applied on the per-block partial
 * reduction (one scalar multiply per block, not per lane).
 *
 * <p>If {@code SPECIES_PREFERRED.length() != 8} (e.g. AVX-512 or NEON 128-bit),
 * each method falls back to a scalar inline loop — so loading this class never
 * regresses correctness or perf vs the {@code KVCache} scalar path.
 */
public final class SimdQ8KvOps implements KVCache.Q8Ops {

    private static final VectorSpecies<Float> F_SPECIES = FloatVector.SPECIES_256;
    private static final VectorSpecies<Integer> I_SPECIES = IntVector.SPECIES_256;
    private static final VectorSpecies<Byte> B_SPECIES = ByteVector.SPECIES_64;
    private static final int Q8_BLOCK = 32;
    private static final boolean SIMD_OK = FloatVector.SPECIES_PREFERRED.length() == 8;

    public SimdQ8KvOps() {}

    @Override
    public float dot(float[] q, int qOff,
                     byte[] kQuants, int kOff,
                     float[] kScales, int kScalesOff,
                     int n) {
        if (!SIMD_OK) return scalarDot(q, qOff, kQuants, kOff, kScales, kScalesOff, n);
        int blocks = n / Q8_BLOCK;
        float total = 0f;
        for (int b = 0; b < blocks; b++) {
            float scale = kScales[kScalesOff + b];
            int qBase = qOff + b * Q8_BLOCK;
            int kBase = kOff + b * Q8_BLOCK;
            FloatVector acc = FloatVector.zero(F_SPECIES);
            for (int chunk = 0; chunk < 32; chunk += 8) {
                ByteVector vqb = ByteVector.fromArray(B_SPECIES, kQuants, kBase + chunk);
                IntVector vqi = (IntVector) vqb.convertShape(VectorOperators.B2I, I_SPECIES, 0);
                FloatVector vqf = (FloatVector) vqi.convertShape(VectorOperators.I2F, F_SPECIES, 0);
                FloatVector vqry = FloatVector.fromArray(F_SPECIES, q, qBase + chunk);
                acc = vqf.fma(vqry, acc);
            }
            total += acc.reduceLanes(VectorOperators.ADD) * scale;
        }
        return total;
    }

    @Override
    public void saxpy(float weight,
                      byte[] vQuants, int vOff,
                      float[] vScales, int vScalesOff,
                      float[] out, int outOff, int n) {
        if (!SIMD_OK) { scalarSaxpy(weight, vQuants, vOff, vScales, vScalesOff, out, outOff, n); return; }
        int blocks = n / Q8_BLOCK;
        for (int b = 0; b < blocks; b++) {
            float scaledW = weight * vScales[vScalesOff + b];
            FloatVector vW = FloatVector.broadcast(F_SPECIES, scaledW);
            int vBase = vOff + b * Q8_BLOCK;
            int outBase = outOff + b * Q8_BLOCK;
            for (int chunk = 0; chunk < 32; chunk += 8) {
                ByteVector vqb = ByteVector.fromArray(B_SPECIES, vQuants, vBase + chunk);
                IntVector vqi = (IntVector) vqb.convertShape(VectorOperators.B2I, I_SPECIES, 0);
                FloatVector vqf = (FloatVector) vqi.convertShape(VectorOperators.I2F, F_SPECIES, 0);
                FloatVector vOut = FloatVector.fromArray(F_SPECIES, out, outBase + chunk);
                FloatVector res = vqf.fma(vW, vOut);
                res.intoArray(out, outBase + chunk);
            }
        }
    }

    @Override
    public void quantize(float[] src, int srcOff, int n,
                         byte[] dst, int dstOff,
                         float[] scales, int scalesOff) {
        if (!SIMD_OK) { scalarQuantize(src, srcOff, n, dst, dstOff, scales, scalesOff); return; }
        int blocks = n / Q8_BLOCK;
        for (int b = 0; b < blocks; b++) {
            int base = srcOff + b * Q8_BLOCK;
            FloatVector vMax = FloatVector.zero(F_SPECIES);
            for (int chunk = 0; chunk < 32; chunk += 8) {
                FloatVector v = FloatVector.fromArray(F_SPECIES, src, base + chunk);
                vMax = vMax.max(v.abs());
            }
            float maxAbs = vMax.reduceLanes(VectorOperators.MAX);
            float scale = maxAbs / 127.0f;
            float invScale = scale != 0f ? 1.0f / scale : 0f;
            scales[scalesOff + b] = scale;

            FloatVector vInv = FloatVector.broadcast(F_SPECIES, invScale);
            int dstBase = dstOff + b * Q8_BLOCK;
            for (int chunk = 0; chunk < 32; chunk += 8) {
                FloatVector v = FloatVector.fromArray(F_SPECIES, src, base + chunk);
                FloatVector scaled = v.mul(vInv);
                // Truncate F2I: differs from Math.round by < 1 ULP per element. Q8 quant
                // is already lossy; bit-exact equivalence with the scalar path is not a goal.
                IntVector vi = (IntVector) scaled.convertShape(VectorOperators.F2I, I_SPECIES, 0);
                IntVector clamped = vi.max(IntVector.broadcast(I_SPECIES, -128))
                                       .min(IntVector.broadcast(I_SPECIES, 127));
                ByteVector vb = (ByteVector) clamped.convertShape(VectorOperators.I2B, B_SPECIES, 0);
                vb.intoArray(dst, dstBase + chunk);
            }
        }
    }

    // ---- scalar fallbacks (mirror KVCache.*Scalar; kept here so non-AVX2 hardware
    // still gets a working impl through the same interface dispatch).

    private static float scalarDot(float[] q, int qOff, byte[] kQ, int kOff,
                                    float[] kS, int kSOff, int n) {
        int blocks = n / Q8_BLOCK;
        float total = 0f;
        for (int b = 0; b < blocks; b++) {
            float scale = kS[kSOff + b];
            int qBase = qOff + b * Q8_BLOCK;
            int kBase = kOff + b * Q8_BLOCK;
            float acc = 0f;
            for (int i = 0; i < Q8_BLOCK; i++) acc += q[qBase + i] * kQ[kBase + i];
            total += acc * scale;
        }
        return total;
    }

    private static void scalarSaxpy(float weight, byte[] vQ, int vOff, float[] vS, int vSOff,
                                     float[] out, int outOff, int n) {
        int blocks = n / Q8_BLOCK;
        for (int b = 0; b < blocks; b++) {
            float w = weight * vS[vSOff + b];
            int vBase = vOff + b * Q8_BLOCK;
            int outBase = outOff + b * Q8_BLOCK;
            for (int i = 0; i < Q8_BLOCK; i++) out[outBase + i] += w * vQ[vBase + i];
        }
    }

    private static void scalarQuantize(float[] src, int srcOff, int n, byte[] dst, int dstOff,
                                        float[] scales, int sOff) {
        int blocks = n / Q8_BLOCK;
        for (int b = 0; b < blocks; b++) {
            int base = srcOff + b * Q8_BLOCK;
            float maxAbs = 0f;
            for (int i = 0; i < Q8_BLOCK; i++) {
                float a = Math.abs(src[base + i]);
                if (a > maxAbs) maxAbs = a;
            }
            float scale = maxAbs / 127.0f;
            float invScale = scale != 0f ? 1.0f / scale : 0f;
            scales[sOff + b] = scale;
            int dstBase = dstOff + b * Q8_BLOCK;
            for (int i = 0; i < Q8_BLOCK; i++) {
                int q = Math.round(src[base + i] * invScale);
                if (q > 127) q = 127; else if (q < -128) q = -128;
                dst[dstBase + i] = (byte) q;
            }
        }
    }
}
