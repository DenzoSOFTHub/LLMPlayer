package it.denzosoft.llmplayer.tensor;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/**
 * SIMD implementation of VectorOps using Java Vector API.
 * Requires Java 21+ with jdk.incubator.vector module.
 */
public final class SimdVectorOps implements VectorOps {

    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;
    private static final int SPECIES_LENGTH = SPECIES.length();

    @Override
    public float dot(float[] a, int aOff, float[] b, int bOff, int len) {
        FloatVector sumVec = FloatVector.zero(SPECIES);
        int i = 0;
        int upperBound = SPECIES.loopBound(len);
        for (; i < upperBound; i += SPECIES_LENGTH) {
            FloatVector va = FloatVector.fromArray(SPECIES, a, aOff + i);
            FloatVector vb = FloatVector.fromArray(SPECIES, b, bOff + i);
            sumVec = va.fma(vb, sumVec);
        }
        float sum = sumVec.reduceLanes(VectorOperators.ADD);
        for (; i < len; i++) {
            sum += a[aOff + i] * b[bOff + i];
        }
        return sum;
    }

    @Override
    public void saxpy(float a, float[] x, int xOff, float[] y, int yOff, int len) {
        FloatVector va = FloatVector.broadcast(SPECIES, a);
        int i = 0;
        int upperBound = SPECIES.loopBound(len);
        for (; i < upperBound; i += SPECIES_LENGTH) {
            FloatVector vx = FloatVector.fromArray(SPECIES, x, xOff + i);
            FloatVector vy = FloatVector.fromArray(SPECIES, y, yOff + i);
            vy = vx.fma(va, vy);
            vy.intoArray(y, yOff + i);
        }
        for (; i < len; i++) {
            y[yOff + i] += a * x[xOff + i];
        }
    }

    @Override
    public void rmsnorm(float[] out, float[] x, float[] w, int size, float eps) {
        FloatVector sumVec = FloatVector.zero(SPECIES);
        int i = 0;
        int upperBound = SPECIES.loopBound(size);
        for (; i < upperBound; i += SPECIES_LENGTH) {
            FloatVector vx = FloatVector.fromArray(SPECIES, x, i);
            sumVec = vx.fma(vx, sumVec);
        }
        float ss = sumVec.reduceLanes(VectorOperators.ADD);
        for (; i < size; i++) {
            ss += x[i] * x[i];
        }
        ss = 1.0f / (float) Math.sqrt(ss / size + eps);

        FloatVector scale = FloatVector.broadcast(SPECIES, ss);
        i = 0;
        for (; i < upperBound; i += SPECIES_LENGTH) {
            FloatVector vx = FloatVector.fromArray(SPECIES, x, i);
            FloatVector vw = FloatVector.fromArray(SPECIES, w, i);
            vx.mul(scale).mul(vw).intoArray(out, i);
        }
        for (; i < size; i++) {
            out[i] = x[i] * ss * w[i];
        }
    }

    @Override
    public void softmax(float[] logits, int offset, int size) {
        float maxVal = Float.NEGATIVE_INFINITY;
        for (int i = 0; i < size; i++) {
            if (logits[offset + i] > maxVal) maxVal = logits[offset + i];
        }
        float sum = 0f;
        for (int i = 0; i < size; i++) {
            logits[offset + i] = (float) Math.exp(logits[offset + i] - maxVal);
            sum += logits[offset + i];
        }
        float invSum = 1.0f / sum;
        FloatVector vInvSum = FloatVector.broadcast(SPECIES, invSum);
        int i = 0;
        int upperBound = SPECIES.loopBound(size);
        for (; i < upperBound; i += SPECIES_LENGTH) {
            FloatVector v = FloatVector.fromArray(SPECIES, logits, offset + i);
            v.mul(vInvSum).intoArray(logits, offset + i);
        }
        for (; i < size; i++) {
            logits[offset + i] *= invSum;
        }
    }

    @Override
    public void scale(float[] x, int offset, int size, float scale) {
        FloatVector vs = FloatVector.broadcast(SPECIES, scale);
        int i = 0;
        int upperBound = SPECIES.loopBound(size);
        for (; i < upperBound; i += SPECIES_LENGTH) {
            FloatVector vx = FloatVector.fromArray(SPECIES, x, offset + i);
            vx.mul(vs).intoArray(x, offset + i);
        }
        for (; i < size; i++) {
            x[offset + i] *= scale;
        }
    }

    @Override
    public void elementwiseMul(float[] a, float[] b, float[] out, int size) {
        int i = 0;
        int upperBound = SPECIES.loopBound(size);
        for (; i < upperBound; i += SPECIES_LENGTH) {
            FloatVector va = FloatVector.fromArray(SPECIES, a, i);
            FloatVector vb = FloatVector.fromArray(SPECIES, b, i);
            va.mul(vb).intoArray(out, i);
        }
        for (; i < size; i++) {
            out[i] = a[i] * b[i];
        }
    }

    @Override
    public void silu(float[] x, int size) {
        for (int i = 0; i < size; i++) {
            x[i] = x[i] / (1.0f + (float) Math.exp(-x[i]));
        }
    }

    @Override
    public void accumulate(float[] y, float[] x, int size) {
        int i = 0;
        int upperBound = SPECIES.loopBound(size);
        for (; i < upperBound; i += SPECIES_LENGTH) {
            FloatVector vy = FloatVector.fromArray(SPECIES, y, i);
            FloatVector vx = FloatVector.fromArray(SPECIES, x, i);
            vy.add(vx).intoArray(y, i);
        }
        for (; i < size; i++) {
            y[i] += x[i];
        }
    }

    @Override
    public void scaleWeighted(float[] x, int xOff, float[] w, float scale, int size) {
        FloatVector vScale = FloatVector.broadcast(SPECIES, scale);
        int i = 0;
        int upperBound = SPECIES.loopBound(size);
        for (; i < upperBound; i += SPECIES_LENGTH) {
            FloatVector vx = FloatVector.fromArray(SPECIES, x, xOff + i);
            FloatVector vw = FloatVector.fromArray(SPECIES, w, i);
            vx.mul(vScale).mul(vw).intoArray(x, xOff + i);
        }
        for (; i < size; i++) {
            x[xOff + i] = x[xOff + i] * scale * w[i];
        }
    }

    @Override
    public void ropeNeox(float[] vec, int off, float[] cos, float[] sin, int cosOff, int halfRope) {
        int i = 0;
        int upperBound = SPECIES.loopBound(halfRope);
        for (; i < upperBound; i += SPECIES_LENGTH) {
            FloatVector vc = FloatVector.fromArray(SPECIES, cos, cosOff + i);
            FloatVector vs = FloatVector.fromArray(SPECIES, sin, cosOff + i);
            FloatVector v0 = FloatVector.fromArray(SPECIES, vec, off + i);
            FloatVector v1 = FloatVector.fromArray(SPECIES, vec, off + halfRope + i);
            // vec[off+i]          = v0 * cos - v1 * sin
            // vec[off+halfRope+i] = v0 * sin + v1 * cos
            v0.mul(vc).sub(v1.mul(vs)).intoArray(vec, off + i);
            v0.mul(vs).add(v1.mul(vc)).intoArray(vec, off + halfRope + i);
        }
        for (; i < halfRope; i++) {
            float c = cos[cosOff + i];
            float s = sin[cosOff + i];
            float val0 = vec[off + i];
            float val1 = vec[off + halfRope + i];
            vec[off + i] = val0 * c - val1 * s;
            vec[off + halfRope + i] = val0 * s + val1 * c;
        }
    }
}
