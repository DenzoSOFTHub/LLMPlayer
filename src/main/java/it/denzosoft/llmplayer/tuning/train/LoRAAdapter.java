package it.denzosoft.llmplayer.tuning.train;

import java.io.*;
import java.nio.file.*;
import java.util.*;

/**
 * LoRA adapter for a single linear layer.
 * W_effective = W_base + (B * A) * (alpha / rank)
 *
 * A: rank x inputDim  (initialized with Kaiming)
 * B: outputDim x rank (initialized to zeros)
 *
 * At init, B*A = 0, so the model starts unchanged.
 */
public class LoRAAdapter {

    private final String name;         // e.g., "blk.0.attn_q"
    private final int inputDim;
    private final int outputDim;
    private final int rank;
    private final float alpha;
    private final float scale;         // alpha / rank

    private final TrainableTensor loraA;   // rank x inputDim
    private final TrainableTensor loraB;   // outputDim x rank

    // Gradients (allocated lazily on first backward)
    private TrainableTensor gradA;
    private TrainableTensor gradB;

    // AdamW optimizer states
    private TrainableTensor mA, vA;    // first/second moment for A
    private TrainableTensor mB, vB;    // first/second moment for B

    // Cached input for backward pass
    private float[] lastInput;

    public LoRAAdapter(String name, int inputDim, int outputDim, int rank, float alpha, Random rng) {
        this.name = name;
        this.inputDim = inputDim;
        this.outputDim = outputDim;
        this.rank = rank;
        this.alpha = alpha;
        this.scale = alpha / rank;

        this.loraA = new TrainableTensor(rank, inputDim);
        this.loraB = new TrainableTensor(outputDim, rank);

        // Initialize: A with Kaiming, B with zeros → initial output is zero
        loraA.initKaiming(rng);
        loraB.zero();
    }

    /** LoRA adapter from pre-existing weights (for loading checkpoints). */
    public LoRAAdapter(String name, int inputDim, int outputDim, int rank, float alpha,
                       TrainableTensor loraA, TrainableTensor loraB) {
        this.name = name;
        this.inputDim = inputDim;
        this.outputDim = outputDim;
        this.rank = rank;
        this.alpha = alpha;
        this.scale = alpha / rank;
        this.loraA = loraA;
        this.loraB = loraB;
    }

    /**
     * Forward pass: compute LoRA delta for input vector x.
     * Returns: (B * A * x) * scale — to be added to the base layer output.
     * Caches input x for backward.
     */
    public float[] forward(float[] x) {
        this.lastInput = x;
        // h = A * x (rank-dimensional)
        float[] h = loraA.matvec(x);
        // output = B * h (outputDim-dimensional)
        float[] output = loraB.matvec(h);
        // Scale by alpha/rank
        for (int i = 0; i < output.length; i++) {
            output[i] *= scale;
        }
        return output;
    }

    /**
     * Backward pass: given dL/d(output) gradient, compute and accumulate
     * gradients for A and B, and return dL/dx for upstream propagation.
     *
     * @param gradOutput gradient w.r.t. the LoRA output (length = outputDim)
     * @return gradient w.r.t. input x (length = inputDim)
     */
    public float[] backward(float[] gradOutput) {
        ensureGradients();

        // Scale gradient by LoRA scale
        float[] scaledGrad = new float[gradOutput.length];
        for (int i = 0; i < gradOutput.length; i++) {
            scaledGrad[i] = gradOutput[i] * scale;
        }

        // Recompute intermediate: h = A * lastInput
        float[] h = loraA.matvec(lastInput);

        // gradB += scaledGrad (outer) h^T  →  gradB[i][j] += scaledGrad[i] * h[j]
        for (int i = 0; i < outputDim; i++) {
            for (int j = 0; j < rank; j++) {
                gradB.set(i, j, gradB.get(i, j) + scaledGrad[i] * h[j]);
            }
        }

        // gradH = B^T * scaledGrad (rank-dimensional)
        float[] gradH = new float[rank];
        for (int j = 0; j < rank; j++) {
            float sum = 0;
            for (int i = 0; i < outputDim; i++) {
                sum += loraB.get(i, j) * scaledGrad[i];
            }
            gradH[j] = sum;
        }

        // gradA += gradH (outer) lastInput^T  →  gradA[r][c] += gradH[r] * lastInput[c]
        for (int r = 0; r < rank; r++) {
            for (int c = 0; c < inputDim; c++) {
                gradA.set(r, c, gradA.get(r, c) + gradH[r] * lastInput[c]);
            }
        }

        // Gradient w.r.t. input: dL/dx = A^T * gradH (inputDim-dimensional)
        float[] gradInput = new float[inputDim];
        for (int c = 0; c < inputDim; c++) {
            float sum = 0;
            for (int r = 0; r < rank; r++) {
                sum += loraA.get(r, c) * gradH[r];
            }
            gradInput[c] = sum;
        }

        return gradInput;
    }

    /** Zero all accumulated gradients. Called at the start of each optimizer step. */
    public void zeroGrad() {
        if (gradA != null) gradA.zero();
        if (gradB != null) gradB.zero();
    }

    /** AdamW update step. */
    public void adamWStep(float lr, float beta1, float beta2, float eps,
                          float weightDecay, int step) {
        ensureOptimizerStates();
        adamWUpdate(loraA, gradA, mA, vA, lr, beta1, beta2, eps, weightDecay, step);
        adamWUpdate(loraB, gradB, mB, vB, lr, beta1, beta2, eps, weightDecay, step);
    }

    private void adamWUpdate(TrainableTensor param, TrainableTensor grad,
                             TrainableTensor m, TrainableTensor v,
                             float lr, float beta1, float beta2, float eps,
                             float weightDecay, int step) {
        float[] p = param.data();
        float[] g = grad.data();
        float[] md = m.data();
        float[] vd = v.data();
        float bc1 = 1f - (float) Math.pow(beta1, step);
        float bc2 = 1f - (float) Math.pow(beta2, step);

        for (int i = 0; i < p.length; i++) {
            // Weight decay
            p[i] *= (1f - lr * weightDecay);
            // Moment updates
            md[i] = beta1 * md[i] + (1f - beta1) * g[i];
            vd[i] = beta2 * vd[i] + (1f - beta2) * g[i] * g[i];
            // Bias-corrected estimates
            float mHat = md[i] / bc1;
            float vHat = vd[i] / bc2;
            // Parameter update
            p[i] -= lr * mHat / ((float) Math.sqrt(vHat) + eps);
        }
    }

    /** Total trainable parameter count. */
    public int paramCount() { return loraA.size() + loraB.size(); }

    /** Compute the merged weight delta: B * A * scale. Returns (outputDim x inputDim). */
    public TrainableTensor computeDelta() {
        TrainableTensor delta = loraB.matmul(loraA);
        delta.scale(scale);
        return delta;
    }

    private void ensureGradients() {
        if (gradA == null) {
            gradA = new TrainableTensor(rank, inputDim);
            gradB = new TrainableTensor(outputDim, rank);
        }
    }

    private void ensureOptimizerStates() {
        if (mA == null) {
            mA = new TrainableTensor(rank, inputDim);
            vA = new TrainableTensor(rank, inputDim);
            mB = new TrainableTensor(outputDim, rank);
            vB = new TrainableTensor(outputDim, rank);
        }
    }

    // --- Getters ---

    public String name() { return name; }
    public int inputDim() { return inputDim; }
    public int outputDim() { return outputDim; }
    public int rank() { return rank; }
    public TrainableTensor loraA() { return loraA; }
    public TrainableTensor loraB() { return loraB; }

    // --- Serialization ---

    public void save(Path dir) throws IOException {
        if (!Files.isDirectory(dir)) Files.createDirectories(dir);
        loraA.save(dir.resolve(name + ".A.bin"));
        loraB.save(dir.resolve(name + ".B.bin"));
        if (mA != null) {
            mA.save(dir.resolve(name + ".mA.bin"));
            vA.save(dir.resolve(name + ".vA.bin"));
            mB.save(dir.resolve(name + ".mB.bin"));
            vB.save(dir.resolve(name + ".vB.bin"));
        }
    }

    public void loadState(Path dir) throws IOException {
        Path mAFile = dir.resolve(name + ".mA.bin");
        if (Files.exists(mAFile)) {
            mA = TrainableTensor.load(mAFile);
            vA = TrainableTensor.load(dir.resolve(name + ".vA.bin"));
            mB = TrainableTensor.load(dir.resolve(name + ".mB.bin"));
            vB = TrainableTensor.load(dir.resolve(name + ".vB.bin"));
        }
    }

    public static LoRAAdapter load(Path dir, String name, int inputDim, int outputDim,
                                   int rank, float alpha) throws IOException {
        TrainableTensor a = TrainableTensor.load(dir.resolve(name + ".A.bin"));
        TrainableTensor b = TrainableTensor.load(dir.resolve(name + ".B.bin"));
        LoRAAdapter adapter = new LoRAAdapter(name, inputDim, outputDim, rank, alpha, a, b);
        adapter.loadState(dir);
        return adapter;
    }
}
