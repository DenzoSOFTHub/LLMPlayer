package it.denzosoft.llmplayer.inference;

import it.denzosoft.llmplayer.gpu.CudaBindings;
import it.denzosoft.llmplayer.gpu.CudaBufferManager;
import it.denzosoft.llmplayer.gpu.CudaContext;
import it.denzosoft.llmplayer.model.ModelConfig;
import it.denzosoft.llmplayer.model.ModelWeights;
import it.denzosoft.llmplayer.model.TransformerLayerWeights;
import it.denzosoft.llmplayer.tensor.CudaFloatTensor;
import it.denzosoft.llmplayer.tensor.FloatTensor;
import it.denzosoft.llmplayer.tensor.GGMLType;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * GPU-resident <b>batched</b> forward pass over K tokens at consecutive positions, for
 * speculative-decoding verification. Given K tokens at positions {@code p .. p+K-1}, it produces
 * K logit vectors in a single pass, amortizing each weight read over K inputs.
 *
 * <p><b>Why this is a speedup.</b> At batch=1 the matmul is DRAM-bandwidth bound — the cost is
 * reading the weight matrix once per output row. Processing K input vectors against the same
 * weight matrix reads each weight row ONCE and computes K dot products (see
 * {@code matmul_q4_k_dp4a_batched.cu}). So the (dominant) matmul cost is amortized up to ~K×.
 * The per-token ops (RMSNorm, RoPE, KV-cache update, attention, SiLU) are launched K times each,
 * reusing the existing single-token kernels — no new kernels are needed for them.
 *
 * <p><b>Scope.</b> Dense pre-norm transformers with separate Q/K/V projections — the
 * Llama / Qwen2 / Qwen3 / Mistral family (optionally with Qwen3-style per-head QK-norm). Models
 * with post-norms (Gemma 2/3), packed FFN (Phi-3/4), merged QKV, QKV bias, Granite scaling,
 * sliding-window attention, NoPE layers, or MoE are explicitly out of scope: {@link #isSupported}
 * returns false and the caller falls back to sequential single-token forwards.
 *
 * <p><b>KV-cache semantics.</b> The KV cache is owned by this instance and is stateful, exactly
 * like the single-token GPU pass. A fresh instance starts empty; the caller MUST drive tokens in
 * position order starting from position 0 — each {@link #forwardBatch(int[], int)} call writes the
 * K new K/V vectors at positions {@code startPosition .. startPosition+K-1} and reads the full
 * prefix already present in the cache. In other words, the prefix {@code 0 .. startPosition-1} must
 * have been populated by prior {@code forwardBatch} calls on the SAME instance. There is no priming
 * API — the caller simply replays the accepted prefix through this pass.
 *
 * <p><b>Correctness.</b> {@code forwardBatch(K)} produces the same logits as K sequential
 * single-token forwards (per query, attention sees only positions {@code <= p+k} via causal masking
 * because all K new K/V rows are written before any attention launch, and each query reads its own
 * {@code tokenParams[k] = [p+k, p+k+1]}).
 *
 * <p><b>Buffer-layout contract.</b>
 * <ul>
 *   <li>FP32 batched activation: vector k at float offset {@code k*stride}, where stride is the
 *       vector length in floats (dim, qDim, kvDim, or ffnDim).</li>
 *   <li>Q8_1 batched input: vector k at byte offset {@code k*((cols/32)*40)} (matches
 *       {@code matmul_q4_k_dp4a_batched.cu}'s {@code inputStride}).</li>
 *   <li>Batched output: {@code output[k*rows + row]} (float offset {@code k*rows}).</li>
 *   <li>tokenParams: one int[2*maxBatch] GPU buffer; entry k = {@code [p+k, p+k+1]} at byte offset
 *       {@code k*8}. RoPE / kv-update / attention read {@code gpuTokenParams + k*8}.</li>
 * </ul>
 * No CUDA graph and no dp4a for non-Q4_K weights (those fall back to K FP32 single-token matmuls).
 */
public class BatchedCudaForwardPass implements AutoCloseable {

    private final CudaContext cudaContext;
    private final CudaBufferManager bufferManager;
    private final Arena arena;
    private final MemorySegment defaultStream;
    private final ModelWeights weights;

    private final int dim, qDim, kvDim, ffnDim, vocabSize;
    private final int headCount, headCountKV, headSize;
    private final int halfRope, ropeType;
    private final float normEps, embeddingScale;
    private final long blockSize;
    private final int blockCount;
    private final int maxSeqLen, maxBatch;
    private final boolean hasQKNorm;

    // GPU RoPE tables
    private final long gpuCosTable, gpuSinTable;

    // Per-k token params: int[2*maxBatch] -> entry k = [p+k, p+k+1] at byte offset k*8
    private final long gpuTokenParams;
    private final MemorySegment hostTokenParams;

    // K-wide (batched) activation buffers. Each vector k starts at float offset k*<vecLen>.
    private final long gpuXBatched;       // [maxBatch * dim]   — residual stream per token
    private final long gpuNormBatched;    // [maxBatch * dim]   — normed input (matmul input)
    private final long gpuQBatched;       // [maxBatch * qDim]
    private final long gpuKBatched;       // [maxBatch * kvDim]
    private final long gpuVBatched;       // [maxBatch * kvDim]
    private final long gpuAttnOutBatched; // [maxBatch * qDim]
    private final long gpuGateBatched;    // [maxBatch * ffnDim]
    private final long gpuUpBatched;      // [maxBatch * ffnDim]
    private final long gpuTmpBatched;     // [maxBatch * dim]   — Wo / Down output (then add to X)
    private final long gpuLogitsBatched;  // [maxBatch * vocabSize]

    // Q8_1 batched scratch — sized for the largest matmul input (dim or ffnDim) × maxBatch.
    // Used by the Q4_K batched dp4a kernel. Stride per vector = (cols/32)*40 bytes.
    private final long gpuQ8Batched;
    private final boolean useDp4a;

    // Host staging
    private final MemorySegment hostX;        // [maxBatch * dim]
    private final MemorySegment hostLogits;   // [maxBatch * vocabSize]

    // Per-layer norm + QK-norm weights and KV caches (one cache per layer, [maxSeqLen * kvDim]).
    private final long[] gpuAttnNorm, gpuFfnNorm, gpuQNorm, gpuKNorm;
    private final long[] gpuKeyCache, gpuValueCache;
    private final long gpuOutputNorm;

    // Kernels (all single-token, reused per-k)
    private final MemorySegment rmsnormFunc, perHeadNormFunc, ropeFunc, kvUpdateFunc, attnFunc;
    private final MemorySegment siluMulFunc, accumFunc, quantizeFunc;
    private final MemorySegment matmulBatchedQ4kFunc; // batched Q4_K × Q8_1 dp4a

    // Pre-computed grid sizes
    private final int normSharedMem, perHeadBlockDim, perHeadSharedMem;
    private final int ropeQGrid, ropeKGrid, kvGrid, accumGrid, siluGrid;

    private static final class PB {
        final MemorySegment args, ptrs;
        PB(Arena a, int n) {
            args = a.allocate(n * 8L, 8);
            ptrs = a.allocate(ValueLayout.ADDRESS, n);
            for (int i = 0; i < n; i++) ptrs.setAtIndex(ValueLayout.ADDRESS, i, args.asSlice(i * 8L, 8));
        }
        void setLong(int i, long v) { args.set(ValueLayout.JAVA_LONG, i * 8L, v); }
        void setInt(int i, int v) { args.set(ValueLayout.JAVA_INT, i * 8L, v); }
        void setFloat(int i, float v) { args.set(ValueLayout.JAVA_FLOAT, i * 8L, v); }
    }

    private final PB normPB, perHeadPB, ropePB, kvPB, attnPB, siluPB, accumPB;
    private final PB quantPB, matmulPB, batchedPB;

    public BatchedCudaForwardPass(ModelConfig config, ModelWeights weights,
                                  CudaBufferManager bufferManager, int maxSeqLen, int maxBatch) {
        this.cudaContext = bufferManager.getCudaContext();
        this.bufferManager = bufferManager;
        this.weights = weights;
        this.arena = Arena.ofShared();
        this.defaultStream = cudaContext.getStream();
        this.maxSeqLen = maxSeqLen;
        this.maxBatch = Math.min(8, Math.max(1, maxBatch)); // batched kernel caps K at MAX_BATCH=8

        this.dim = config.embeddingLength();
        this.qDim = config.headCount() * config.headSize();
        this.kvDim = config.kvDim();
        this.ffnDim = config.intermediateSize();
        this.vocabSize = config.vocabSize();
        this.headCount = config.headCount();
        this.headCountKV = config.headCountKV();
        this.headSize = config.headSize();
        this.normEps = config.normEps();
        this.blockCount = config.blockCount();

        // Embedding scale: Gemma uses sqrt(dim), Granite uses an explicit factor. For the
        // Llama/Qwen/Mistral family this is 0 (no scaling) — matches InferenceEngine.embeddingScale.
        if (config.embeddingScale() > 0f) {
            this.embeddingScale = config.embeddingScale();
        } else {
            this.embeddingScale = 0f;
        }

        RoPE rope = new RoPE(headSize, config.ropeDimensionCount(), maxSeqLen,
            config.ropeFreqBase(), config.ropeType(), weights.ropeFreqFactors());
        this.halfRope = rope.getRopeDimCount() / 2;
        this.ropeType = rope.getRopeType();

        long maxWg = cudaContext.getDeviceInfo().maxWorkGroupSize();
        this.blockSize = Math.min(256, maxWg);

        this.hasQKNorm = weights.layers()[0].qNorm() != null;

        long fb = Float.BYTES;

        // Compile kernels
        rmsnormFunc  = cudaContext.compileKernel("kernels/cuda/rmsnorm.cu", "rmsnorm_fused");
        ropeFunc     = cudaContext.compileKernel("kernels/cuda/rope.cu", "rope_apply");
        kvUpdateFunc = cudaContext.compileKernel("kernels/cuda/attention.cu", "kv_cache_update");
        attnFunc     = cudaContext.compileKernel("kernels/cuda/attention.cu", "attention_full");
        siluMulFunc  = cudaContext.compileKernel("kernels/cuda/silu_mul.cu", "silu_mul");
        accumFunc    = cudaContext.compileKernel("kernels/cuda/accumulate.cu", "accumulate");
        perHeadNormFunc = hasQKNorm
            ? cudaContext.compileKernel("kernels/cuda/rmsnorm_per_head.cu", "rmsnorm_per_head") : null;

        boolean dp4aReq = !"false".equals(System.getProperty("cuda.dp4a", "true"));
        MemorySegment qFunc = null, bFunc = null;
        if (dp4aReq) {
            try {
                qFunc = cudaContext.compileKernel("kernels/cuda/quantize_q8.cu", "quantize_q8");
                bFunc = cudaContext.compileKernel("kernels/cuda/matmul_q4_k_dp4a_batched.cu",
                                                  "matmul_q4_k_dp4a_batched");
            } catch (Exception e) {
                System.err.println("BatchedCuda: batched Q4_K dp4a unavailable — " + e.getMessage());
                qFunc = null; bFunc = null;
            }
        }
        quantizeFunc = qFunc;
        matmulBatchedQ4kFunc = bFunc;
        if (DEBUG_SYNC) System.err.println("BATCHED funcs: quant=" + (qFunc == null ? "NULL" : qFunc.address())
            + " batched=" + (bFunc == null ? "NULL" : bFunc.address())
            + " rmsnorm=" + rmsnormFunc.address());
        useDp4a = (qFunc != null && bFunc != null);

        int mb = this.maxBatch;

        // Batched activation buffers
        gpuXBatched       = bufferManager.createBuffer((long) mb * dim * fb);
        gpuNormBatched    = bufferManager.createBuffer((long) mb * dim * fb);
        gpuQBatched       = bufferManager.createBuffer((long) mb * qDim * fb);
        gpuKBatched       = bufferManager.createBuffer((long) mb * kvDim * fb);
        gpuVBatched       = bufferManager.createBuffer((long) mb * kvDim * fb);
        gpuAttnOutBatched = bufferManager.createBuffer((long) mb * qDim * fb);
        gpuGateBatched    = bufferManager.createBuffer((long) mb * ffnDim * fb);
        gpuUpBatched      = bufferManager.createBuffer((long) mb * ffnDim * fb);
        gpuTmpBatched     = bufferManager.createBuffer((long) mb * dim * fb);
        gpuLogitsBatched  = bufferManager.createBuffer((long) mb * vocabSize * fb);

        // Q8_1 batched scratch: per-vector stride = (cols/32)*40. Largest input is max(dim, ffnDim).
        int maxIn = Math.max(dim, ffnDim);
        long q8StrideMax = ((long) (maxIn + 31) / 32) * 40;
        gpuQ8Batched = useDp4a ? bufferManager.createBuffer((long) mb * q8StrideMax) : 0;

        hostX = arena.allocate(ValueLayout.JAVA_FLOAT, (long) mb * dim);
        hostLogits = arena.allocate(ValueLayout.JAVA_FLOAT, (long) mb * vocabSize);

        // Per-k token params buffer (int[2*maxBatch]); entry k at byte offset k*8.
        gpuTokenParams = bufferManager.createBuffer((long) mb * 8);
        hostTokenParams = arena.allocate(ValueLayout.JAVA_INT, (long) mb * 2);

        gpuCosTable = uploadFloatArray(rope.getCosTable());
        gpuSinTable = uploadFloatArray(rope.getSinTable());

        // Per-layer weights + KV cache
        gpuAttnNorm = new long[blockCount];
        gpuFfnNorm  = new long[blockCount];
        gpuQNorm    = hasQKNorm ? new long[blockCount] : null;
        gpuKNorm    = hasQKNorm ? new long[blockCount] : null;
        gpuKeyCache   = new long[blockCount];
        gpuValueCache = new long[blockCount];
        long kvBytes = (long) maxSeqLen * kvDim * fb;
        for (int i = 0; i < blockCount; i++) {
            TransformerLayerWeights lw = weights.layers()[i];
            gpuAttnNorm[i] = uploadNormWeights(lw.attnNorm(), dim);
            gpuFfnNorm[i]  = uploadNormWeights(lw.ffnNorm(), dim);
            if (hasQKNorm) {
                gpuQNorm[i] = uploadNormWeights(lw.qNorm(), headSize);
                gpuKNorm[i] = uploadNormWeights(lw.kNorm(), headSize);
            }
            gpuKeyCache[i] = bufferManager.createBuffer(kvBytes);
            gpuValueCache[i] = bufferManager.createBuffer(kvBytes);
            cudaContext.fillBufferZero(gpuKeyCache[i], kvBytes);
            cudaContext.fillBufferZero(gpuValueCache[i], kvBytes);
        }
        gpuOutputNorm = uploadNormWeights(weights.outputNorm(), dim);

        // Param buffers
        normPB = new PB(arena, 5);
        normPB.setInt(3, dim); normPB.setFloat(4, normEps);

        perHeadPB = hasQKNorm ? new PB(arena, 4) : null;
        if (hasQKNorm) { perHeadPB.setInt(2, headSize); perHeadPB.setFloat(3, normEps); }

        ropePB = new PB(arena, 8);
        ropePB.setLong(1, gpuCosTable); ropePB.setLong(2, gpuSinTable);
        ropePB.setInt(4, headSize); ropePB.setInt(5, halfRope); ropePB.setInt(7, ropeType);

        kvPB = new PB(arena, 6);
        kvPB.setInt(4, kvDim);

        attnPB = new PB(arena, 10);
        attnPB.setInt(4, headCount); attnPB.setInt(5, headCountKV);
        attnPB.setInt(6, headSize); attnPB.setInt(7, kvDim);
        attnPB.setInt(9, 0); // slidingWindow = 0 (full attention; SWA models are rejected)

        siluPB = new PB(arena, 3);
        siluPB.setInt(2, ffnDim);

        accumPB = new PB(arena, 3);
        accumPB.setInt(2, dim);

        quantPB = new PB(arena, 3);
        matmulPB = new PB(arena, 6);
        batchedPB = new PB(arena, 8);

        int normNumWarps = (int) (blockSize / 32);
        this.normSharedMem = (normNumWarps + 1) * Float.BYTES;
        this.perHeadBlockDim = hasQKNorm
            ? (int) Math.min(Math.max(32, ((headSize + 31) / 32) * 32), blockSize) : 0;
        this.perHeadSharedMem = hasQKNorm ? ((perHeadBlockDim / 32) + 1) * Float.BYTES : 0;
        this.ropeQGrid = (int) ((headCount * halfRope + blockSize - 1) / blockSize);
        this.ropeKGrid = (int) ((headCountKV * halfRope + blockSize - 1) / blockSize);
        this.kvGrid = (int) ((kvDim + blockSize - 1) / blockSize);
        this.accumGrid = (int) ((dim + blockSize - 1) / blockSize);
        this.siluGrid = (int) ((ffnDim + blockSize - 1) / blockSize);
    }

    /**
     * Whether this batched pass can handle the model. Restrictive on purpose: dense pre-norm
     * transformer with separate Q/K/V, no post-norm, no packed FFN, no merged QKV, no QKV bias,
     * no Granite scaling, no sliding window, no NoPE layers, no MoE. Optional Qwen3-style per-head
     * QK-norm IS allowed. All matmul weights + the output projection must be GPU-resident
     * {@link CudaFloatTensor}. Caller falls back to sequential single-token forwards when false.
     */
    public static boolean isSupported(ModelConfig config, ModelWeights weights) {
        if (config.expertCount() > 0) return false;
        if (config.useLayerNorm()) return false;
        if (config.slidingWindow() > 0) return false;           // SWA out of scope (attnPB[9]=0 only)
        if (config.noRopeLayerInterval() != 0) return false;    // NoPE layers out of scope
        if (config.residualScale() > 0 || config.attentionScale() > 0) return false; // Granite scaling
        if (config.embeddingScale() > 0f) return false;         // Gemma-style scaling out of scope
        if (config.logitScale() > 0
            && config.architecture() == it.denzosoft.llmplayer.model.ModelArchitecture.GRANITE) return false;

        if (weights.output() == null || !(weights.output() instanceof CudaFloatTensor)) return false;

        TransformerLayerWeights first = weights.layers()[0];
        // Require pre-norm; reject post-norm (Gemma 2/3) and packed FFN / merged QKV (Phi).
        if (first.attnNorm() == null || first.ffnNorm() == null) return false;
        if (first.postAttnNorm() != null || first.postFfnNorm() != null) return false;
        if (first.wqkv() != null) return false;     // merged QKV out of scope
        if (first.wGate() == null) return false;    // packed FFN (Phi) out of scope
        if (first.qBias() != null) return false;    // QKV bias out of scope

        boolean qkn = first.qNorm() != null;
        for (TransformerLayerWeights lw : weights.layers()) {
            // Every layer must be GPU-resident (no partial offload for the batched verifier).
            FloatTensor[] mm = { lw.wq(), lw.wk(), lw.wv(), lw.wo(), lw.wGate(), lw.wUp(), lw.wDown() };
            for (FloatTensor t : mm) if (!(t instanceof CudaFloatTensor)) return false;
            // QK-norm presence must be uniform across layers (we assume it from layer 0).
            if ((lw.qNorm() != null) != qkn) return false;
            // No mixing of post-norm / packed-FFN / merged-QKV / bias mid-stack.
            if (lw.postAttnNorm() != null || lw.postFfnNorm() != null) return false;
            if (lw.wqkv() != null || lw.wGate() == null || lw.qBias() != null) return false;
        }
        return true;
    }

    public int getMaxBatch() { return maxBatch; }

    /**
     * Run a batched forward over {@code tokens.length} tokens at positions
     * {@code startPosition .. startPosition + tokens.length - 1}. Returns {@code float[K][vocabSize]}.
     *
     * <p>The KV cache is stateful: positions {@code 0 .. startPosition-1} must already be populated
     * (by prior calls on this instance). This call writes the K new K/V vectors at the consecutive
     * positions and reads the full prefix.
     *
     * @throws IllegalArgumentException if {@code tokens.length > getMaxBatch()} or the positions
     *         would exceed {@code maxSeqLen}.
     */
    public float[][] forwardBatch(int[] tokens, int startPosition) {
        int K = tokens.length;
        if (K < 1 || K > maxBatch) {
            throw new IllegalArgumentException("batch size " + K + " out of range [1," + maxBatch + "]");
        }
        if (startPosition + K > maxSeqLen) {
            throw new IllegalArgumentException("positions " + startPosition + ".." + (startPosition + K - 1)
                + " exceed maxSeqLen=" + maxSeqLen);
        }

        // 1. Embedding lookup (CPU) into the K x[] slices, with optional embedding scale.
        for (int k = 0; k < K; k++) {
            int token = tokens[k];
            int base = k * dim;
            FloatTensor emb = weights.tokenEmbedding();
            for (int i = 0; i < dim; i++) {
                float v = emb.getFloat((long) token * dim + i);
                if (embeddingScale > 0f) v *= embeddingScale;
                hostX.set(ValueLayout.JAVA_FLOAT, (long) (base + i) * Float.BYTES, v);
            }
        }
        cudaContext.writeBuffer(gpuXBatched, hostX, (long) K * dim * Float.BYTES);

        // 2. Per-k token params [p+k, p+k+1] at byte offset k*8.
        for (int k = 0; k < K; k++) {
            int p = startPosition + k;
            hostTokenParams.set(ValueLayout.JAVA_INT, (long) (2 * k) * Integer.BYTES, p);
            hostTokenParams.set(ValueLayout.JAVA_INT, (long) (2 * k + 1) * Integer.BYTES, p + 1);
        }
        cudaContext.writeBuffer(gpuTokenParams, hostTokenParams, (long) K * 8);

        // 3. Layers
        for (int li = 0; li < blockCount; li++) {
            forwardLayer(li, startPosition, K);
        }

        // 4. Final norm + output projection (batched), per-k.
        for (int k = 0; k < K; k++) {
            normPB.setLong(0, gpuNormBatched + (long) k * dim * Float.BYTES);
            normPB.setLong(1, gpuXBatched + (long) k * dim * Float.BYTES);
            normPB.setLong(2, gpuOutputNorm);
            launch(rmsnormFunc, 1, (int) blockSize, normSharedMem, normPB);
        }
        batchedMatmul((CudaFloatTensor) weights.output(),
                      gpuNormBatched, gpuQ8Batched, gpuLogitsBatched, vocabSize, dim, K);

        // 5. Download logits.
        cudaContext.readBuffer(gpuLogitsBatched, hostLogits, (long) K * vocabSize * Float.BYTES);
        float[][] out = new float[K][vocabSize];
        for (int k = 0; k < K; k++) {
            MemorySegment.copy(hostLogits, ValueLayout.JAVA_FLOAT, (long) k * vocabSize,
                               out[k], 0, vocabSize);
        }
        return out;
    }

    private void forwardLayer(int li, int startPosition, int K) {
        TransformerLayerWeights lw = weights.layers()[li];
        long fb = Float.BYTES;

        // 1. attn norm: x[k] -> normed[k]
        for (int k = 0; k < K; k++) {
            normPB.setLong(0, gpuNormBatched + (long) k * dim * fb);
            normPB.setLong(1, gpuXBatched + (long) k * dim * fb);
            normPB.setLong(2, gpuAttnNorm[li]);
            launch(rmsnormFunc, 1, (int) blockSize, normSharedMem, normPB);
        }

        // 2. Q/K/V projections (batched — amortize weight read over K inputs).
        batchedMatmul((CudaFloatTensor) lw.wq(), gpuNormBatched, gpuQ8Batched, gpuQBatched, qDim, dim, K);
        batchedMatmul((CudaFloatTensor) lw.wk(), gpuNormBatched, gpuQ8Batched, gpuKBatched, kvDim, dim, K);
        batchedMatmul((CudaFloatTensor) lw.wv(), gpuNormBatched, gpuQ8Batched, gpuVBatched, kvDim, dim, K);

        // 3. per-k: optional QK-norm, RoPE on Q and K, KV cache update.
        for (int k = 0; k < K; k++) {
            long qk = gpuQBatched + (long) k * qDim * fb;
            long kk = gpuKBatched + (long) k * kvDim * fb;
            long vk = gpuVBatched + (long) k * kvDim * fb;
            long tpk = gpuTokenParams + (long) k * 8;

            if (hasQKNorm) {
                perHeadPB.setLong(0, qk); perHeadPB.setLong(1, gpuQNorm[li]);
                launch(perHeadNormFunc, headCount, perHeadBlockDim, perHeadSharedMem, perHeadPB);
                perHeadPB.setLong(0, kk); perHeadPB.setLong(1, gpuKNorm[li]);
                launch(perHeadNormFunc, headCountKV, perHeadBlockDim, perHeadSharedMem, perHeadPB);
            }

            ropePB.setLong(6, tpk);
            ropePB.setLong(0, qk); ropePB.setInt(3, headCount);
            launch(ropeFunc, ropeQGrid, (int) blockSize, 0, ropePB);
            ropePB.setLong(0, kk); ropePB.setInt(3, headCountKV);
            launch(ropeFunc, ropeKGrid, (int) blockSize, 0, ropePB);

            // KV cache update at position p+k for THIS layer.
            kvPB.setLong(0, gpuKeyCache[li]); kvPB.setLong(1, gpuValueCache[li]);
            kvPB.setLong(2, kk); kvPB.setLong(3, vk); kvPB.setLong(5, tpk);
            launch(kvUpdateFunc, kvGrid, (int) blockSize, 0, kvPB);
        }

        // 4. per-k attention. All K new K/V are already written, so query k sees positions
        //    0..p+k via causal masking (seqLen = p+k+1 from tokenParams[k][1]).
        for (int k = 0; k < K; k++) {
            int p = startPosition + k;
            attnPB.setLong(0, gpuAttnOutBatched + (long) k * qDim * fb);
            attnPB.setLong(1, gpuQBatched + (long) k * qDim * fb);
            attnPB.setLong(2, gpuKeyCache[li]); attnPB.setLong(3, gpuValueCache[li]);
            attnPB.setLong(8, gpuTokenParams + (long) k * 8);
            int attnSM = (p + 1 + 32) * Float.BYTES;
            launch(attnFunc, headCount, (int) Math.min(256, blockSize), attnSM, attnPB);
        }

        // 5. Wo: attnOut[k] -> tmp[k] (batched).
        batchedMatmul((CudaFloatTensor) lw.wo(), gpuAttnOutBatched, gpuQ8Batched, gpuTmpBatched, dim, qDim, K);

        // 6. per-k residual x[k] += tmp[k]; ffn norm x[k] -> normed[k].
        for (int k = 0; k < K; k++) {
            accumPB.setLong(0, gpuXBatched + (long) k * dim * fb);
            accumPB.setLong(1, gpuTmpBatched + (long) k * dim * fb);
            launch(accumFunc, accumGrid, (int) blockSize, 0, accumPB);

            normPB.setLong(0, gpuNormBatched + (long) k * dim * fb);
            normPB.setLong(1, gpuXBatched + (long) k * dim * fb);
            normPB.setLong(2, gpuFfnNorm[li]);
            launch(rmsnormFunc, 1, (int) blockSize, normSharedMem, normPB);
        }

        // 7. gate + up (batched).
        batchedMatmul((CudaFloatTensor) lw.wGate(), gpuNormBatched, gpuQ8Batched, gpuGateBatched, ffnDim, dim, K);
        batchedMatmul((CudaFloatTensor) lw.wUp(),   gpuNormBatched, gpuQ8Batched, gpuUpBatched,   ffnDim, dim, K);

        // 8. per-k silu_mul: gate[k] = silu(gate[k]) * up[k].
        for (int k = 0; k < K; k++) {
            siluPB.setLong(0, gpuGateBatched + (long) k * ffnDim * fb);
            siluPB.setLong(1, gpuUpBatched + (long) k * ffnDim * fb);
            launch(siluMulFunc, siluGrid, (int) blockSize, 0, siluPB);
        }

        // 9. down: gate[k] -> tmp[k] (batched).
        batchedMatmul((CudaFloatTensor) lw.wDown(), gpuGateBatched, gpuQ8Batched, gpuTmpBatched, dim, ffnDim, K);

        // 10. per-k residual x[k] += tmp[k].
        for (int k = 0; k < K; k++) {
            accumPB.setLong(0, gpuXBatched + (long) k * dim * fb);
            accumPB.setLong(1, gpuTmpBatched + (long) k * dim * fb);
            launch(accumFunc, accumGrid, (int) blockSize, 0, accumPB);
        }
    }

    /**
     * Batched matmul of K input vectors against weight {@code w}, producing {@code out[k*rows+row]}.
     *
     * <p>Q4_K: quantize each of the K FP32 input vectors into the batched Q8_1 buffer
     * ({@code inQ8 + k*inputStride}, inputStride=(cols/32)*40 bytes), then a single batched dp4a
     * launch reads each weight row once and computes all K dot products.
     *
     * <p>Any other type (Q6_K/Q5_K/Q8_0/…): fall back to K separate single-token FP32 matmuls,
     * each reading {@code inFp32 + k*cols} (float offset) → {@code out + k*rows} (float offset),
     * reusing the tensor's own FP32 kernel.
     *
     * @param w       weight tensor (GPU-resident)
     * @param inFp32  batched FP32 input, vector k at float offset k*cols
     * @param inQ8    batched Q8_1 scratch (used only for Q4_K), vector k at byte offset k*(cols/32)*40
     * @param out     batched output, out[k*rows + row]
     */
    private void batchedMatmul(CudaFloatTensor w, long inFp32, long inQ8, long out,
                               int rows, int cols, int K) {
        long fb = Float.BYTES;
        if (useDp4a && w.type() == GGMLType.Q4_K && (cols % 256) == 0) {
            int inputStride = (cols / 32) * 40; // bytes per Q8_1 vector
            // Quantize each input vector into its slot in the batched Q8_1 buffer.
            for (int k = 0; k < K; k++) {
                quantPB.setLong(0, inFp32 + (long) k * cols * fb);
                quantPB.setLong(1, inQ8 + (long) k * inputStride);
                quantPB.setInt(2, cols);
                launch(quantizeFunc, (((cols + 31) / 32) + 7) / 8, 256, 0, quantPB);
            }
            // Single batched launch: grid/block exactly like the single-token dp4a matmul.
            batchedPB.setLong(0, w.getGpuWeights());
            batchedPB.setLong(1, inQ8);
            batchedPB.setLong(2, out);
            batchedPB.setInt(3, rows);
            batchedPB.setInt(4, cols);
            batchedPB.setInt(5, K);
            batchedPB.setInt(6, inputStride);
            batchedPB.setInt(7, 0); // addToOutput = write mode
            launch(matmulBatchedQ4kFunc, w.getMatmulGridDim(rows, cols), w.getMatmulBlockDim(cols), 0, batchedPB);
            return;
        }
        // Fallback: K separate single-token FP32 matmuls via the tensor's own kernel.
        for (int k = 0; k < K; k++) {
            matmulPB.setLong(0, w.getGpuWeights());
            matmulPB.setLong(1, inFp32 + (long) k * cols * fb);
            matmulPB.setLong(2, out + (long) k * rows * fb);
            matmulPB.setInt(3, rows);
            matmulPB.setInt(4, cols);
            matmulPB.setInt(5, 0); // write mode
            launch(w.getCudaFunction(), w.getMatmulGridDim(rows, cols), w.getMatmulBlockDim(cols),
                   w.getMatmulSharedMem(cols), matmulPB);
        }
    }

    private static final boolean DEBUG_SYNC = "true".equals(System.getProperty("cuda.debug", "false"));
    private String lastLaunch = "?";
    private void launch(MemorySegment fn, int grid, int block, int sm, PB params) {
        int err = CudaBindings.launchKernel(fn, grid, 1, 1, block, 1, 1, sm, defaultStream,
                                            params.ptrs, MemorySegment.NULL);
        if (err != CudaBindings.CUDA_SUCCESS) {
            throw new RuntimeException("BatchedCuda CUDA error: " + err + " (launch err; prev=" + lastLaunch + ")");
        }
        if (DEBUG_SYNC) {
            try { cudaContext.finish(); }
            catch (RuntimeException re) {
                throw new RuntimeException("BatchedCuda sync-fail after grid=" + grid
                    + " block=" + block + " sm=" + sm + " (prev=" + lastLaunch + "): " + re.getMessage());
            }
        }
    }
    private void launchDbg(MemorySegment fn, int grid, int block, int sm, PB params, String name) {
        lastLaunch = name; launch(fn, grid, block, sm, params);
    }

    private long uploadNormWeights(FloatTensor t, int size) {
        float[] w = new float[size];
        for (int i = 0; i < size; i++) w[i] = t.getFloat(i);
        return bufferManager.uploadNormWeights(w);
    }

    private long uploadFloatArray(float[] data) {
        long bytes = (long) data.length * Float.BYTES;
        long ptr = bufferManager.createBuffer(bytes);
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment host = temp.allocate(ValueLayout.JAVA_FLOAT, data.length);
            MemorySegment.copy(data, 0, host, ValueLayout.JAVA_FLOAT, 0, data.length);
            cudaContext.writeBuffer(ptr, host, bytes);
        }
        return ptr;
    }

    @Override
    public void close() {
        try { cudaContext.freeBuffer(gpuXBatched); } catch (Exception ignored) {}
        try { cudaContext.freeBuffer(gpuNormBatched); } catch (Exception ignored) {}
        try { cudaContext.freeBuffer(gpuQBatched); } catch (Exception ignored) {}
        try { cudaContext.freeBuffer(gpuKBatched); } catch (Exception ignored) {}
        try { cudaContext.freeBuffer(gpuVBatched); } catch (Exception ignored) {}
        try { cudaContext.freeBuffer(gpuAttnOutBatched); } catch (Exception ignored) {}
        try { cudaContext.freeBuffer(gpuGateBatched); } catch (Exception ignored) {}
        try { cudaContext.freeBuffer(gpuUpBatched); } catch (Exception ignored) {}
        try { cudaContext.freeBuffer(gpuTmpBatched); } catch (Exception ignored) {}
        try { cudaContext.freeBuffer(gpuLogitsBatched); } catch (Exception ignored) {}
        if (gpuQ8Batched != 0) try { cudaContext.freeBuffer(gpuQ8Batched); } catch (Exception ignored) {}
        try { cudaContext.freeBuffer(gpuTokenParams); } catch (Exception ignored) {}
        try { cudaContext.freeBuffer(gpuCosTable); } catch (Exception ignored) {}
        try { cudaContext.freeBuffer(gpuSinTable); } catch (Exception ignored) {}
        try { cudaContext.freeBuffer(gpuOutputNorm); } catch (Exception ignored) {}
        for (long ptr : gpuAttnNorm) if (ptr != 0) try { cudaContext.freeBuffer(ptr); } catch (Exception ignored) {}
        for (long ptr : gpuFfnNorm)  if (ptr != 0) try { cudaContext.freeBuffer(ptr); } catch (Exception ignored) {}
        if (gpuQNorm != null) for (long ptr : gpuQNorm) if (ptr != 0) try { cudaContext.freeBuffer(ptr); } catch (Exception ignored) {}
        if (gpuKNorm != null) for (long ptr : gpuKNorm) if (ptr != 0) try { cudaContext.freeBuffer(ptr); } catch (Exception ignored) {}
        for (long ptr : gpuKeyCache)   if (ptr != 0) try { cudaContext.freeBuffer(ptr); } catch (Exception ignored) {}
        for (long ptr : gpuValueCache) if (ptr != 0) try { cudaContext.freeBuffer(ptr); } catch (Exception ignored) {}
        arena.close();
    }
}
