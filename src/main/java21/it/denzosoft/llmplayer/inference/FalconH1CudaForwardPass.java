package it.denzosoft.llmplayer.inference;

import it.denzosoft.llmplayer.gpu.CudaBindings;
import it.denzosoft.llmplayer.gpu.CudaBufferManager;
import it.denzosoft.llmplayer.gpu.CudaContext;
import it.denzosoft.llmplayer.model.FalconH1LayerWeights;
import it.denzosoft.llmplayer.model.FalconH1Weights;
import it.denzosoft.llmplayer.model.ModelConfig;
import it.denzosoft.llmplayer.tensor.CudaFloatTensor;
import it.denzosoft.llmplayer.tensor.FloatTensor;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * GPU-resident per-layer forward pass for Falcon-H1 (parallel Mamba-2 + GQA attention hybrid).
 * Every layer runs both a GQA attention path and a Mamba-2 SSM path on the same pre-normed input,
 * sums the two outputs, then a SwiGLU FFN — all GPU-resident, reusing the Nemotron-H Mamba-2
 * kernels (conv1d_short, mamba2_dt_softplus, mamba2_scan, mamba2_gate_norm) + the standard
 * attention/rope/rmsnorm/silu_mul kernels. FP32 matmuls (no dp4a/graph in this first version).
 * Gated by {@link #isSupported}: falls back to the per-tensor path if any weight is not GPU-resident.
 * Channel/attention multipliers are baked into the GGUF weights, so none are applied here.
 */
public class FalconH1CudaForwardPass implements AutoCloseable {

    private final CudaContext cudaContext;
    private final CudaBufferManager bufferManager;
    private final Arena arena;
    private final MemorySegment defaultStream;
    private final FalconH1Weights weights;

    private final int dim, vocabSize, blockCount, maxSeqLen;
    private final int headCount, headCountKV, headSize, kvDim, qDim, ffnDim;
    private final int ssmInner, ssmState, ssmGroups, nheads, ssmConv, convChannels, projDim, headDim;
    private final int halfRope, ropeType;
    private final float normEps;
    private final long blockSize;

    private final long gpuCombined, gpuX, gpuTokenParams;
    private final long gpuNorm, gpuAttnRes, gpuSsmRes;
    private final long gpuQ, gpuK, gpuV, gpuAttnOut;
    private final long gpuZxBCdt, gpuXBC, gpuDt, gpuY;
    private final long gpuGate, gpuUp;
    private final long gpuLogits, gpuLogitsBytes, gpuOutputNorm;
    private final long gpuCosTable, gpuSinTable;
    private final MemorySegment hostCombined, hostX, hostLogits;

    private final long[] gpuAttnNorm, gpuFfnNorm;
    private final long[] gpuConvW, gpuConvBias, gpuDtBias, gpuSsmA, gpuSsmD, gpuSsmNorm, gpuConvState, gpuSsmState;
    private final long[] gpuKeyCache, gpuValueCache;

    private final MemorySegment rmsnormFunc, ropeFunc, kvUpdateFunc, attnFunc;
    private final MemorySegment convFunc, dtSoftplusFunc, scanFunc, gateNormFunc, siluFunc, siluMulFunc, elemMulFunc, accumFunc;

    // dp4a (int8) matmul path: quantize FP32 input -> Q8_1, then per-type dp4a kernel (FP32 fallback).
    // OPT-IN for Falcon-H1 (default off): measured neutral-to-slower than FP32 here because the
    // per-token cost is dominated by the Mamba-2 scan + many small matmuls, so the extra per-matmul
    // quantize launch outweighs the int8 speedup (unlike LFM2, where dp4a is a clear +37%).
    private final boolean useDp4a = "true".equals(System.getProperty("cuda.falcon.dp4a", "false"));
    private final MemorySegment quantizeFunc, dp4aQ4kFunc, dp4aQ5kFunc, dp4aQ50Func, dp4aQ80Func,
                                dp4aQ3kFunc, dp4aIq4nlFunc, dp4aIq4xsFunc;
    private final long gpuQ8In;
    private final PB quantPB, dp4aPB;

    private final int normSharedMem, ropeQGrid, ropeKGrid, kvGrid, convGrid, accumGrid, ffnGrid, innerGrid;
    private final int gateNormBlockDim, gateNormSharedMem;

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

    private final PB matmulPB, normPB, ropePB, kvPB, attnPB, convPB, convBiasPB, dtPB, scanPB, gateNormPB, siluPB, siluMulPB, elemMulPB, accumPB;

    public FalconH1CudaForwardPass(ModelConfig config, FalconH1Weights weights,
                                   CudaBufferManager bufferManager, int maxSeqLen) {
        this.cudaContext = bufferManager.getCudaContext();
        this.bufferManager = bufferManager;
        this.weights = weights;
        this.arena = Arena.ofShared();
        this.defaultStream = cudaContext.getStream();
        this.maxSeqLen = maxSeqLen;

        this.dim = config.embeddingLength();
        this.vocabSize = config.vocabSize();
        this.blockCount = config.blockCount();
        this.headCount = config.headCount();
        this.headCountKV = config.headCountKV();
        this.headSize = config.headSize();
        this.kvDim = config.kvDim();
        this.qDim = headCount * headSize;
        this.ffnDim = config.intermediateSize();
        this.ssmInner = config.ssmInnerSize();
        this.ssmState = config.ssmStateSize();
        this.ssmGroups = config.ssmGroupCount();
        this.nheads = config.ssmTimeStepRank();
        this.ssmConv = config.ssmConvKernel();
        this.convChannels = ssmInner + 2 * ssmGroups * ssmState;
        this.projDim = ssmInner + convChannels + nheads;
        this.headDim = ssmInner / nheads;
        this.normEps = config.normEps();
        long maxWg = cudaContext.getDeviceInfo().maxWorkGroupSize();
        this.blockSize = Math.min(256, maxWg);

        RoPE rope = new RoPE(headSize, config.ropeDimensionCount(), maxSeqLen,
            config.ropeFreqBase(), config.ropeType(), weights.ropeFreqFactors());
        this.halfRope = rope.getRopeDimCount() / 2;
        this.ropeType = rope.getRopeType();

        long fb = Float.BYTES;

        rmsnormFunc    = cudaContext.compileKernel("kernels/cuda/rmsnorm.cu", "rmsnorm_fused");
        ropeFunc       = cudaContext.compileKernel("kernels/cuda/rope.cu", "rope_apply");
        kvUpdateFunc   = cudaContext.compileKernel("kernels/cuda/attention.cu", "kv_cache_update");
        attnFunc       = cudaContext.compileKernel("kernels/cuda/attention.cu", "attention_full");
        convFunc       = cudaContext.compileKernel("kernels/cuda/conv1d_short.cu", "conv1d_short");
        dtSoftplusFunc = cudaContext.compileKernel("kernels/cuda/mamba2_dt_softplus.cu", "mamba2_dt_softplus");
        scanFunc       = cudaContext.compileKernel("kernels/cuda/mamba2_scan.cu", "mamba2_scan");
        gateNormFunc   = cudaContext.compileKernel("kernels/cuda/mamba2_gate_norm.cu", "mamba2_gate_norm");
        siluFunc       = cudaContext.compileKernel("kernels/cuda/silu.cu", "silu");
        siluMulFunc    = cudaContext.compileKernel("kernels/cuda/silu_mul.cu", "silu_mul");
        elemMulFunc    = cudaContext.compileKernel("kernels/cuda/elementwise_mul.cu", "elementwise_mul");
        accumFunc      = cudaContext.compileKernel("kernels/cuda/accumulate.cu", "accumulate");

        if (useDp4a) {
            quantizeFunc  = cudaContext.compileKernel("kernels/cuda/quantize_q8.cu", "quantize_q8");
            dp4aQ4kFunc   = cudaContext.compileKernel("kernels/cuda/matmul_q4_k_dp4a.cu", "matmul_q4_k_dp4a");
            dp4aQ5kFunc   = cudaContext.compileKernel("kernels/cuda/matmul_q5_k_dp4a.cu", "matmul_q5_k_dp4a");
            dp4aQ50Func   = cudaContext.compileKernel("kernels/cuda/matmul_q5_0_dp4a.cu", "matmul_q5_0_dp4a");
            dp4aQ80Func   = cudaContext.compileKernel("kernels/cuda/matmul_q8_0_dp4a.cu", "matmul_q8_0_dp4a");
            dp4aQ3kFunc   = cudaContext.compileKernel("kernels/cuda/matmul_q3_k_dp4a.cu", "matmul_q3_k_dp4a");
            dp4aIq4nlFunc = cudaContext.compileKernel("kernels/cuda/matmul_iq4_nl_dp4a.cu", "matmul_iq4_nl_dp4a");
            dp4aIq4xsFunc = cudaContext.compileKernel("kernels/cuda/matmul_iq4_xs_dp4a.cu", "matmul_iq4_xs_dp4a");
        } else {
            quantizeFunc = dp4aQ4kFunc = dp4aQ5kFunc = dp4aQ50Func = dp4aQ80Func
                = dp4aQ3kFunc = dp4aIq4nlFunc = dp4aIq4xsFunc = null;
        }

        long combinedBytes = dim * fb + 8;
        gpuCombined = bufferManager.createBuffer(combinedBytes);
        gpuX = gpuCombined;
        gpuTokenParams = gpuCombined + dim * fb;
        hostCombined = arena.allocate(combinedBytes, 8);
        hostX = arena.allocate(ValueLayout.JAVA_FLOAT, dim);

        gpuNorm    = bufferManager.createBuffer(dim * fb);
        gpuAttnRes = bufferManager.createBuffer(dim * fb);
        gpuSsmRes  = bufferManager.createBuffer(dim * fb);
        gpuQ       = bufferManager.createBuffer((long) qDim * fb);
        gpuK       = bufferManager.createBuffer((long) kvDim * fb);
        gpuV       = bufferManager.createBuffer((long) kvDim * fb);
        gpuAttnOut = bufferManager.createBuffer((long) qDim * fb);
        gpuZxBCdt  = bufferManager.createBuffer((long) projDim * fb);
        gpuXBC     = bufferManager.createBuffer((long) convChannels * fb);
        gpuDt      = bufferManager.createBuffer((long) nheads * fb);
        gpuY       = bufferManager.createBuffer((long) ssmInner * fb);
        gpuGate    = bufferManager.createBuffer((long) ffnDim * fb);
        gpuUp      = bufferManager.createBuffer((long) ffnDim * fb);
        int maxIn = Math.max(Math.max(dim, ffnDim), Math.max(ssmInner, convChannels));
        gpuQ8In = useDp4a ? bufferManager.createBuffer((long) ((maxIn + 31) / 32) * 40) : 0;

        gpuCosTable = uploadFloatArray(rope.getCosTable());
        gpuSinTable = uploadFloatArray(rope.getSinTable());

        gpuAttnNorm = new long[blockCount]; gpuFfnNorm = new long[blockCount];
        gpuConvW = new long[blockCount]; gpuConvBias = new long[blockCount]; gpuDtBias = new long[blockCount];
        gpuSsmA = new long[blockCount]; gpuSsmD = new long[blockCount]; gpuSsmNorm = new long[blockCount];
        gpuConvState = new long[blockCount]; gpuSsmState = new long[blockCount];
        gpuKeyCache = new long[blockCount]; gpuValueCache = new long[blockCount];
        long kvBytes = (long) maxSeqLen * kvDim * fb;
        long convBytes = (long) (ssmConv - 1) * convChannels * fb;
        long stateBytes = (long) nheads * headDim * ssmState * fb;
        for (int i = 0; i < blockCount; i++) {
            FalconH1LayerWeights lw = weights.layers()[i];
            gpuAttnNorm[i] = uploadNorm(lw.attnNorm(), dim);
            gpuFfnNorm[i] = uploadNorm(lw.ffnNorm(), dim);
            gpuConvW[i] = uploadFloats(lw.ssmConv1d(), ssmConv * convChannels);
            gpuConvBias[i] = uploadFloats(lw.ssmConv1dBias(), convChannels);
            gpuDtBias[i] = uploadFloats(lw.ssmDtBias(), nheads);
            gpuSsmA[i] = uploadFloats(lw.ssmA(), nheads);
            gpuSsmD[i] = uploadFloats(lw.ssmD(), nheads);
            gpuSsmNorm[i] = (lw.ssmNorm() != null) ? uploadNorm(lw.ssmNorm(), ssmInner) : 0;
            gpuConvState[i] = bufferManager.createBuffer(convBytes);
            gpuSsmState[i] = bufferManager.createBuffer(stateBytes);
            gpuKeyCache[i] = bufferManager.createBuffer(kvBytes);
            gpuValueCache[i] = bufferManager.createBuffer(kvBytes);
        }
        gpuOutputNorm = uploadNorm(weights.outputNorm(), dim);
        gpuLogits = bufferManager.createBuffer((long) vocabSize * fb);
        gpuLogitsBytes = (long) vocabSize * fb;
        hostLogits = arena.allocate(ValueLayout.JAVA_FLOAT, vocabSize);

        matmulPB = new PB(arena, 6);
        quantPB = new PB(arena, 3);
        dp4aPB = new PB(arena, 6);
        normPB = new PB(arena, 5);
        normPB.setLong(0, gpuNorm); normPB.setLong(1, gpuX); normPB.setInt(3, dim); normPB.setFloat(4, normEps);
        ropePB = new PB(arena, 8);
        ropePB.setLong(1, gpuCosTable); ropePB.setLong(2, gpuSinTable);
        ropePB.setInt(4, headSize); ropePB.setInt(5, halfRope); ropePB.setLong(6, gpuTokenParams); ropePB.setInt(7, ropeType);
        kvPB = new PB(arena, 6);
        kvPB.setLong(2, gpuK); kvPB.setLong(3, gpuV); kvPB.setInt(4, kvDim); kvPB.setLong(5, gpuTokenParams);
        attnPB = new PB(arena, 10);
        attnPB.setLong(0, gpuAttnOut); attnPB.setLong(1, gpuQ);
        attnPB.setInt(4, headCount); attnPB.setInt(5, headCountKV);
        attnPB.setInt(6, headSize); attnPB.setInt(7, kvDim); attnPB.setLong(8, gpuTokenParams); attnPB.setInt(9, 0);
        convPB = new PB(arena, 6);
        convPB.setLong(0, gpuXBC); convPB.setInt(3, convChannels); convPB.setInt(4, ssmConv); convPB.setLong(5, gpuTokenParams);
        convBiasPB = new PB(arena, 3);
        convBiasPB.setLong(0, gpuXBC); convBiasPB.setInt(2, convChannels);
        dtPB = new PB(arena, 3);
        dtPB.setLong(0, gpuDt); dtPB.setInt(2, nheads);
        scanPB = new PB(arena, 12);
        int bOff = ssmInner; int cOff = ssmInner + ssmGroups * ssmState;
        scanPB.setLong(1, gpuXBC); scanPB.setLong(2, gpuXBC + (long) bOff * fb); scanPB.setLong(3, gpuXBC + (long) cOff * fb);
        scanPB.setLong(4, gpuDt); scanPB.setLong(7, gpuY);
        scanPB.setInt(8, nheads); scanPB.setInt(9, headDim); scanPB.setInt(10, ssmState); scanPB.setInt(11, ssmGroups);
        gateNormPB = new PB(arena, 6);
        gateNormPB.setLong(0, gpuY); gateNormPB.setLong(1, gpuZxBCdt);
        gateNormPB.setInt(3, ssmInner); gateNormPB.setInt(4, ssmGroups); gateNormPB.setFloat(5, normEps);
        siluPB = new PB(arena, 2);
        siluPB.setLong(0, gpuZxBCdt); siluPB.setInt(1, ssmInner);   // gate-only path: silu(z) in place
        siluMulPB = new PB(arena, 3);
        siluMulPB.setLong(0, gpuGate); siluMulPB.setLong(1, gpuUp); siluMulPB.setInt(2, ffnDim);
        elemMulPB = new PB(arena, 3);
        elemMulPB.setLong(0, gpuY); elemMulPB.setLong(1, gpuZxBCdt); elemMulPB.setInt(2, ssmInner);
        accumPB = new PB(arena, 3);
        accumPB.setInt(2, dim);

        int normNumWarps = (int) (blockSize / 32);
        this.normSharedMem = (normNumWarps + 1) * Float.BYTES;
        this.ropeQGrid = (int) ((headCount * halfRope + blockSize - 1) / blockSize);
        this.ropeKGrid = (int) ((headCountKV * halfRope + blockSize - 1) / blockSize);
        this.kvGrid = (int) ((kvDim + blockSize - 1) / blockSize);
        this.convGrid = (int) ((convChannels + blockSize - 1) / blockSize);
        this.accumGrid = (int) ((dim + blockSize - 1) / blockSize);
        this.ffnGrid = (int) ((ffnDim + blockSize - 1) / blockSize);
        this.innerGrid = (int) ((ssmInner + blockSize - 1) / blockSize);
        this.gateNormBlockDim = (int) Math.min(256, maxWg);
        this.gateNormSharedMem = (gateNormBlockDim / 32 + 1) * Float.BYTES;
    }

    public static boolean isSupported(ModelConfig config, FalconH1Weights weights) {
        if (weights.layers().length == 0) return false;
        if (!(weights.output() instanceof CudaFloatTensor)) return false;
        for (FalconH1LayerWeights lw : weights.layers()) {
            FloatTensor[] mm = { lw.wq(), lw.wk(), lw.wv(), lw.wo(), lw.ssmIn(), lw.ssmOut(),
                                 lw.ffnGate(), lw.ffnUp(), lw.ffnDown() };
            for (FloatTensor t : mm) if (!(t instanceof CudaFloatTensor)) return false;
            // FFN/attn biases not handled on GPU yet — fall back if present.
            if (lw.woBias() != null || lw.ffnGateBias() != null || lw.ffnUpBias() != null || lw.ffnDownBias() != null)
                return false;
        }
        return true;
    }

    public int getGpuLayerCount() { return blockCount; }

    public void uploadXAndUpdateParams(float[] x, int position) {
        long embBytes = (long) dim * Float.BYTES;
        MemorySegment.copy(x, 0, hostCombined, ValueLayout.JAVA_FLOAT, 0, dim);
        hostCombined.set(ValueLayout.JAVA_INT, embBytes, position);
        hostCombined.set(ValueLayout.JAVA_INT, embBytes + 4, position + 1);
        cudaContext.writeBuffer(gpuCombined, hostCombined, embBytes + 8);
    }

    public void downloadX(float[] x) {
        cudaContext.readBuffer(gpuX, hostX, (long) dim * Float.BYTES);
        MemorySegment.copy(hostX, ValueLayout.JAVA_FLOAT, 0, x, 0, dim);
    }

    public void forwardLayer(int li, int position) {
        FalconH1LayerWeights lw = weights.layers()[li];
        long fb = Float.BYTES;

        // shared pre-norm: gpuX -> gpuNorm (feeds both attention and mamba)
        normPB.setLong(2, gpuAttnNorm[li]);
        launch(rmsnormFunc, 1, (int) blockSize, normSharedMem, normPB);

        // ---- attention path -> gpuAttnRes ----
        matmul((CudaFloatTensor) lw.wq(), gpuNorm, gpuQ, qDim, dim);
        matmul((CudaFloatTensor) lw.wk(), gpuNorm, gpuK, kvDim, dim);
        matmul((CudaFloatTensor) lw.wv(), gpuNorm, gpuV, kvDim, dim);
        ropePB.setLong(0, gpuQ); ropePB.setInt(3, headCount);
        launch(ropeFunc, ropeQGrid, (int) blockSize, 0, ropePB);
        ropePB.setLong(0, gpuK); ropePB.setInt(3, headCountKV);
        launch(ropeFunc, ropeKGrid, (int) blockSize, 0, ropePB);
        kvPB.setLong(0, gpuKeyCache[li]); kvPB.setLong(1, gpuValueCache[li]);
        launch(kvUpdateFunc, kvGrid, (int) blockSize, 0, kvPB);
        attnPB.setLong(2, gpuKeyCache[li]); attnPB.setLong(3, gpuValueCache[li]);
        int attnSM = (position + 1 + 32) * Float.BYTES;
        launch(attnFunc, headCount, Math.min(256, (int) blockSize), attnSM, attnPB);
        matmul((CudaFloatTensor) lw.wo(), gpuAttnOut, gpuAttnRes, dim, qDim);

        // ---- mamba-2 path -> gpuSsmRes ----
        matmul((CudaFloatTensor) lw.ssmIn(), gpuNorm, gpuZxBCdt, projDim, dim);
        cudaContext.copyBufferDtoD(gpuXBC, gpuZxBCdt + (long) ssmInner * fb, (long) convChannels * fb);
        cudaContext.copyBufferDtoD(gpuDt, gpuZxBCdt + (long) (ssmInner + convChannels) * fb, (long) nheads * fb);
        // conv1d (plain) -> +bias -> SiLU
        convPB.setLong(1, gpuConvState[li]); convPB.setLong(2, gpuConvW[li]);
        launch(convFunc, convGrid, (int) blockSize, 0, convPB);
        convBiasPB.setLong(1, gpuConvBias[li]);
        launch(accumFunc, convGrid, (int) blockSize, 0, convBiasPB);
        siluPB.setLong(0, gpuXBC); siluPB.setInt(1, convChannels);
        launch(siluFunc, convGrid, (int) blockSize, 0, siluPB);
        // dt softplus (dt += dt_bias, softplus)
        dtPB.setLong(1, gpuDtBias[li]);
        launch(dtSoftplusFunc, 1, Math.max(32, nheads), 0, dtPB);
        // SSD scan
        scanPB.setLong(0, gpuSsmState[li]); scanPB.setLong(5, gpuSsmA[li]); scanPB.setLong(6, gpuSsmD[li]);
        launch(scanFunc, nheads, headDim, 0, scanPB);
        // gate (+ grouped norm if present)
        if (gpuSsmNorm[li] != 0) {
            gateNormPB.setLong(2, gpuSsmNorm[li]);
            launch(gateNormFunc, ssmGroups, gateNormBlockDim, gateNormSharedMem, gateNormPB);
        } else {
            // gate only: y *= silu(z) ; z = gpuZxBCdt[0:ssmInner]
            siluPB.setLong(0, gpuZxBCdt); siluPB.setInt(1, ssmInner);
            launch(siluFunc, innerGrid, (int) blockSize, 0, siluPB);
            elemMulPB.setLong(0, gpuY); elemMulPB.setLong(1, gpuZxBCdt); elemMulPB.setInt(2, ssmInner);
            launch(elemMulFunc, innerGrid, (int) blockSize, 0, elemMulPB);
        }
        matmul((CudaFloatTensor) lw.ssmOut(), gpuY, gpuSsmRes, dim, ssmInner);

        // ---- parallel aggregation: x += attn_out + ssm_out ----
        accumPB.setLong(0, gpuX); accumPB.setLong(1, gpuAttnRes); launch(accumFunc, accumGrid, (int) blockSize, 0, accumPB);
        accumPB.setLong(0, gpuX); accumPB.setLong(1, gpuSsmRes); launch(accumFunc, accumGrid, (int) blockSize, 0, accumPB);

        // ---- SwiGLU FFN ----
        normPB.setLong(2, gpuFfnNorm[li]);
        launch(rmsnormFunc, 1, (int) blockSize, normSharedMem, normPB);
        matmul((CudaFloatTensor) lw.ffnGate(), gpuNorm, gpuGate, ffnDim, dim);
        matmul((CudaFloatTensor) lw.ffnUp(), gpuNorm, gpuUp, ffnDim, dim);
        launch(siluMulFunc, ffnGrid, (int) blockSize, 0, siluMulPB);
        matmul((CudaFloatTensor) lw.ffnDown(), gpuGate, gpuAttnRes, dim, ffnDim);
        accumPB.setLong(0, gpuX); accumPB.setLong(1, gpuAttnRes); launch(accumFunc, accumGrid, (int) blockSize, 0, accumPB);
    }

    public boolean forwardFinalLogits(float[] logits) {
        normPB.setLong(2, gpuOutputNorm);
        launch(rmsnormFunc, 1, (int) blockSize, normSharedMem, normPB);
        matmul((CudaFloatTensor) weights.output(), gpuNorm, gpuLogits, vocabSize, dim);
        cudaContext.readBuffer(gpuLogits, hostLogits, gpuLogitsBytes);
        MemorySegment.copy(hostLogits, ValueLayout.JAVA_FLOAT, 0, logits, 0, vocabSize);
        return true;
    }

    private void matmul(CudaFloatTensor t, long in, long out, int rows, int cols) {
        MemorySegment dp4a = useDp4a ? dp4aFunc(t) : null;
        if (dp4a != null) {
            quantPB.setLong(0, in); quantPB.setLong(1, gpuQ8In); quantPB.setInt(2, cols);
            launch(quantizeFunc, (((cols + 31) / 32) + 7) / 8, 256, 0, quantPB);
            dp4aPB.setLong(0, t.getGpuWeights()); dp4aPB.setLong(1, gpuQ8In); dp4aPB.setLong(2, out);
            dp4aPB.setInt(3, rows); dp4aPB.setInt(4, cols); dp4aPB.setInt(5, 0);
            launch(dp4a, t.getMatmulGridDim(rows, cols), t.getMatmulBlockDim(cols), 0, dp4aPB);
            return;
        }
        matmulPB.setLong(0, t.getGpuWeights()); matmulPB.setLong(1, in); matmulPB.setLong(2, out);
        matmulPB.setInt(3, rows); matmulPB.setInt(4, cols); matmulPB.setInt(5, 0);
        launch(t.getCudaFunction(), t.getMatmulGridDim(rows, cols), t.getMatmulBlockDim(cols), t.getMatmulSharedMem(cols), matmulPB);
    }

    private MemorySegment dp4aFunc(CudaFloatTensor t) {
        switch (t.type()) {
            case Q4_K:   return dp4aQ4kFunc;
            case Q5_K:   return dp4aQ5kFunc;
            case Q5_0:   return dp4aQ50Func;
            case Q8_0:   return dp4aQ80Func;
            case Q3_K:   return dp4aQ3kFunc;
            case IQ4_NL: return dp4aIq4nlFunc;
            case IQ4_XS: return dp4aIq4xsFunc;
            default:     return null;
        }
    }

    private void launch(MemorySegment fn, int grid, int block, int sm, PB params) {
        int err = CudaBindings.launchKernel(fn, grid, 1, 1, block, 1, 1, sm, defaultStream, params.ptrs, MemorySegment.NULL);
        if (err != CudaBindings.CUDA_SUCCESS) throw new RuntimeException("FalconH1 CUDA error: " + err);
    }

    private long uploadNorm(FloatTensor t, int size) {
        float[] w = new float[size]; for (int i = 0; i < size; i++) w[i] = t.getFloat(i);
        return bufferManager.uploadNormWeights(w);
    }

    private long uploadFloats(FloatTensor t, int size) {
        float[] w = new float[size]; for (int i = 0; i < size; i++) w[i] = t.getFloat(i);
        return uploadFloatArray(w);
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
    public void close() { arena.close(); }
}
