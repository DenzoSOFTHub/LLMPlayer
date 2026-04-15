package it.denzosoft.llmplayer.inference;

import it.denzosoft.llmplayer.gpu.CudaBindings;
import it.denzosoft.llmplayer.gpu.CudaBufferManager;
import it.denzosoft.llmplayer.gpu.CudaContext;
import it.denzosoft.llmplayer.model.ModelConfig;
import it.denzosoft.llmplayer.model.NemotronHLayerWeights;
import it.denzosoft.llmplayer.model.NemotronHWeights;
import it.denzosoft.llmplayer.tensor.CudaFloatTensor;
import it.denzosoft.llmplayer.tensor.FloatTensor;

import java.lang.foreign.*;

/**
 * CUDA GPU-resident forward pass for Nemotron-H hybrid Mamba-2 + Attention + FFN architecture.
 * Supports CUDA graph for all layer types.
 */
public class NemotronHCudaForwardPass implements AutoCloseable {

    private final CudaContext cudaContext;
    private final CudaBufferManager bufferManager;
    private final Arena arena;
    private final MemorySegment defaultStream;

    // Dimensions
    private final int dim, vocabSize, blockCount, maxSeqLen;
    private final int ssmInnerSize, ssmStateSize, ssmGroupCount, ssmTimeStepRank, ssmConvKernel;
    private final int headDim, convChannels, projDim;
    private final int headCount, headCountKV, headSize, kvDim, halfRope, ropeType;
    private final float normEps;
    private final long blockSize;

    // Layer types
    private final int[] layerTypes; // 0=Mamba, 1=Attention, 2=FFN

    // GPU activation buffers
    private final long gpuX, gpuXb, gpuTokenParams;
    private final long gpuCombined;
    private final MemorySegment hostCombined, hostX;

    // Mamba buffers
    private final long gpuZxBCdt; // [projDim] ssm_in output
    private final long gpuXBC;    // [convChannels] after conv+SiLU
    private final long gpuY;      // [ssmInnerSize] SSM output
    private final long gpuDt;     // [ssmTimeStepRank]

    // Mamba persistent state
    private final long[] gpuSsmState;  // per Mamba layer: [nheads * headDim * stateSize]
    private final long[] gpuConvState; // per Mamba layer: [histSize * convChannels]

    // Attention buffers
    private final long gpuQ, gpuK, gpuV, gpuXb2;
    private final long[] gpuKeyCache, gpuValueCache;
    private final long gpuCosTable, gpuSinTable;

    // FFN buffer
    private final long gpuHb; // [ffnDim]
    private final long gpuHb2; // [ffnDim] — second FFN projection buffer for integrated SwiGLU (Granite Hybrid)
    private final long gpuHbQ8; // [(ffnDim/32)*40] — Q8_1 scratch for down projection (dp4a path)

    // Per-layer weights on GPU
    private final long[] gpuAttnNormWeights;
    private final long[] gpuConvWeights, gpuConvBias, gpuDtBias, gpuSsmA, gpuSsmD, gpuSsmNormWeights;

    // Output
    private final long gpuOutputNormWeights, gpuLogits, gpuLogitsBytes;
    private final MemorySegment hostLogits;

    // Compiled CUDA functions
    private final MemorySegment rmsnormFunc, siluFunc, ropeFunc, kvUpdateFunc, attnFunc;
    private final MemorySegment siluMulFunc;   // silu_mul for integrated SwiGLU FFN
    private final MemorySegment accumulateFunc, conv1dSiluFunc, mamba2ScanFunc;
    private final MemorySegment dtSoftplusFunc, gateNormFunc, sqreluFunc;

    // ParamBuffers (zero-allocation hot path)
    private static class ParamBuffer {
        final MemorySegment args, ptrs;
        ParamBuffer(Arena a, int n) {
            args = a.allocate(n * 8L, 8);
            ptrs = a.allocate(ValueLayout.ADDRESS, n);
            for (int i = 0; i < n; i++) ptrs.setAtIndex(ValueLayout.ADDRESS, i, args.asSlice(i * 8L, 8));
        }
        void setLong(int i, long v) { args.set(ValueLayout.JAVA_LONG, i * 8L, v); }
        void setInt(int i, int v) { args.set(ValueLayout.JAVA_INT, i * 8L, v); }
        void setFloat(int i, float v) { args.set(ValueLayout.JAVA_FLOAT, i * 8L, v); }
    }

    private static class MatmulLaunch {
        final MemorySegment function; final long weightPtr;
        final int gridDim, blockDim, sharedMem, rows, cols, addToOutput;
        final long inputPtr, outputPtr;
        // dp4a: type code (0=ineligible, 4=Q4_K, 5=Q5_K, 6=Q6_K, 50=Q5_0, 80=Q8_0, 41=IQ4_NL, 42=IQ4_XS)
        final int dp4aType;
        // Q8_1 buffer pointer for inputPtr (0 if not dp4a-eligible). Set by caller at construction.
        final long q8InputPtr;
        MatmulLaunch(CudaFloatTensor t, long in, long out, int r, int c, int add) {
            this(t, in, out, r, c, add, 0L);
        }
        MatmulLaunch(CudaFloatTensor t, long in, long out, int r, int c, int add, long q8In) {
            function = t.getCudaFunction(); weightPtr = t.getGpuWeights();
            blockDim = t.getMatmulBlockDim(c); gridDim = t.getMatmulGridDim(r, c);
            sharedMem = t.getMatmulSharedMem(c);
            inputPtr = in; outputPtr = out; rows = r; cols = c; addToOutput = add;
            it.denzosoft.llmplayer.tensor.GGMLType type = t.type();
            if      (type == it.denzosoft.llmplayer.tensor.GGMLType.Q4_K)   dp4aType = 4;
            else if (type == it.denzosoft.llmplayer.tensor.GGMLType.Q5_K)   dp4aType = 5;
            else if (type == it.denzosoft.llmplayer.tensor.GGMLType.Q6_K)   dp4aType = 6;
            else if (type == it.denzosoft.llmplayer.tensor.GGMLType.Q5_0)   dp4aType = 50;
            else if (type == it.denzosoft.llmplayer.tensor.GGMLType.Q8_0)   dp4aType = 80;
            else if (type == it.denzosoft.llmplayer.tensor.GGMLType.IQ4_NL) dp4aType = 41;
            else if (type == it.denzosoft.llmplayer.tensor.GGMLType.IQ4_XS) dp4aType = 42;
            else                                                            dp4aType = 0;
            q8InputPtr = q8In;
        }
    }

    private final ParamBuffer matmulPB, normPB, ropePB, kvPB, attnPB;
    private final ParamBuffer conv1dPB, scanPB, dtPB, gateNormPB, siluPB, sqreluPB, convBiasPB;

    // === dp4a path (mirrors CudaForwardPass extension) ===
    private final boolean useDp4a;
    private final long gpuXbQ8;       // Q8_1 buffer for gpuXb input (size: dim/32 * 40)
    private final ParamBuffer quantizeXbPB;   // 3 args: gpuXb, gpuXbQ8, dim
    private final ParamBuffer dp4aPB;         // 6 args
    private final int quantizeXbGridDim;
    private final MemorySegment quantizeFunc;
    private final MemorySegment dp4aQ4kFunc;
    private final MemorySegment dp4aQ5kFunc;
    private final MemorySegment dp4aQ6kFunc;
    private final MemorySegment dp4aQ50Func;
    private final MemorySegment dp4aQ80Func;
    private final MemorySegment dp4aIq4nlFunc;
    private final MemorySegment dp4aIq4xsFunc;

    // === Granite Hybrid scaling factors (0 = not applied) ===
    // embeddingScale: applied CPU-side before uploadX (typically 12.0 for Granite 4.0-h).
    // logitScale: divide final logits (typically 8.0 for Granite 4.0-h).
    // residualScale: saxpy factor for residual updates (e.g. 0.22 for Granite 4.0-h-micro).
    // attentionScale: replaces standard 1/sqrt(headSize) attention scale.
    private final float graniteEmbeddingScale;
    private final float graniteLogitScale;
    private final float graniteResidualScale;
    private final float graniteAttentionScale;
    private final MemorySegment scaleFunc;       // scale_inplace.cu
    private final MemorySegment accumulateFunc2; // accumulate.cu (for saxpy)
    private final ParamBuffer scalePB;           // 3 args: x, scale, size
    private final ParamBuffer biasPB2;           // 3 args: y, bias, size (for accumulate)
    private final int scaleDimGridDim;
    private final int scaleVocabGridDim;

    // Per-layer matmul descriptors
    // Mamba: [0]=ssmIn, [1]=ssmOut
    // Attention: [0]=wq, [1]=wk, [2]=wv, [3]=wo
    // FFN: [0]=ffnUp, [1]=ffnDown
    private final MatmulLaunch[][] layerMatmuls;

    // Granite Hybrid integrated SwiGLU FFN (inside Mamba/Attention layers where lw.ffnUp() != null).
    // Per-layer: [0]=ffnGate, [1]=ffnUp, [2]=ffnDown. NULL when layer has no integrated FFN.
    private final MatmulLaunch[][] integratedFfnMatmuls;
    private final long[] gpuIntegratedFfnNormWeights;  // per-layer FFN norm (0 when absent)
    private final MatmulLaunch outputMatmul;

    private final int gpuLayerCount;

    // CUDA graph
    private MemorySegment graphExec;
    private final boolean graphAvailable;
    private final int graphAttnSharedMem;

    // Pre-computed grid sizes
    private final int normSharedMem, normNumWarps;
    private final int convGridDim, scanGridDim, siluConvGridDim;
    private final int ropeQGridDim, ropeKGridDim, kvUpdateGridDim;
    private final int gateNormBlockDim, gateNormSharedMem;

    public NemotronHCudaForwardPass(ModelConfig config, NemotronHWeights weights,
                                     CudaBufferManager bufferManager, int maxSeqLen) {
        this.bufferManager = bufferManager;
        this.cudaContext = bufferManager.getCudaContext();
        this.arena = Arena.ofShared();
        this.maxSeqLen = maxSeqLen;
        this.defaultStream = cudaContext.getStream();

        this.dim = config.embeddingLength();
        this.vocabSize = config.vocabSize();
        this.blockCount = config.blockCount();
        this.normEps = config.normEps();
        this.ssmInnerSize = config.ssmInnerSize();
        this.ssmStateSize = config.ssmStateSize();
        this.ssmGroupCount = config.ssmGroupCount();
        this.ssmTimeStepRank = config.ssmTimeStepRank();
        this.ssmConvKernel = config.ssmConvKernel();
        this.headDim = ssmInnerSize / ssmTimeStepRank;
        this.convChannels = ssmInnerSize + 2 * ssmGroupCount * ssmStateSize;
        this.projDim = ssmInnerSize + convChannels + ssmTimeStepRank;
        this.headCount = config.headCount();
        this.headCountKV = config.headCountKV();
        this.headSize = config.headSize();
        this.kvDim = config.kvDim();
        long maxWg = cudaContext.getDeviceInfo().maxWorkGroupSize();
        this.blockSize = Math.min(256, maxWg);

        RoPE rope = new RoPE(headSize, config.ropeDimensionCount(), maxSeqLen,
            config.ropeFreqBase(), config.ropeType(), weights.ropeFreqFactors());
        this.halfRope = rope.getRopeDimCount() / 2;
        this.ropeType = rope.getRopeType();

        long fb = Float.BYTES;

        // Layer types
        layerTypes = new int[blockCount];
        for (int i = 0; i < blockCount; i++) layerTypes[i] = config.nemotronLayerType(i);

        gpuLayerCount = countGpuLayers(weights);

        // Combined upload buffer
        long combinedBytes = dim * fb + 8;
        gpuCombined = bufferManager.createBuffer(combinedBytes);
        gpuX = gpuCombined;
        gpuTokenParams = gpuCombined + dim * fb;
        hostCombined = arena.allocate(combinedBytes, 8);
        hostX = arena.allocate(ValueLayout.JAVA_FLOAT, dim);

        // Activation buffers
        gpuXb = bufferManager.createBuffer(dim * fb);
        gpuZxBCdt = bufferManager.createBuffer(projDim * fb);
        gpuXBC = bufferManager.createBuffer(convChannels * fb);
        gpuY = bufferManager.createBuffer(ssmInnerSize * fb);
        gpuDt = bufferManager.createBuffer(ssmTimeStepRank * fb);
        gpuQ = bufferManager.createBuffer((long) headCount * headSize * fb);
        gpuK = bufferManager.createBuffer(kvDim * fb);
        gpuV = bufferManager.createBuffer(kvDim * fb);
        gpuXb2 = bufferManager.createBuffer((long) headCount * headSize * fb);
        int maxFfnDim = config.intermediateSize();
        gpuHb = bufferManager.createBuffer((long) maxFfnDim * fb);
        // Granite Hybrid: second FFN buffer (for integrated SwiGLU up output) + Q8_1 scratch for down.
        // Always allocated; costs ~10 KB for ffnDim=8192.
        gpuHb2 = bufferManager.createBuffer((long) maxFfnDim * fb);
        int hbQ8Bytes = ((maxFfnDim + 31) / 32) * 40;
        gpuHbQ8 = bufferManager.createBuffer(hbQ8Bytes);

        // RoPE tables
        gpuCosTable = uploadFloatArray(rope.getCosTable());
        gpuSinTable = uploadFloatArray(rope.getSinTable());

        // Persistent state
        int histSize = ssmConvKernel - 1;
        gpuSsmState = new long[blockCount];
        gpuConvState = new long[blockCount];
        gpuKeyCache = new long[blockCount];
        gpuValueCache = new long[blockCount];
        for (int i = 0; i < gpuLayerCount; i++) {
            if (layerTypes[i] == 0) { // Mamba
                long stateBytes = (long) ssmTimeStepRank * headDim * ssmStateSize * fb;
                gpuSsmState[i] = bufferManager.createBuffer(stateBytes);
                cudaContext.fillBufferZero(gpuSsmState[i], stateBytes);
                long convBytes = (long) histSize * convChannels * fb;
                gpuConvState[i] = bufferManager.createBuffer(convBytes);
                cudaContext.fillBufferZero(gpuConvState[i], convBytes);
            } else if (layerTypes[i] == 1) { // Attention
                long kvBytes = (long) maxSeqLen * kvDim * fb;
                gpuKeyCache[i] = bufferManager.createBuffer(kvBytes);
                gpuValueCache[i] = bufferManager.createBuffer(kvBytes);
                cudaContext.fillBufferZero(gpuKeyCache[i], kvBytes);
                cudaContext.fillBufferZero(gpuValueCache[i], kvBytes);
            }
        }

        // Compile kernels
        rmsnormFunc = cudaContext.compileKernel("kernels/cuda/rmsnorm.cu", "rmsnorm_fused");
        siluFunc = cudaContext.compileKernel("kernels/cuda/silu.cu", "silu");
        siluMulFunc = cudaContext.compileKernel("kernels/cuda/silu_mul.cu", "silu_mul");
        ropeFunc = cudaContext.compileKernel("kernels/cuda/rope.cu", "rope_apply");
        kvUpdateFunc = cudaContext.compileKernel("kernels/cuda/attention.cu", "kv_cache_update");
        attnFunc = cudaContext.compileKernel("kernels/cuda/attention.cu", "attention_full");
        accumulateFunc = cudaContext.compileKernel("kernels/cuda/accumulate.cu", "accumulate");
        conv1dSiluFunc = cudaContext.compileKernel("kernels/cuda/conv1d_short.cu", "conv1d_short");
        mamba2ScanFunc = cudaContext.compileKernel("kernels/cuda/mamba2_scan.cu", "mamba2_scan");
        dtSoftplusFunc = cudaContext.compileKernel("kernels/cuda/mamba2_dt_softplus.cu", "mamba2_dt_softplus");
        gateNormFunc = cudaContext.compileKernel("kernels/cuda/mamba2_gate_norm.cu", "mamba2_gate_norm");
        sqreluFunc = cudaContext.compileKernel("kernels/cuda/sqrelu.cu", "sqrelu");

        // Upload per-layer weights
        gpuAttnNormWeights = new long[blockCount];
        gpuConvWeights = new long[blockCount];
        gpuConvBias = new long[blockCount];
        gpuDtBias = new long[blockCount];
        gpuSsmA = new long[blockCount];
        gpuSsmD = new long[blockCount];
        gpuSsmNormWeights = new long[blockCount];
        for (int i = 0; i < gpuLayerCount; i++) {
            NemotronHLayerWeights lw = weights.layers()[i];
            gpuAttnNormWeights[i] = uploadNormWeights(lw.attnNorm(), dim);
            if (layerTypes[i] == 0) { // Mamba
                gpuConvWeights[i] = uploadTensorAsFloats(lw.ssmConv1d(), convChannels * ssmConvKernel);
                gpuConvBias[i] = uploadTensorAsFloats(lw.ssmConv1dBias(), convChannels);
                gpuDtBias[i] = uploadTensorAsFloats(lw.ssmDtBias(), ssmTimeStepRank);
                gpuSsmA[i] = uploadTensorAsFloats(lw.ssmA(), ssmTimeStepRank);
                gpuSsmD[i] = uploadTensorAsFloats(lw.ssmD(), ssmTimeStepRank);
                gpuSsmNormWeights[i] = uploadTensorAsFloats(lw.ssmNorm(), ssmInnerSize);
            }
        }

        // Output
        gpuOutputNormWeights = uploadNormWeights(weights.outputNorm(), dim);
        FloatTensor outTensor = weights.output();
        if (outTensor instanceof CudaFloatTensor) {
            gpuLogits = bufferManager.createBuffer((long) vocabSize * fb);
            gpuLogitsBytes = (long) vocabSize * fb;
            hostLogits = arena.allocate(ValueLayout.JAVA_FLOAT, vocabSize);
            outputMatmul = new MatmulLaunch((CudaFloatTensor) outTensor, gpuXb, gpuLogits, vocabSize, dim, 0);
        } else {
            gpuLogits = 0; gpuLogitsBytes = 0; hostLogits = null; outputMatmul = null;
        }

        // ParamBuffers
        matmulPB = new ParamBuffer(arena, 6);

        // === dp4a setup (mirrors CudaForwardPass extension) ===
        boolean dp4aReq = !"false".equals(System.getProperty("cuda.dp4a", "true"));
        MemorySegment qFunc = null, dQ4kFunc = null, dQ5kFunc = null, dQ6kFunc = null;
        MemorySegment dQ50Func = null, dQ80Func = null, dIq4nlFunc = null, dIq4xsFunc = null;
        boolean dp4aAvail = false;
        if (dp4aReq) {
            try {
                qFunc = cudaContext.compileKernel("kernels/cuda/quantize_q8.cu", "quantize_q8");
                dQ4kFunc = cudaContext.compileKernel("kernels/cuda/matmul_q4_k_dp4a.cu", "matmul_q4_k_dp4a");
                dp4aAvail = true;
                try { dQ5kFunc = cudaContext.compileKernel("kernels/cuda/matmul_q5_k_dp4a.cu", "matmul_q5_k_dp4a"); } catch (Exception ignored) {}
                try { dQ6kFunc = cudaContext.compileKernel("kernels/cuda/matmul_q6_k_dp4a.cu", "matmul_q6_k_dp4a"); } catch (Exception ignored) {}
                try { dQ50Func = cudaContext.compileKernel("kernels/cuda/matmul_q5_0_dp4a.cu", "matmul_q5_0_dp4a"); } catch (Exception ignored) {}
                try { dQ80Func = cudaContext.compileKernel("kernels/cuda/matmul_q8_0_dp4a.cu", "matmul_q8_0_dp4a"); } catch (Exception ignored) {}
                try { dIq4nlFunc = cudaContext.compileKernel("kernels/cuda/matmul_iq4_nl_dp4a.cu", "matmul_iq4_nl_dp4a"); } catch (Exception ignored) {}
                try { dIq4xsFunc = cudaContext.compileKernel("kernels/cuda/matmul_iq4_xs_dp4a.cu", "matmul_iq4_xs_dp4a"); } catch (Exception ignored) {}
            } catch (Exception e) {
                System.err.println("NemotronH CUDA: dp4a kernels unavailable — " + e.getMessage());
            }
        }
        useDp4a = dp4aAvail;
        quantizeFunc = qFunc;
        dp4aQ4kFunc = dQ4kFunc;
        dp4aQ5kFunc = dQ5kFunc;
        dp4aQ6kFunc = dQ6kFunc;
        dp4aQ50Func = dQ50Func;
        dp4aQ80Func = dQ80Func;
        dp4aIq4nlFunc = dIq4nlFunc;
        dp4aIq4xsFunc = dIq4xsFunc;
        if (useDp4a) {
            int xbQ8Bytes = ((dim + 31) / 32) * 40;
            gpuXbQ8 = bufferManager.createBuffer(xbQ8Bytes);
            int xbBlocks = (dim + 31) / 32;
            quantizeXbGridDim = (xbBlocks + 7) / 8;
            quantizeXbPB = new ParamBuffer(arena, 3);
            quantizeXbPB.setLong(0, gpuXb);
            quantizeXbPB.setLong(1, gpuXbQ8);
            quantizeXbPB.setInt(2, dim);
            dp4aPB = new ParamBuffer(arena, 6);
            System.err.println("NemotronH CUDA: dp4a enabled");
        } else {
            gpuXbQ8 = 0;
            quantizeXbGridDim = 0;
            quantizeXbPB = null;
            dp4aPB = null;
        }

        // === Granite Hybrid scaling ===
        graniteEmbeddingScale = config.embeddingScale();
        graniteLogitScale = config.logitScale();
        graniteResidualScale = config.residualScale();
        graniteAttentionScale = config.attentionScale();
        boolean graniteActive = graniteEmbeddingScale > 0f || graniteLogitScale > 0f
                             || graniteResidualScale > 0f || graniteAttentionScale > 0f;
        MemorySegment sFunc = null;
        MemorySegment accumFunc = null;
        ParamBuffer sPB = null;
        ParamBuffer bPB = null;
        if (graniteActive) {
            try {
                sFunc = cudaContext.compileKernel("kernels/cuda/scale_inplace.cu", "scale_inplace");
                accumFunc = cudaContext.compileKernel("kernels/cuda/accumulate.cu", "accumulate");
                sPB = new ParamBuffer(arena, 3);
                bPB = new ParamBuffer(arena, 3);
                System.err.println("NemotronH CUDA: Granite scaling enabled (embed="
                    + graniteEmbeddingScale + ", logit=" + graniteLogitScale
                    + ", res=" + graniteResidualScale + ", attn=" + graniteAttentionScale + ")");
            } catch (Exception e) {
                System.err.println("NemotronH CUDA: Granite scale kernels unavailable: " + e.getMessage());
            }
        }
        scaleFunc = sFunc;
        accumulateFunc2 = accumFunc;
        scalePB = sPB;
        biasPB2 = bPB;
        scaleDimGridDim = (sFunc != null) ? (int) ((dim + blockSize - 1) / blockSize) : 0;
        scaleVocabGridDim = (sFunc != null) ? (int) ((vocabSize + blockSize - 1) / blockSize) : 0;

        normPB = new ParamBuffer(arena, 5);
        normPB.setLong(0, gpuXb); normPB.setLong(1, gpuX); normPB.setInt(3, dim); normPB.setFloat(4, normEps);

        ropePB = new ParamBuffer(arena, 8);
        ropePB.setLong(1, gpuCosTable); ropePB.setLong(2, gpuSinTable);
        ropePB.setInt(4, headSize); ropePB.setInt(5, halfRope);
        ropePB.setLong(6, gpuTokenParams); ropePB.setInt(7, ropeType);

        kvPB = new ParamBuffer(arena, 6);
        kvPB.setLong(2, gpuK); kvPB.setLong(3, gpuV); kvPB.setInt(4, kvDim); kvPB.setLong(5, gpuTokenParams);

        attnPB = new ParamBuffer(arena, 9);
        attnPB.setLong(0, gpuXb2); attnPB.setLong(1, gpuQ);
        attnPB.setInt(4, headCount); attnPB.setInt(5, headCountKV);
        attnPB.setInt(6, headSize); attnPB.setInt(7, kvDim); attnPB.setLong(8, gpuTokenParams);

        conv1dPB = new ParamBuffer(arena, 6);
        conv1dPB.setLong(0, gpuXBC); conv1dPB.setInt(3, convChannels);
        conv1dPB.setInt(4, ssmConvKernel); conv1dPB.setLong(5, gpuTokenParams);

        // mamba2_scan: S, x, B, C, dt, A, D, output, nheads, headDim, stateSize, ngroups
        scanPB = new ParamBuffer(arena, 12);
        int bOff = ssmInnerSize; int cOff = ssmInnerSize + ssmGroupCount * ssmStateSize;
        scanPB.setLong(1, gpuXBC);             // x = first ssmInnerSize of xBC
        scanPB.setLong(2, gpuXBC + bOff * fb); // B
        scanPB.setLong(3, gpuXBC + cOff * fb); // C
        scanPB.setLong(4, gpuDt);
        scanPB.setLong(7, gpuY);
        scanPB.setInt(8, ssmTimeStepRank); scanPB.setInt(9, headDim);
        scanPB.setInt(10, ssmStateSize); scanPB.setInt(11, ssmGroupCount);

        dtPB = new ParamBuffer(arena, 3);
        dtPB.setLong(0, gpuDt); dtPB.setInt(2, ssmTimeStepRank);

        // mamba2_gate_norm: y, z, normW, innerSize, ngroups, eps
        gateNormPB = new ParamBuffer(arena, 6);
        gateNormPB.setLong(0, gpuY);
        gateNormPB.setLong(1, gpuZxBCdt); // z = first ssmInnerSize of zxBCdt
        gateNormPB.setInt(3, ssmInnerSize); gateNormPB.setInt(4, ssmGroupCount); gateNormPB.setFloat(5, normEps);

        siluPB = new ParamBuffer(arena, 2);
        sqreluPB = new ParamBuffer(arena, 2);

        // Pre-allocated bias accumulate PB for conv1d bias
        ParamBuffer cBiasPB = new ParamBuffer(arena, 3);
        cBiasPB.setLong(0, gpuXBC); cBiasPB.setInt(2, convChannels);
        this.convBiasPB = cBiasPB;

        // Grid sizes
        this.normNumWarps = (int)(blockSize / 32);
        this.normSharedMem = (normNumWarps + 1) * Float.BYTES;
        this.convGridDim = (int)((convChannels + blockSize - 1) / blockSize);
        this.scanGridDim = ssmTimeStepRank; // one block per head
        this.siluConvGridDim = convGridDim;
        this.ropeQGridDim = (int)((headCount * halfRope + blockSize - 1) / blockSize);
        this.ropeKGridDim = (int)((headCountKV * halfRope + blockSize - 1) / blockSize);
        this.kvUpdateGridDim = (int)((kvDim + blockSize - 1) / blockSize);
        this.gateNormBlockDim = (int) Math.min(256, maxWg);
        this.gateNormSharedMem = (gateNormBlockDim / 32 + 1) * Float.BYTES;

        // Matmul descriptors
        layerMatmuls = new MatmulLaunch[gpuLayerCount][];
        for (int i = 0; i < gpuLayerCount; i++) {
            NemotronHLayerWeights lw = weights.layers()[i];
            if (layerTypes[i] == 0) { // Mamba
                // Granite residual scaling: route ssmOut to gpuXb (scratch) + saxpy later.
                long ssmOutTarget = graniteResidualScale > 0f ? gpuXb : gpuX;
                int ssmOutAddTo = graniteResidualScale > 0f ? 0 : 1;
                layerMatmuls[i] = new MatmulLaunch[] {
                    new MatmulLaunch((CudaFloatTensor) lw.ssmIn(), gpuXb, gpuZxBCdt, projDim, dim, 0, gpuXbQ8),
                    new MatmulLaunch((CudaFloatTensor) lw.ssmOut(), gpuY, ssmOutTarget, dim, ssmInnerSize, ssmOutAddTo)
                };
            } else if (layerTypes[i] == 1) { // Attention
                int qDim = headCount * headSize;
                long woTarget = graniteResidualScale > 0f ? gpuXb : gpuX;
                int woAddTo = graniteResidualScale > 0f ? 0 : 1;
                layerMatmuls[i] = new MatmulLaunch[] {
                    new MatmulLaunch((CudaFloatTensor) lw.wq(), gpuXb, gpuQ, qDim, dim, 0, gpuXbQ8),
                    new MatmulLaunch((CudaFloatTensor) lw.wk(), gpuXb, gpuK, kvDim, dim, 0, gpuXbQ8),
                    new MatmulLaunch((CudaFloatTensor) lw.wv(), gpuXb, gpuV, kvDim, dim, 0, gpuXbQ8),
                    new MatmulLaunch((CudaFloatTensor) lw.wo(), gpuXb2, woTarget, dim, qDim, woAddTo)
                };
            } else { // FFN
                int ffnDim = config.nemotronLayerFfnLength(i);
                long ffnDownTarget = graniteResidualScale > 0f ? gpuXb : gpuX;
                int ffnDownAddTo = graniteResidualScale > 0f ? 0 : 1;
                layerMatmuls[i] = new MatmulLaunch[] {
                    new MatmulLaunch((CudaFloatTensor) lw.ffnUp(), gpuXb, gpuHb, ffnDim, dim, 0, gpuXbQ8),
                    new MatmulLaunch((CudaFloatTensor) lw.ffnDown(), gpuHb, ffnDownTarget, dim, ffnDim, ffnDownAddTo)
                };
            }
        }

        // === Granite Hybrid integrated FFN (inside Mamba/Attention layers) ===
        // When lw.ffnUp() != null on a Mamba/Attention layer, we need an additional SwiGLU
        // block after the residual add. Build its matmul descriptors here (output dim = dim).
        integratedFfnMatmuls = new MatmulLaunch[gpuLayerCount][];
        gpuIntegratedFfnNormWeights = new long[gpuLayerCount];
        for (int i = 0; i < gpuLayerCount; i++) {
            NemotronHLayerWeights lw = weights.layers()[i];
            if ((lw.isMamba() || lw.isAttention()) && lw.ffnUp() != null) {
                int ffnDim = ((CudaFloatTensor) lw.ffnUp()).type() == null ? config.intermediateSize()
                                                                           : config.intermediateSize();
                // ffn_norm may be null — fall back to attn_norm (matches CPU engine behavior).
                FloatTensor ffnNormT = lw.ffnNorm() != null ? lw.ffnNorm() : lw.attnNorm();
                gpuIntegratedFfnNormWeights[i] = uploadNormWeights(ffnNormT, dim);
                // Integrated FFN writes to gpuXb (which we'll saxpy into gpuX) when residual scaling is active,
                // or straight to gpuX (accumulate) otherwise.
                long intFfnDownTarget = graniteResidualScale > 0f ? gpuXb : gpuX;
                int intFfnDownAddTo = graniteResidualScale > 0f ? 0 : 1;
                integratedFfnMatmuls[i] = new MatmulLaunch[] {
                    new MatmulLaunch((CudaFloatTensor) lw.ffnGate(), gpuXb, gpuHb,  ffnDim, dim, 0, gpuXbQ8),
                    new MatmulLaunch((CudaFloatTensor) lw.ffnUp(),   gpuXb, gpuHb2, ffnDim, dim, 0, gpuXbQ8),
                    new MatmulLaunch((CudaFloatTensor) lw.ffnDown(), gpuHb, intFfnDownTarget, dim, ffnDim, intFfnDownAddTo, gpuHbQ8)
                };
            } else {
                integratedFfnMatmuls[i] = null;
                gpuIntegratedFfnNormWeights[i] = 0;
            }
        }

        // CUDA graph
        graphAttnSharedMem = (maxSeqLen + 32) * Float.BYTES;
        // CUDA graph: Mamba layers use DtoD copies (split zxBCdt into xBC and dt). These go through
        // cuMemcpyDtoDAsync on the captured stream, which IS recordable as a memcpy node. The
        // previous "disabled" comment was conservative — in practice graph capture works. We can
        // still force per-layer mode via -Dcuda.nograph=true if needed.
        boolean graphDisabled = "true".equals(System.getProperty("cuda.nograph"));
        graphAvailable = !graphDisabled;

        System.err.println("NemotronH CUDA: " + gpuLayerCount + "/" + blockCount + " layers on GPU"
                + " (graph: " + (graphAvailable ? "available" : "unavailable") + ")");
    }

    // === Public API ===

    public static boolean isSupported(ModelConfig config, NemotronHWeights weights) {
        if (weights.layers().length == 0) return false;
        // Granite Hybrid fully supported as of 2026-04-14:
        //   - scale factors (embed/logit/residual/attention)
        //   - integrated SwiGLU FFN inside Mamba/Attention layers
        // Verify weight tensors are all on GPU (CudaFloatTensor).
        for (int i = 0; i < Math.min(3, weights.layers().length); i++) {
            NemotronHLayerWeights lw = weights.layers()[i];
            FloatTensor t = lw.isMamba() ? lw.ssmIn() : lw.isAttention() ? lw.wq() : lw.ffnUp();
            if (!(t instanceof CudaFloatTensor)) return false;
            // If integrated FFN is used, its tensors must also be on GPU
            if ((lw.isMamba() || lw.isAttention()) && lw.ffnUp() != null) {
                if (!(lw.ffnGate() instanceof CudaFloatTensor)) return false;
                if (!(lw.ffnUp() instanceof CudaFloatTensor)) return false;
                if (!(lw.ffnDown() instanceof CudaFloatTensor)) return false;
            }
        }
        return true;
    }

    public static int countGpuLayers(NemotronHWeights weights) {
        int count = 0;
        for (NemotronHLayerWeights lw : weights.layers()) {
            FloatTensor t = lw.isMamba() ? lw.ssmIn() : lw.isAttention() ? lw.wq() : lw.ffnUp();
            if (t instanceof CudaFloatTensor) count++; else break;
        }
        return count;
    }

    public int getGpuLayerCount() { return gpuLayerCount; }

    public void uploadXAndUpdateParams(float[] x, int position) {
        // Note: Granite Hybrid embeddingScale is applied by NemotronHInferenceEngine
        // on the CPU side BEFORE calling uploadX (see NemotronHInferenceEngine.forward).
        // Doing it again here would double-scale. Only logit scaling happens GPU-side.
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

    public void forwardLayer(int layerIdx, int position) {
        if (layerTypes[layerIdx] == 0) forwardMamba(layerIdx);
        else if (layerTypes[layerIdx] == 1) forwardAttention(layerIdx, position);
        else forwardFFN(layerIdx);
    }

    public boolean forwardGraph(float[] logits) {
        if (!graphAvailable) return false;
        if (graphExec == null) {
            boolean capturing = false;
            try {
                cudaContext.beginCapture(); capturing = true;
                for (int i = 0; i < gpuLayerCount; i++) forwardLayerKernels(i);
                normPB.setLong(2, gpuOutputNormWeights);
                launchKernel(rmsnormFunc, 1, (int) blockSize, normSharedMem, normPB.ptrs);
                launchMatmul(outputMatmul);
                MemorySegment graph = cudaContext.endCapture(); capturing = false;
                graphExec = cudaContext.instantiateGraph(graph);
                cudaContext.destroyGraph(graph);
                System.err.println("NemotronH CUDA graph: captured " + gpuLayerCount + " layers");
            } catch (Exception e) {
                if (capturing) try { cudaContext.endCapture(); } catch (Exception ignored) {}
                System.err.println("NemotronH CUDA graph failed: " + e.getMessage());
                return false;
            }
        }
        cudaContext.launchGraph(graphExec);
        // Granite logit scaling (NOT in graph — applied after)
        if (graniteLogitScale > 0f && scaleFunc != null) {
            scalePB.setLong(0, gpuLogits);
            scalePB.setFloat(1, 1.0f / graniteLogitScale);
            scalePB.setInt(2, vocabSize);
            launchKernel(scaleFunc, scaleVocabGridDim, (int) blockSize, 0, scalePB.ptrs);
        }
        cudaContext.readBuffer(gpuLogits, hostLogits, gpuLogitsBytes);
        MemorySegment.copy(hostLogits, ValueLayout.JAVA_FLOAT, 0, logits, 0, vocabSize);
        return true;
    }

    public boolean forwardFinalLogits(float[] logits) {
        if (outputMatmul == null) return false;
        normPB.setLong(2, gpuOutputNormWeights);
        launchKernel(rmsnormFunc, 1, (int) blockSize, normSharedMem, normPB.ptrs);
        launchMatmul(outputMatmul);
        if (graniteLogitScale > 0f && scaleFunc != null) {
            scalePB.setLong(0, gpuLogits);
            scalePB.setFloat(1, 1.0f / graniteLogitScale);
            scalePB.setInt(2, vocabSize);
            launchKernel(scaleFunc, scaleVocabGridDim, (int) blockSize, 0, scalePB.ptrs);
        }
        cudaContext.readBuffer(gpuLogits, hostLogits, gpuLogitsBytes);
        MemorySegment.copy(hostLogits, ValueLayout.JAVA_FLOAT, 0, logits, 0, vocabSize);
        return true;
    }

    // === Mamba-2 layer ===

    private void forwardMamba(int li) {
        MatmulLaunch[] ml = layerMatmuls[li];
        // RMSNorm
        normPB.setLong(2, gpuAttnNormWeights[li]);
        launchKernel(rmsnormFunc, 1, (int) blockSize, normSharedMem, normPB.ptrs);
        // For Granite: gpuXb is needed at end for residual saxpy; ssmOut writes to gpuXb.
        // quantizeXb() happens BEFORE ssmIn consumes gpuXb.
        quantizeXb();
        launchMatmulDp4a(ml[0]);
        // Copy xBC portion from zxBCdt to gpuXBC
        cudaContext.copyBufferDtoD(gpuXBC, gpuZxBCdt + (long) ssmInnerSize * Float.BYTES,
                (long) convChannels * Float.BYTES);
        // Copy dt portion to gpuDt
        cudaContext.copyBufferDtoD(gpuDt, gpuZxBCdt + (long)(ssmInnerSize + convChannels) * Float.BYTES,
                (long) ssmTimeStepRank * Float.BYTES);
        // Conv1d (no SiLU) → add bias → SiLU (bias must be before SiLU)
        conv1dPB.setLong(1, gpuConvState[li]); conv1dPB.setLong(2, gpuConvWeights[li]);
        launchKernel(conv1dSiluFunc, convGridDim, (int) blockSize, 0, conv1dPB.ptrs);
        // Add conv1d bias
        convBiasPB.setLong(1, gpuConvBias[li]);
        launchKernel(accumulateFunc, convGridDim, (int) blockSize, 0, convBiasPB.ptrs);
        // SiLU activation
        siluPB.setLong(0, gpuXBC); siluPB.setInt(1, convChannels);
        launchKernel(siluFunc, siluConvGridDim, (int) blockSize, 0, siluPB.ptrs);
        // dt softplus
        dtPB.setLong(1, gpuDtBias[li]);
        launchKernel(dtSoftplusFunc, 1, (int) Math.max(32, ssmTimeStepRank), 0, dtPB.ptrs);
        // Mamba-2 scan
        scanPB.setLong(0, gpuSsmState[li]); scanPB.setLong(5, gpuSsmA[li]); scanPB.setLong(6, gpuSsmD[li]);
        launchKernel(mamba2ScanFunc, scanGridDim, headDim, 0, scanPB.ptrs);
        // Gate + grouped RMSNorm (fused)
        gateNormPB.setLong(2, gpuSsmNormWeights[li]);
        launchKernel(gateNormFunc, ssmGroupCount, gateNormBlockDim, gateNormSharedMem, gateNormPB.ptrs);
        // ssm_out projection. Standard: accumulate to gpuX. Granite: write to gpuXb + saxpy.
        launchMatmul(ml[1]);
        graniteResidualAdd();
        // Granite Hybrid: Mamba layer has integrated SwiGLU FFN after the residual add.
        runIntegratedFFN(li);
    }

    // === Attention layer ===

    private void forwardAttention(int li, int position) {
        MatmulLaunch[] ml = layerMatmuls[li];
        normPB.setLong(2, gpuAttnNormWeights[li]);
        launchKernel(rmsnormFunc, 1, (int) blockSize, normSharedMem, normPB.ptrs);
        // dp4a: quantize gpuXb before QKV
        quantizeXb();
        launchMatmulDp4a(ml[0]); launchMatmulDp4a(ml[1]); launchMatmulDp4a(ml[2]); // Q K V
        // Granite attention-scale correction on Q (replaces kernel's 1/sqrt(headSize)).
        graniteAttnPreScale(ml[0].rows);
        ropePB.setLong(0, gpuQ); ropePB.setInt(3, headCount);
        launchKernel(ropeFunc, ropeQGridDim, (int) blockSize, 0, ropePB.ptrs);
        ropePB.setLong(0, gpuK); ropePB.setInt(3, headCountKV);
        launchKernel(ropeFunc, ropeKGridDim, (int) blockSize, 0, ropePB.ptrs);
        kvPB.setLong(0, gpuKeyCache[li]); kvPB.setLong(1, gpuValueCache[li]);
        launchKernel(kvUpdateFunc, kvUpdateGridDim, (int) blockSize, 0, kvPB.ptrs);
        attnPB.setLong(2, gpuKeyCache[li]); attnPB.setLong(3, gpuValueCache[li]);
        int attnSM = (position + 1 + 32) * Float.BYTES;
        launchKernel(attnFunc, headCount, Math.min(256, (int) blockSize), attnSM, attnPB.ptrs);
        launchMatmul(ml[3]); // wo — standard: accumulate; Granite: write to gpuXb + saxpy
        graniteResidualAdd();
        // Granite Hybrid: Attention layer has integrated SwiGLU FFN after the residual add.
        runIntegratedFFN(li);
    }

    // === FFN layer ===

    private void forwardFFN(int li) {
        MatmulLaunch[] ml = layerMatmuls[li];
        normPB.setLong(2, gpuAttnNormWeights[li]);
        launchKernel(rmsnormFunc, 1, (int) blockSize, normSharedMem, normPB.ptrs);
        // dp4a: quantize gpuXb before ffn_up
        quantizeXb();
        launchMatmulDp4a(ml[0]); // ffn_up
        int ffnDim = ml[0].rows;
        sqreluPB.setLong(0, gpuHb); sqreluPB.setInt(1, ffnDim);
        launchKernel(sqreluFunc, (int)((ffnDim + blockSize - 1) / blockSize), (int) blockSize, 0, sqreluPB.ptrs);
        launchMatmul(ml[1]); // ffn_down — standard: accumulate; Granite: write to gpuXb + saxpy
        graniteResidualAdd();
    }

    // === Graph capture (fixed shared mem for attention) ===

    private void forwardLayerKernels(int li) {
        if (layerTypes[li] == 0) forwardMamba(li);
        else if (layerTypes[li] == 1) {
            MatmulLaunch[] ml = layerMatmuls[li];
            normPB.setLong(2, gpuAttnNormWeights[li]);
            launchKernel(rmsnormFunc, 1, (int) blockSize, normSharedMem, normPB.ptrs);
            quantizeXb();
            launchMatmulDp4a(ml[0]); launchMatmulDp4a(ml[1]); launchMatmulDp4a(ml[2]);
            graniteAttnPreScale(ml[0].rows);
            ropePB.setLong(0, gpuQ); ropePB.setInt(3, headCount);
            launchKernel(ropeFunc, ropeQGridDim, (int) blockSize, 0, ropePB.ptrs);
            ropePB.setLong(0, gpuK); ropePB.setInt(3, headCountKV);
            launchKernel(ropeFunc, ropeKGridDim, (int) blockSize, 0, ropePB.ptrs);
            kvPB.setLong(0, gpuKeyCache[li]); kvPB.setLong(1, gpuValueCache[li]);
            launchKernel(kvUpdateFunc, kvUpdateGridDim, (int) blockSize, 0, kvPB.ptrs);
            attnPB.setLong(2, gpuKeyCache[li]); attnPB.setLong(3, gpuValueCache[li]);
            launchKernel(attnFunc, headCount, Math.min(256, (int) blockSize), graphAttnSharedMem, attnPB.ptrs);
            launchMatmul(ml[3]);
            graniteResidualAdd();
            runIntegratedFFN(li);
        } else forwardFFN(li);
    }

    // === Helpers ===

    private void launchMatmul(MatmulLaunch ml) {
        matmulPB.setLong(0, ml.weightPtr); matmulPB.setLong(1, ml.inputPtr);
        matmulPB.setLong(2, ml.outputPtr); matmulPB.setInt(3, ml.rows);
        matmulPB.setInt(4, ml.cols); matmulPB.setInt(5, ml.addToOutput);
        launchKernel(ml.function, ml.gridDim, ml.blockDim, ml.sharedMem, matmulPB.ptrs);
    }

    /** Quantize gpuXb (FP32, dim) → gpuXbQ8 (Q8_1). No-op if dp4a disabled. */
    private void quantizeXb() {
        if (!useDp4a) return;
        launchKernel(quantizeFunc, quantizeXbGridDim, 256, 0, quantizeXbPB.ptrs);
    }

    /** Quantize gpuHb (FP32, ffnDim) → gpuHbQ8 (Q8_1). No-op if dp4a disabled. */
    private void quantizeHb(int ffnDim) {
        if (!useDp4a) return;
        // Temporarily reuse quantizeXbPB layout — set 3 params, launch.
        // Actually use a dedicated buffer layout since dimensions differ.
        quantizeXbPB.setLong(0, gpuHb);
        quantizeXbPB.setLong(1, gpuHbQ8);
        quantizeXbPB.setInt(2, ffnDim);
        int gridDim = (((ffnDim + 31) / 32) + 7) / 8;
        launchKernel(quantizeFunc, gridDim, 256, 0, quantizeXbPB.ptrs);
        // Restore for next gpuXb quantize call
        quantizeXbPB.setLong(0, gpuXb);
        quantizeXbPB.setLong(1, gpuXbQ8);
        quantizeXbPB.setInt(2, dim);
    }

    /**
     * Granite Hybrid integrated SwiGLU FFN:
     *   ffn_norm(gpuX) → gpuXb
     *   quantize gpuXb → gpuXbQ8
     *   ffnGate(gpuXbQ8) → gpuHb
     *   ffnUp(gpuXbQ8)   → gpuHb2
     *   silu_mul(gpuHb, gpuHb2) → gpuHb
     *   quantize gpuHb → gpuHbQ8
     *   ffnDown(gpuHbQ8) → gpuXb (or gpuX when no residual scaling)
     *   [if residual scale] saxpy: gpuX += residualScale * gpuXb
     */
    private void runIntegratedFFN(int layerIdx) {
        MatmulLaunch[] iff = integratedFfnMatmuls[layerIdx];
        if (iff == null) return;
        int ffnDim = iff[0].rows;
        // RMSNorm with the integrated ffn norm weights
        normPB.setLong(2, gpuIntegratedFfnNormWeights[layerIdx]);
        launchKernel(rmsnormFunc, 1, (int) blockSize, normSharedMem, normPB.ptrs);
        // dp4a quantize
        quantizeXb();
        // gate + up
        launchMatmulDp4a(iff[0]); // gate → gpuHb
        launchMatmulDp4a(iff[1]); // up → gpuHb2
        // silu_mul: gpuHb = silu(gpuHb) * gpuHb2
        scalePB.setLong(0, gpuHb);        // reuse scalePB layout: a, b, size
        scalePB.setLong(1, gpuHb2);
        scalePB.setInt(2, ffnDim);
        int siluMulGridDim = (int) ((ffnDim + blockSize - 1) / blockSize);
        launchKernel(siluMulFunc, siluMulGridDim, (int) blockSize, 0, scalePB.ptrs);
        // quantize gpuHb for down projection
        quantizeHb(ffnDim);
        // down projection
        launchMatmulDp4a(iff[2]);
        // residual
        graniteResidualAdd();
    }

    /**
     * Granite residual scaling: gpuX += residualScale * gpuXb.
     * Called after ssmOut/wo/ffnDown when graniteResidualScale > 0.
     * Uses scale_inplace (scales gpuXb) + accumulate (gpuX += gpuXb).
     */
    private void graniteResidualAdd() {
        if (graniteResidualScale <= 0f) return;
        scalePB.setLong(0, gpuXb);
        scalePB.setFloat(1, graniteResidualScale);
        scalePB.setInt(2, dim);
        launchKernel(scaleFunc, scaleDimGridDim, (int) blockSize, 0, scalePB.ptrs);
        biasPB2.setLong(0, gpuX);
        biasPB2.setLong(1, gpuXb);
        biasPB2.setInt(2, dim);
        launchKernel(accumulateFunc2, scaleDimGridDim, (int) blockSize, 0, biasPB2.ptrs);
    }

    /** Granite attention pre-scale: multiply Q by (attentionScale * sqrt(headSize)) so the
     *  in-kernel 1/sqrt(headSize) factor cancels and only attentionScale remains. */
    private void graniteAttnPreScale(int qDim) {
        if (graniteAttentionScale <= 0f) return;
        scalePB.setLong(0, gpuQ);
        scalePB.setFloat(1, graniteAttentionScale * (float) Math.sqrt(headSize));
        scalePB.setInt(2, qDim);
        int qGridDim = (int) ((qDim + blockSize - 1) / blockSize);
        launchKernel(scaleFunc, qGridDim, (int) blockSize, 0, scalePB.ptrs);
    }

    /** Dispatch matmul through dp4a kernel if eligible, else FP32 fallback. */
    private void launchMatmulDp4a(MatmulLaunch ml) {
        if (!useDp4a || ml.dp4aType == 0 || ml.q8InputPtr == 0) {
            launchMatmul(ml);
            return;
        }
        MemorySegment func;
        switch (ml.dp4aType) {
            case 4:  func = dp4aQ4kFunc; break;
            case 5:  func = dp4aQ5kFunc; break;
            case 6:  // Q6_K dp4a slower than FP32 — opt-in only
                if (!"true".equals(System.getProperty("cuda.dp4a.q6", "false"))) { launchMatmul(ml); return; }
                func = dp4aQ6kFunc; break;
            case 50: func = dp4aQ50Func; break;
            case 80: func = dp4aQ80Func; break;
            case 41: func = dp4aIq4nlFunc; break;
            case 42: func = dp4aIq4xsFunc; break;
            default: launchMatmul(ml); return;
        }
        if (func == null) { launchMatmul(ml); return; }
        dp4aPB.setLong(0, ml.weightPtr);
        dp4aPB.setLong(1, ml.q8InputPtr);
        dp4aPB.setLong(2, ml.outputPtr);
        dp4aPB.setInt(3, ml.rows);
        dp4aPB.setInt(4, ml.cols);
        dp4aPB.setInt(5, ml.addToOutput);
        launchKernel(func, ml.gridDim, ml.blockDim, 0, dp4aPB.ptrs);
    }

    private void launchKernel(MemorySegment fn, int grid, int block, int sm, MemorySegment params) {
        int err = CudaBindings.launchKernel(fn, grid, 1, 1, block, 1, 1, sm, defaultStream, params, MemorySegment.NULL);
        if (err != CudaBindings.CUDA_SUCCESS) throw new RuntimeException("NemotronH CUDA error: " + err);
    }

    private long uploadNormWeights(FloatTensor t, int size) {
        float[] w = new float[size]; for (int i = 0; i < size; i++) w[i] = t.getFloat(i);
        return bufferManager.uploadNormWeights(w);
    }

    private long uploadTensorAsFloats(FloatTensor t, int size) {
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
    public void close() {
        if (graphExec != null) try { cudaContext.destroyGraphExec(graphExec); } catch (Exception ignored) {}
        arena.close();
    }
}
