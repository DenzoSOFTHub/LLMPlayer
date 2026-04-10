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

    // Per-layer weights on GPU
    private final long[] gpuAttnNormWeights;
    private final long[] gpuConvWeights, gpuConvBias, gpuDtBias, gpuSsmA, gpuSsmD, gpuSsmNormWeights;

    // Output
    private final long gpuOutputNormWeights, gpuLogits, gpuLogitsBytes;
    private final MemorySegment hostLogits;

    // Compiled CUDA functions
    private final MemorySegment rmsnormFunc, siluFunc, ropeFunc, kvUpdateFunc, attnFunc;
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
        MatmulLaunch(CudaFloatTensor t, long in, long out, int r, int c, int add) {
            function = t.getCudaFunction(); weightPtr = t.getGpuWeights();
            blockDim = t.getMatmulBlockDim(c); gridDim = t.getMatmulGridDim(r, c);
            sharedMem = t.getMatmulSharedMem(c);
            inputPtr = in; outputPtr = out; rows = r; cols = c; addToOutput = add;
        }
    }

    private final ParamBuffer matmulPB, normPB, ropePB, kvPB, attnPB;
    private final ParamBuffer conv1dPB, scanPB, dtPB, gateNormPB, siluPB, sqreluPB, convBiasPB;

    // Per-layer matmul descriptors
    // Mamba: [0]=ssmIn, [1]=ssmOut
    // Attention: [0]=wq, [1]=wk, [2]=wv, [3]=wo
    // FFN: [0]=ffnUp, [1]=ffnDown
    private final MatmulLaunch[][] layerMatmuls;
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
        ParamBuffer bPB = new ParamBuffer(arena, 3);
        bPB.setLong(0, gpuXBC); bPB.setInt(2, convChannels);
        this.convBiasPB = bPB;

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
                layerMatmuls[i] = new MatmulLaunch[] {
                    new MatmulLaunch((CudaFloatTensor) lw.ssmIn(), gpuXb, gpuZxBCdt, projDim, dim, 0),
                    new MatmulLaunch((CudaFloatTensor) lw.ssmOut(), gpuY, gpuX, dim, ssmInnerSize, 1)
                };
            } else if (layerTypes[i] == 1) { // Attention
                int qDim = headCount * headSize;
                layerMatmuls[i] = new MatmulLaunch[] {
                    new MatmulLaunch((CudaFloatTensor) lw.wq(), gpuXb, gpuQ, qDim, dim, 0),
                    new MatmulLaunch((CudaFloatTensor) lw.wk(), gpuXb, gpuK, kvDim, dim, 0),
                    new MatmulLaunch((CudaFloatTensor) lw.wv(), gpuXb, gpuV, kvDim, dim, 0),
                    new MatmulLaunch((CudaFloatTensor) lw.wo(), gpuXb2, gpuX, dim, qDim, 1)
                };
            } else { // FFN
                int ffnDim = config.nemotronLayerFfnLength(i);
                layerMatmuls[i] = new MatmulLaunch[] {
                    new MatmulLaunch((CudaFloatTensor) lw.ffnUp(), gpuXb, gpuHb, ffnDim, dim, 0),
                    new MatmulLaunch((CudaFloatTensor) lw.ffnDown(), gpuHb, gpuX, dim, ffnDim, 1)
                };
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
        for (int i = 0; i < Math.min(3, weights.layers().length); i++) {
            NemotronHLayerWeights lw = weights.layers()[i];
            FloatTensor t = lw.isMamba() ? lw.ssmIn() : lw.isAttention() ? lw.wq() : lw.ffnUp();
            if (!(t instanceof CudaFloatTensor)) return false;
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
        cudaContext.readBuffer(gpuLogits, hostLogits, gpuLogitsBytes);
        MemorySegment.copy(hostLogits, ValueLayout.JAVA_FLOAT, 0, logits, 0, vocabSize);
        return true;
    }

    public boolean forwardFinalLogits(float[] logits) {
        if (outputMatmul == null) return false;
        normPB.setLong(2, gpuOutputNormWeights);
        launchKernel(rmsnormFunc, 1, (int) blockSize, normSharedMem, normPB.ptrs);
        launchMatmul(outputMatmul);
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
        // ssm_in projection
        launchMatmul(ml[0]);
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
        // ssm_out projection (accumulate to x)
        launchMatmul(ml[1]);
    }

    // === Attention layer ===

    private void forwardAttention(int li, int position) {
        MatmulLaunch[] ml = layerMatmuls[li];
        normPB.setLong(2, gpuAttnNormWeights[li]);
        launchKernel(rmsnormFunc, 1, (int) blockSize, normSharedMem, normPB.ptrs);
        launchMatmul(ml[0]); launchMatmul(ml[1]); launchMatmul(ml[2]); // Q K V
        ropePB.setLong(0, gpuQ); ropePB.setInt(3, headCount);
        launchKernel(ropeFunc, ropeQGridDim, (int) blockSize, 0, ropePB.ptrs);
        ropePB.setLong(0, gpuK); ropePB.setInt(3, headCountKV);
        launchKernel(ropeFunc, ropeKGridDim, (int) blockSize, 0, ropePB.ptrs);
        kvPB.setLong(0, gpuKeyCache[li]); kvPB.setLong(1, gpuValueCache[li]);
        launchKernel(kvUpdateFunc, kvUpdateGridDim, (int) blockSize, 0, kvPB.ptrs);
        attnPB.setLong(2, gpuKeyCache[li]); attnPB.setLong(3, gpuValueCache[li]);
        int attnSM = (position + 1 + 32) * Float.BYTES;
        launchKernel(attnFunc, headCount, Math.min(256, (int) blockSize), attnSM, attnPB.ptrs);
        launchMatmul(ml[3]); // wo (accumulate)
    }

    // === FFN layer ===

    private void forwardFFN(int li) {
        MatmulLaunch[] ml = layerMatmuls[li];
        normPB.setLong(2, gpuAttnNormWeights[li]);
        launchKernel(rmsnormFunc, 1, (int) blockSize, normSharedMem, normPB.ptrs);
        launchMatmul(ml[0]); // ffn_up
        int ffnDim = ml[0].rows;
        sqreluPB.setLong(0, gpuHb); sqreluPB.setInt(1, ffnDim);
        launchKernel(sqreluFunc, (int)((ffnDim + blockSize - 1) / blockSize), (int) blockSize, 0, sqreluPB.ptrs);
        launchMatmul(ml[1]); // ffn_down (accumulate)
    }

    // === Graph capture (fixed shared mem for attention) ===

    private void forwardLayerKernels(int li) {
        if (layerTypes[li] == 0) forwardMamba(li);
        else if (layerTypes[li] == 1) {
            MatmulLaunch[] ml = layerMatmuls[li];
            normPB.setLong(2, gpuAttnNormWeights[li]);
            launchKernel(rmsnormFunc, 1, (int) blockSize, normSharedMem, normPB.ptrs);
            launchMatmul(ml[0]); launchMatmul(ml[1]); launchMatmul(ml[2]);
            ropePB.setLong(0, gpuQ); ropePB.setInt(3, headCount);
            launchKernel(ropeFunc, ropeQGridDim, (int) blockSize, 0, ropePB.ptrs);
            ropePB.setLong(0, gpuK); ropePB.setInt(3, headCountKV);
            launchKernel(ropeFunc, ropeKGridDim, (int) blockSize, 0, ropePB.ptrs);
            kvPB.setLong(0, gpuKeyCache[li]); kvPB.setLong(1, gpuValueCache[li]);
            launchKernel(kvUpdateFunc, kvUpdateGridDim, (int) blockSize, 0, kvPB.ptrs);
            attnPB.setLong(2, gpuKeyCache[li]); attnPB.setLong(3, gpuValueCache[li]);
            launchKernel(attnFunc, headCount, Math.min(256, (int) blockSize), graphAttnSharedMem, attnPB.ptrs);
            launchMatmul(ml[3]);
        } else forwardFFN(li);
    }

    // === Helpers ===

    private void launchMatmul(MatmulLaunch ml) {
        matmulPB.setLong(0, ml.weightPtr); matmulPB.setLong(1, ml.inputPtr);
        matmulPB.setLong(2, ml.outputPtr); matmulPB.setInt(3, ml.rows);
        matmulPB.setInt(4, ml.cols); matmulPB.setInt(5, ml.addToOutput);
        launchKernel(ml.function, ml.gridDim, ml.blockDim, ml.sharedMem, matmulPB.ptrs);
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
