package it.denzosoft.llmplayer;

import it.denzosoft.llmplayer.gguf.*;
import it.denzosoft.llmplayer.model.*;
import it.denzosoft.llmplayer.tensor.*;
import it.denzosoft.llmplayer.tokenizer.*;

import it.denzosoft.llmplayer.inference.*;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;

public class DiagnosticRunner {
    public static void main(String[] args) throws Exception {
        Path path = Paths.get("gguf/Llama-3.2-3B-Instruct-Q3_K_L.gguf");
        GGUFFile gguf = GGUFParser.parse(path);

        System.out.println("=== Tensor Types ===");
        Map<String, GGMLType> typeMap = new LinkedHashMap<>();
        for (GGUFTensorInfo t : gguf.getTensorInfos()) {
            typeMap.put(t.name(), t.type());
            if (t.name().contains("blk.0") || t.name().equals("token_embd.weight") ||
                t.name().equals("output.weight") || t.name().equals("output_norm.weight")) {
                System.out.println("  " + t);
            }
        }

        // Check embedding
        GGUFTensorInfo embInfo = gguf.findTensor("token_embd.weight");
        FloatTensor emb = TensorFactory.create(embInfo.type(), gguf.getTensorData(embInfo), embInfo.elementCount());
        System.out.println("\n=== Token Embedding ===");
        System.out.println("Type: " + emb.type() + ", size: " + emb.size());
        // Print first 10 values of token 0
        System.out.print("Token 0 embedding[0:10]: ");
        for (int i = 0; i < 10; i++) {
            System.out.printf("%.6f ", emb.getFloat(i));
        }
        System.out.println();
        // Token 1
        System.out.print("Token 1 embedding[0:10]: ");
        int dim = 3072;
        for (int i = 0; i < 10; i++) {
            System.out.printf("%.6f ", emb.getFloat(dim + i));
        }
        System.out.println();

        // Check tokenizer
        System.out.println("\n=== Tokenizer ===");
        String tokModel = gguf.getMetadata().getString("tokenizer.ggml.model", "unknown");
        System.out.println("Tokenizer model: " + tokModel);

        Tokenizer tok = TokenizerFactory.create(gguf.getMetadata());
        System.out.println("Vocab size: " + tok.vocabSize());

        // Encode test strings
        String[] tests = {"Hello", "Hello, world!", "The quick brown fox"};
        for (String test : tests) {
            int[] tokens = tok.encode(test);
            System.out.println("Encode('" + test + "'): " + Arrays.toString(tokens));
            String decoded = tok.decode(tokens);
            System.out.println("  Decode: '" + decoded + "'");
        }

        // Check what the chat template produces
        SpecialTokens st = SpecialTokens.fromMetadata(gguf.getMetadata());
        System.out.println("\nBOS id: " + st.getBosId() + " EOS id: " + st.getEosId());
        ChatTemplate ct = new ChatTemplate(ModelArchitecture.LLAMA,
            gguf.getMetadata().getString("tokenizer.chat_template", ""));
        String formatted = ct.formatUserMessage("Hello");
        System.out.println("Chat formatted: '" + formatted + "'");
        int[] chatTokens = tok.encode(formatted);
        System.out.println("Chat tokens: " + Arrays.toString(chatTokens));
        System.out.println("Chat decoded: '" + tok.decode(chatTokens) + "'");

        // Check norm weights
        GGUFTensorInfo normInfo = gguf.findTensor("blk.0.attn_norm.weight");
        FloatTensor norm = TensorFactory.create(normInfo.type(), gguf.getTensorData(normInfo), normInfo.elementCount());
        System.out.println("\n=== Layer 0 Norm ===");
        System.out.println("Type: " + norm.type());
        System.out.print("Values[0:10]: ");
        for (int i = 0; i < 10; i++) {
            System.out.printf("%.6f ", norm.getFloat(i));
        }
        System.out.println();

        // Check Q weight
        GGUFTensorInfo wqInfo = gguf.findTensor("blk.0.attn_q.weight");
        FloatTensor wq = TensorFactory.create(wqInfo.type(), gguf.getTensorData(wqInfo), wqInfo.elementCount());
        System.out.println("\n=== Layer 0 WQ ===");
        System.out.println("Type: " + wq.type() + " " + wqInfo);
        System.out.print("Values[0:10]: ");
        for (int i = 0; i < 10; i++) {
            System.out.printf("%.6f ", wq.getFloat(i));
        }
        System.out.println();

        // === DOT PRODUCT CONSISTENCY TEST ===
        System.out.println("\n=== Dot Product Consistency Test ===");
        // Test with a non-trivial input vector
        float[] testVec = new float[dim];
        Random rng = new Random(42);
        for (int i = 0; i < dim; i++) {
            testVec[i] = rng.nextFloat() * 2 - 1;  // [-1, 1]
        }

        // Test each tensor type used in the model
        String[] tensorNames = {
            "token_embd.weight",     // Q6_K
            "blk.0.attn_q.weight",   // Q3_K
            "blk.0.attn_k.weight",   // Q3_K
            "blk.0.attn_v.weight",   // Q5_K
            "blk.0.attn_output.weight", // Q5_K
            "blk.0.ffn_gate.weight", // Q3_K
            "blk.0.ffn_up.weight",   // Q3_K
            "blk.0.ffn_down.weight", // Q5_K
            "output.weight"          // Q6_K (or tied)
        };

        for (String name : tensorNames) {
            GGUFTensorInfo info = gguf.findTensor(name);
            if (info == null) {
                System.out.println("  " + name + ": NOT FOUND");
                continue;
            }
            FloatTensor tensor = TensorFactory.create(info.type(), gguf.getTensorData(info), info.elementCount());

            // Compute dot product via getFloat() element-by-element
            double refDot = 0;
            for (int i = 0; i < dim; i++) {
                refDot += (double) tensor.getFloat(i) * testVec[i];
            }

            // Compute dot product via dot() method
            float fastDot = tensor.dot(0, testVec, 0, dim);

            double absDiff = Math.abs(refDot - fastDot);
            double relDiff = refDot != 0 ? absDiff / Math.abs(refDot) : absDiff;
            boolean ok = relDiff < 0.01; // 1% tolerance

            System.out.printf("  %-30s [%5s] ref=%.6f fast=%.6f diff=%.2e relDiff=%.2e %s%n",
                name, info.type(), refDot, fastDot, absDiff, relDiff, ok ? "OK" : "MISMATCH!");
        }

        // Also test dot at a non-zero offset (row 1)
        System.out.println("\n=== Dot Product at Row 1 (offset=dim) ===");
        for (String name : new String[]{"token_embd.weight", "blk.0.attn_q.weight", "blk.0.attn_v.weight", "output.weight"}) {
            GGUFTensorInfo info = gguf.findTensor(name);
            if (info == null) continue;
            FloatTensor tensor = TensorFactory.create(info.type(), gguf.getTensorData(info), info.elementCount());

            double refDot = 0;
            for (int i = 0; i < dim; i++) {
                refDot += (double) tensor.getFloat(dim + i) * testVec[i];
            }
            float fastDot = tensor.dot(dim, testVec, 0, dim);

            double absDiff = Math.abs(refDot - fastDot);
            double relDiff = refDot != 0 ? absDiff / Math.abs(refDot) : absDiff;
            boolean ok = relDiff < 0.01;

            System.out.printf("  %-30s [%5s] ref=%.6f fast=%.6f diff=%.2e relDiff=%.2e %s%n",
                name, info.type(), refDot, fastDot, absDiff, relDiff, ok ? "OK" : "MISMATCH!");
        }

        // Test at row 10 (offset=10*dim) to verify block boundary handling
        System.out.println("\n=== Dot Product at Row 10 (offset=10*dim) ===");
        for (String name : new String[]{"token_embd.weight", "blk.0.attn_q.weight", "blk.0.attn_v.weight"}) {
            GGUFTensorInfo info = gguf.findTensor(name);
            if (info == null) continue;
            FloatTensor tensor = TensorFactory.create(info.type(), gguf.getTensorData(info), info.elementCount());

            long offset = 10L * dim;
            double refDot = 0;
            for (int i = 0; i < dim; i++) {
                refDot += (double) tensor.getFloat(offset + i) * testVec[i];
            }
            float fastDot = tensor.dot(offset, testVec, 0, dim);

            double absDiff = Math.abs(refDot - fastDot);
            double relDiff = refDot != 0 ? absDiff / Math.abs(refDot) : absDiff;
            boolean ok = relDiff < 0.01;

            System.out.printf("  %-30s [%5s] ref=%.6f fast=%.6f diff=%.2e relDiff=%.2e %s%n",
                name, info.type(), refDot, fastDot, absDiff, relDiff, ok ? "OK" : "MISMATCH!");
        }

        // === Q3_K DETAILED VERIFICATION ===
        System.out.println("\n=== Q3_K Detailed Verification ===");
        GGUFTensorInfo wqInfo2 = gguf.findTensor("blk.0.attn_q.weight");
        FloatTensor wqTensor = TensorFactory.create(wqInfo2.type(), gguf.getTensorData(wqInfo2), wqInfo2.elementCount());
        TensorData wqSeg = gguf.getTensorData(wqInfo2);

        // Print raw bytes of first Q3_K block
        System.out.print("Block 0 d (fp16 at offset 108): ");
        int d_raw = Short.toUnsignedInt(wqSeg.getShortLE(108));
        float d_val = Float16.toFloat((short) d_raw);
        System.out.printf("raw=0x%04X float=%.8f%n", d_raw, d_val);

        System.out.print("Block 0 hmask[0..3]: ");
        for (int bi = 0; bi < 4; bi++) {
            System.out.printf("0x%02X ", Byte.toUnsignedInt(wqSeg.getByte(bi)));
        }
        System.out.println();

        System.out.print("Block 0 qs[0..3]: ");
        for (int bi = 0; bi < 4; bi++) {
            System.out.printf("0x%02X ", Byte.toUnsignedInt(wqSeg.getByte(32 + bi)));
        }
        System.out.println();

        System.out.print("Block 0 scales raw[0..11]: ");
        for (int bi = 0; bi < 12; bi++) {
            System.out.printf("0x%02X ", Byte.toUnsignedInt(wqSeg.getByte(96 + bi)));
        }
        System.out.println();

        // Decode scales manually
        byte[] rawScales = new byte[12];
        for (int bi = 0; bi < 12; bi++) {
            rawScales[bi] = wqSeg.getByte(96 + bi);
        }
        int[] scalesManual = new int[16];
        for (int bi = 0; bi < 8; bi++) {
            int sv = Byte.toUnsignedInt(rawScales[bi]);
            scalesManual[bi] = sv & 0x0F;
            scalesManual[bi + 8] = sv >> 4;
        }
        for (int bi = 0; bi < 4; bi++) {
            int sv = Byte.toUnsignedInt(rawScales[8 + bi]);
            scalesManual[bi]     |= (sv & 0x03) << 4;
            scalesManual[bi + 4] |= ((sv >> 2) & 0x03) << 4;
            scalesManual[bi + 8] |= ((sv >> 4) & 0x03) << 4;
            scalesManual[bi + 12]|= ((sv >> 6) & 0x03) << 4;
        }
        System.out.print("Decoded scales: ");
        for (int bi = 0; bi < 16; bi++) {
            System.out.printf("%d ", scalesManual[bi]);
        }
        System.out.println();

        // Manually dequantize first 16 elements (sub-block 0)
        System.out.println("Manual dequant first 16 elements:");
        for (int j = 0; j < 16; j++) {
            // Reference: element j (0..15) -> qs[j] shift 0, hmask[j] bit 0, scale[0]
            int qsByte2 = Byte.toUnsignedInt(wqSeg.getByte(32 + j));
            int lowBits2 = qsByte2 & 3;
            int hmByte2 = Byte.toUnsignedInt(wqSeg.getByte(j));
            int hbit2 = hmByte2 & 1;
            int q2 = (lowBits2 | (hbit2 << 2)) - 4;
            float manualVal = d_val * (scalesManual[0] - 32) * q2;
            float ourVal = wqTensor.getFloat(j);
            System.out.printf("  j=%2d: qs=0x%02X low=%d hm=0x%02X hbit=%d q=%d scale=%d manual=%.8f ours=%.8f %s%n",
                j, qsByte2, lowBits2, hmByte2, hbit2, q2, scalesManual[0]-32, manualVal, ourVal,
                Math.abs(manualVal - ourVal) < 1e-10 ? "OK" : "MISMATCH");
        }

        // Also check element 32 (sub-block 2: qs[0] shift 2, hmask[0] bit 1)
        System.out.println("Manual dequant elements 32..35:");
        for (int j = 32; j < 36; j++) {
            int half = j / 128;
            int jInHalf = j % 128;
            int pair2 = jInHalf / 32;
            int jInPair = jInHalf % 32;
            int which16 = jInPair / 16;
            int l2 = jInPair % 16;

            int qsByteIdx = half * 32 + which16 * 16 + l2;
            int qsShift = pair2 * 2;
            int qsByte2 = Byte.toUnsignedInt(wqSeg.getByte(32 + qsByteIdx));
            int lowBits2 = (qsByte2 >> qsShift) & 3;
            int hmByteIdx = which16 * 16 + l2;
            int hmBit2 = half * 4 + pair2;
            int hmByte2 = Byte.toUnsignedInt(wqSeg.getByte(hmByteIdx));
            int hbit2 = (hmByte2 >> hmBit2) & 1;
            int q2 = (lowBits2 | (hbit2 << 2)) - 4;
            int subBlock = j / 16;
            float manualVal = d_val * (scalesManual[subBlock] - 32) * q2;
            float ourVal = wqTensor.getFloat(j);
            System.out.printf("  j=%2d: qsB=%d shift=%d low=%d hmB=%d hmBit=%d hbit=%d q=%d sc=%d manual=%.8f ours=%.8f %s%n",
                j, qsByteIdx, qsShift, lowBits2, hmByteIdx, hmBit2, hbit2, q2, scalesManual[subBlock]-32,
                manualVal, ourVal, Math.abs(manualVal - ourVal) < 1e-10 ? "OK" : "MISMATCH");
        }

        gguf.close();

        // === PREFILL TEST ===
        System.out.println("\n\n=== Prefill Test ===");
        ModelLoader.LoadedModel loaded = ModelLoader.load(Paths.get("gguf/Llama-3.2-3B-Instruct-Q3_K_L.gguf"));
        ModelConfig modelConfig = loaded.config();
        ModelWeights modelWeights = loaded.weights();
        Tokenizer tok2 = loaded.tokenizer();

        int maxSeqLen = 128;
        InferenceEngine engine = new InferenceEngine(modelConfig, modelWeights, maxSeqLen);

        // Test 1: Single token BOS at position 0 (RoPE = identity)
        System.out.println("\n--- Test 1: BOS token (128000) at position 0 ---");
        InferenceState state = engine.createState(maxSeqLen);
        float[] logits = engine.forward(state, 128000, 0);
        printTopTokens("After BOS", logits, modelConfig.vocabSize(), tok2, 10);
        System.out.println("  Expected: token 128006 (<|start_header_id|>) should be top-1");
        System.out.printf("  Logit for 128006: %.4f%n", logits[128006]);

        // Test 2: BOS then <|start_header_id|> at position 1 (RoPE matters!)
        System.out.println("\n--- Test 2: BOS + <|start_header_id|> (positions 0,1) ---");
        state = engine.createState(maxSeqLen);
        engine.forward(state, 128000, 0);  // BOS at pos 0
        logits = engine.forward(state, 128006, 1);  // <|start_header_id|> at pos 1
        printTopTokens("After BOS + start_header", logits, modelConfig.vocabSize(), tok2, 10);
        System.out.println("  Expected: 'system' or 'user' should be near top");
        System.out.printf("  Logit for 'system' (9125): %.4f%n", logits[9125]);
        System.out.printf("  Logit for 'user' (882): %.4f%n", logits[882]);

        // Test 3: Full chat template prefill
        System.out.println("\n--- Test 3: Full chat template prefill ---");
        int[] chatTokens2 = {128000, 128006, 882, 128007, 271, 9906, 128009, 128006, 78191, 128007, 271};
        state = engine.createState(maxSeqLen);
        logits = engine.prefill(state, chatTokens2);
        printTopTokens("After full chat prefill", logits, modelConfig.vocabSize(), tok2, 10);
        System.out.println("  (This is the model's first generated token prediction)");

        // Test 4: Position-by-position prefill to track where predictions go wrong
        System.out.println("\n--- Test 4: Step-by-step prefill ---");
        state = engine.createState(maxSeqLen);
        String[] expectedNext = {"<|start_header_id|>", "user/system", "<|end_header_id|>", "\\n\\n", "Hello", "<|eot_id|>", "<|start_header_id|>", "assistant", "<|end_header_id|>", "\\n\\n", "(first gen token)"};
        for (int i = 0; i < chatTokens2.length; i++) {
            logits = engine.forward(state, chatTokens2[i], i);
            String tokenStr;
            try { tokenStr = tok2.decode(new int[]{chatTokens2[i]}); } catch (Exception e) { tokenStr = "?"; }
            int topToken = argmax(logits, modelConfig.vocabSize());
            String topStr;
            try { topStr = tok2.decode(new int[]{topToken}); } catch (Exception e) { topStr = "?"; }
            int expectedToken = i < chatTokens2.length - 1 ? chatTokens2[i + 1] : -1;
            boolean correct = (topToken == expectedToken);
            System.out.printf("  pos=%2d token=%6d '%-25s' -> top pred=%6d '%-20s' logit=%.2f %s (expected: %s)%n",
                i, chatTokens2[i], tokenStr, topToken, topStr, logits[topToken],
                correct ? "CORRECT" : "", expectedNext[i]);
        }

        // Test 5: Manual layer 0 trace
        System.out.println("\n--- Test 5: Manual layer 0 trace for BOS ---");
        {
            int modelDim = modelConfig.embeddingLength();
            int kvDimVal = modelConfig.kvDim();
            int ffnDim = modelConfig.intermediateSize();
            int headCount = modelConfig.headCount();
            int headCountKV = modelConfig.headCountKV();
            int headSz = modelConfig.headSize();

            state = engine.createState(maxSeqLen);

            // Step 1: Embedding lookup
            for (int i = 0; i < modelDim; i++) {
                state.x[i] = modelWeights.tokenEmbedding().getFloat((long) 128000 * modelDim + i);
            }
            printStats("emb(BOS)", state.x, modelDim);

            // Step 2: Attn norm
            float[] layer0AttnNorm = new float[modelDim];
            for (int i = 0; i < modelDim; i++) {
                layer0AttnNorm[i] = modelWeights.layers()[0].attnNorm().getFloat(i);
            }
            VectorOpsFactory.get().rmsnorm(state.xb, state.x, layer0AttnNorm, modelDim, modelConfig.normEps());
            printStats("attn_norm(x)", state.xb, modelDim);

            // Step 3: V projection (this is what matters for pos=0 attention)
            float[] vProj = new float[kvDimVal];
            modelWeights.layers()[0].wv().matmulParallel(state.xb, vProj, kvDimVal, modelDim);
            printStats("V projection", vProj, kvDimVal);

            // Step 4: At pos=0, attention output = V directly. Then Wo * V.
            float[] attnOut = new float[modelDim];
            // For GQA: each Q head shares KV head. But at pos=0, the output is:
            // xb2[h*headSize .. (h+1)*headSize-1] = V[kvHead*headSize .. (kvHead+1)*headSize-1]
            // where kvHead = h / kvMul
            int kvMul = headCount / headCountKV;
            float[] xb2_manual = new float[modelDim];
            for (int h = 0; h < headCount; h++) {
                int kvHead = h / kvMul;
                for (int i = 0; i < headSz; i++) {
                    xb2_manual[h * headSz + i] = vProj[kvHead * headSz + i];
                }
            }
            printStats("xb2 (attention concat)", xb2_manual, modelDim);

            // Output projection: attnOut = Wo * xb2
            modelWeights.layers()[0].wo().matmulParallel(xb2_manual, attnOut, modelDim, modelDim);
            printStats("Wo * V (attn output)", attnOut, modelDim);

            // Step 5: Residual
            float[] x_after_attn = new float[modelDim];
            for (int i = 0; i < modelDim; i++) {
                x_after_attn[i] = state.x[i] + attnOut[i];
            }
            printStats("x + attn_out", x_after_attn, modelDim);

            // Step 6: FFN norm
            float[] layer0FfnNorm = new float[modelDim];
            for (int i = 0; i < modelDim; i++) {
                layer0FfnNorm[i] = modelWeights.layers()[0].ffnNorm().getFloat(i);
            }
            float[] ffnInput = new float[modelDim];
            VectorOpsFactory.get().rmsnorm(ffnInput, x_after_attn, layer0FfnNorm, modelDim, modelConfig.normEps());
            printStats("ffn_norm(x)", ffnInput, modelDim);

            // Step 7: SwiGLU FFN
            float[] gate = new float[ffnDim];
            float[] up = new float[ffnDim];
            modelWeights.layers()[0].wGate().matmulParallel(ffnInput, gate, ffnDim, modelDim);
            modelWeights.layers()[0].wUp().matmulParallel(ffnInput, up, ffnDim, modelDim);
            printStats("gate", gate, ffnDim);
            printStats("up", up, ffnDim);

            VectorOpsFactory.get().silu(gate, ffnDim);
            VectorOpsFactory.get().elementwiseMul(gate, up, gate, ffnDim);
            printStats("silu(gate)*up", gate, ffnDim);

            float[] ffnOut = new float[modelDim];
            modelWeights.layers()[0].wDown().matmulParallel(gate, ffnOut, modelDim, ffnDim);
            printStats("Wdown * hidden", ffnOut, modelDim);

            // Step 8: Residual after FFN
            float[] x_after_ffn = new float[modelDim];
            for (int i = 0; i < modelDim; i++) {
                x_after_ffn[i] = x_after_attn[i] + ffnOut[i];
            }
            printStats("x after layer 0", x_after_ffn, modelDim);

            // Now compare with engine's 1-layer output
            state = engine.createState(maxSeqLen);
            engine.forwardLayers(state, 128000, 0, 1);
            printStats("engine 1-layer x", state.x, modelDim);

            // Check if they match
            double maxDiff = 0;
            for (int i = 0; i < modelDim; i++) {
                double diff = Math.abs(x_after_ffn[i] - state.x[i]);
                if (diff > maxDiff) maxDiff = diff;
            }
            System.out.printf("Max diff between manual and engine layer 0: %.6e%n", maxDiff);
        }

        // Test 5b: Layer-by-layer analysis - where does corruption start?
        System.out.println("\n--- Test 5: Layer-by-layer BOS prediction ---");
        int vocabSize = modelConfig.vocabSize();
        for (int numLayers : new int[]{0, 1, 2, 4, 8, 14, 28}) {
            state = engine.createState(maxSeqLen);
            logits = engine.forwardLayers(state, 128000, 0, numLayers);
            int topToken = argmax(logits, vocabSize);
            String topStr; try { topStr = tok2.decode(new int[]{topToken}); } catch (Exception e) { topStr = "?"; }
            float logitFor128006 = logits[128006];
            System.out.printf("  Layers=%2d: top=%6d logit=%.4f '%-20s' logit[128006]=%.4f%n",
                numLayers, topToken, logits[topToken], topStr, logitFor128006);
        }

        // Test 6: With 0 layers, the output should be embedding similarity
        // logits[i] = dot(emb[i], norm(emb[BOS]))
        // The top token should be BOS itself (highest self-similarity)
        System.out.println("\n--- Test 6: 0-layer check (emb * norm(emb[BOS])) ---");
        state = engine.createState(maxSeqLen);
        logits = engine.forwardLayers(state, 128000, 0, 0);
        System.out.println("  BOS self-logit (token 128000): " + logits[128000]);
        System.out.println("  Top token: " + argmax(logits, vocabSize));

        loaded.close();
    }

    static void printTopTokens(String label, float[] logits, int vocabSize, Tokenizer tok, int n) {
        int[] topIdx = new int[n];
        float[] topVal = new float[n];
        Arrays.fill(topVal, Float.NEGATIVE_INFINITY);
        for (int i = 0; i < vocabSize; i++) {
            for (int j = 0; j < n; j++) {
                if (logits[i] > topVal[j]) {
                    for (int k = n - 1; k > j; k--) { topVal[k] = topVal[k-1]; topIdx[k] = topIdx[k-1]; }
                    topVal[j] = logits[i]; topIdx[j] = i;
                    break;
                }
            }
        }
        System.out.println(label + " top " + n + " tokens:");
        for (int i = 0; i < n; i++) {
            String s; try { s = tok.decode(new int[]{topIdx[i]}); } catch (Exception e) { s = "?"; }
            System.out.printf("  %d: token=%d logit=%.4f '%s'%n", i, topIdx[i], topVal[i], s);
        }
    }

    static int argmax(float[] arr, int len) {
        int best = 0; float bestVal = arr[0];
        for (int i = 1; i < len; i++) { if (arr[i] > bestVal) { bestVal = arr[i]; best = i; } }
        return best;
    }

    static void printVec(String name, float[] vec, int n) {
        System.out.print(name + "[0:" + n + "]: ");
        for (int i = 0; i < n; i++) {
            System.out.printf("%.6f ", vec[i]);
        }
        System.out.println();
    }

    static void printStats(String name, float[] vec, int len) {
        float min = Float.MAX_VALUE, max = -Float.MAX_VALUE;
        double sum = 0, sumSq = 0;
        int nanCount = 0, infCount = 0;
        for (int i = 0; i < len; i++) {
            if (Float.isNaN(vec[i])) nanCount++;
            if (Float.isInfinite(vec[i])) infCount++;
            if (vec[i] < min) min = vec[i];
            if (vec[i] > max) max = vec[i];
            sum += vec[i];
            sumSq += (double) vec[i] * vec[i];
        }
        double mean = sum / len;
        double std = Math.sqrt(sumSq / len - mean * mean);
        System.out.printf("  %s stats: min=%.6f max=%.6f mean=%.6f std=%.6f nan=%d inf=%d%n",
            name, min, max, mean, std, nanCount, infCount);
    }
}
