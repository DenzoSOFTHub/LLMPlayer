package it.denzosoft.llmplayer.tts;

import it.denzosoft.llmplayer.inference.InferenceEngine;
import it.denzosoft.llmplayer.inference.InferenceState;
import it.denzosoft.llmplayer.model.ModelConfig;
import it.denzosoft.llmplayer.model.ModelLoader;
import it.denzosoft.llmplayer.tensor.FloatTensor;
import it.denzosoft.llmplayer.tokenizer.Tokenizer;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * Qwen3-TTS text-to-speech in the llama.cpp GGUF layout: a talker GGUF ({@code general.architecture
 * = qwen3tts}) and its mmproj ({@code qwen3tts_gen}). Follows llama.cpp
 * {@code tools/mtmd/mtmd-helper-gen.cpp} (qwen3tts_gen_audio_pipeline):
 *
 * <ol>
 *   <li>The prompt is a stream of summed embeddings: the chat role, then
 *       {@code tts_pad + codec_think / think_bos / language / think_eos}, {@code tts_bos + codec_pad},
 *       each text token {@code + codec_pad}, {@code tts_eos + codec_pad} and {@code tts_pad + codec_bos}.</li>
 *   <li>Per 80 ms frame the talker (a Qwen3-VL-style decoder) samples the codebook-0 code from its
 *       3072-entry codec head; the code predictor samples the other 15 codebooks from the talker's
 *       hidden state; the sum of the 16 code embeddings plus {@code tts_pad} is the talker's next
 *       input. The talker stops with {@code codec_eos}.</li>
 *   <li>Codes are decoded to 24 kHz PCM in 72-frame chunks.</li>
 * </ol>
 * Voice cloning from a reference recording (the mmproj speaker encoder) is not supported, so the
 * Base checkpoint speaks with the voice it produces unconditioned.
 */
public final class Qwen3Tts implements AutoCloseable {

    /** Sampling and length options; defaults follow the checkpoint's generation_config. */
    public static final class Options {
        public String language = "english";
        public int maxFrames = 1500;              // 120 s
        public long seed = System.nanoTime();
        public float temperature = 0.9f;
        public int topK = 50;
        public float topP = 1.0f;
        public float repetitionPenalty = 1.05f;
        public float predictorTemperature = 0.9f;
        public int predictorTopK = 50;
    }

    /** Progress callback: frames generated so far. */
    public interface Listener {
        void onFrame(int frames);
    }

    private static final Map<String, String> LANG = new HashMap<>();
    static {
        String[][] l = {{"zh", "chinese"}, {"en", "english"}, {"de", "german"}, {"it", "italian"},
            {"pt", "portuguese"}, {"es", "spanish"}, {"ja", "japanese"}, {"ko", "korean"},
            {"fr", "french"}, {"ru", "russian"}};
        for (String[] e : l) LANG.put(e[0], e[1]);
    }

    private final ModelLoader.LoadedModel talker;
    private final InferenceEngine engine;
    private final Qwen3TtsCodec codec;
    private final Tokenizer tokenizer;
    private final int dim, codec0, codecVocab, codecBos, codecEos, codecPad, cThink, cThinkB, cThinkE;
    private final int ttsPad, ttsBos, ttsEos;
    private final boolean[] suppressed;           // indexed by codec id (id - codec0)

    private Qwen3Tts(ModelLoader.LoadedModel talker, Qwen3TtsCodec codec) {
        this.talker = talker;
        this.codec = codec;
        ModelConfig cfg = talker.config();
        this.dim = cfg.embeddingLength();
        if (codec.talkerDim() != dim) {
            throw new IllegalArgumentException("mmproj talker width " + codec.talkerDim() + " != model width " + dim);
        }
        this.tokenizer = talker.tokenizer();
        this.engine = new InferenceEngine(cfg, talker.weights(), 32768, talker.weights().ropeFreqFactors());
        codec0 = special("<|codec_0|>");
        codecBos = special("<|codec_bos|>");
        codecEos = special("<|codec_eos_token|>");
        codecPad = special("<|codec_pad|>");
        cThink = special("<|codec_think|>");
        cThinkB = special("<|codec_think_bos|>");
        cThinkE = special("<|codec_think_eos|>");
        ttsPad = special("<tts_pad>");
        ttsBos = special("<tts_text_bos>");
        ttsEos = special("<tts_text_eod>");
        if (talker.weights().output() == null) throw new IllegalStateException("talker has no codec head");
        codecVocab = (int) (talker.ggufFile().findTensor("output.weight").dimensions()[1]);
        suppressed = new boolean[codecVocab];
        int[] sup = talker.ggufFile().getMetadata().getIntArray("tokenizer.ggml.suppress_tokens");
        if (sup != null) {
            for (int id : sup) {
                int j = id - codec0;
                if (j >= 0 && j < codecVocab && id != codecEos) suppressed[j] = true;
            }
        }
    }

    public static Qwen3Tts load(Path talkerGguf, Path mmproj) throws IOException {
        ModelLoader.LoadedModel m = ModelLoader.load(talkerGguf, true);
        String arch = m.ggufFile().getMetadata().getString("general.architecture", "");
        if (!"qwen3tts".equals(arch)) {
            m.close();
            throw new IllegalArgumentException("Not a Qwen3-TTS talker (general.architecture = '" + arch + "')");
        }
        return new Qwen3Tts(m, Qwen3TtsCodec.load(mmproj));
    }

    public int sampleRate() { return Qwen3TtsCodec.SAMPLE_RATE; }

    /** Synthesize {@code text}; returns mono PCM samples in [-1, 1] at {@link #sampleRate()}. */
    public float[] synthesize(String text, Options opt, Listener listener) {
        String lang = LANG.getOrDefault(opt.language.toLowerCase(), opt.language.toLowerCase());
        int cLang = special("<|codec_language_" + lang + "|>");

        // Prompt: the chat wrap is tokenized once and sliced like the reference ([0:3] role, [3:-5] body)
        int[] ids = tokenizer.encode("<|im_start|>assistant\n" + text + "<|im_end|>\n<|im_start|>assistant\n");
        List<float[]> rows = new ArrayList<>();
        for (int i = 0; i < 3; i++) rows.add(row(ids[i]));
        rows.add(sum(ttsPad, cThink));
        rows.add(sum(ttsPad, cThinkB));
        rows.add(sum(ttsPad, cLang));
        rows.add(sum(ttsPad, cThinkE));
        rows.add(sum(ttsBos, codecPad));
        for (int i = 3; i < ids.length - 5; i++) rows.add(sum(ids[i], codecPad));
        rows.add(sum(ttsEos, codecPad));
        rows.add(sum(ttsPad, codecBos));
        float[][] prompt = rows.toArray(new float[0][]);

        int maxCtx = prompt.length + opt.maxFrames + 4;
        InferenceState state = engine.createState(Math.min(maxCtx, 32768));
        long t0 = System.nanoTime();
        engine.prefillEmbeddings(state, prompt, 0, 0);
        float[] hidden = new float[dim];
        float[] logits = new float[codecVocab];
        headLogits(state, hidden, logits);
        long tPrompt = System.nanoTime();

        Random rng = new Random(opt.seed);
        float[] padRow = row(ttsPad);
        Qwen3TtsCodec.PredictorState ps = codec.newPredictorState();
        Qwen3TtsCodec.DecoderState ds = codec.newDecoderState();
        int nCodes = codec.codebooks();
        int[][] window = new int[Qwen3TtsCodec.WINDOW][nCodes];
        int inWindow = 0;
        List<float[]> pcm = new ArrayList<>();
        int total = 0;
        boolean[] seen = new boolean[codecVocab];
        float[] next = new float[dim];
        int pos = prompt.length;
        int frames = 0;
        long decodeNs = 0;
        while (frames < opt.maxFrames) {
            int code0 = sampleTalker(logits, seen, opt, rng);
            if (code0 + codec0 == codecEos) break;
            seen[code0] = true;
            codec.predictFrame(ps, hidden, code0, opt.predictorTopK, 1.0f, opt.predictorTemperature, rng,
                window[inWindow], next);
            inWindow++;
            frames++;
            if (listener != null) listener.onFrame(frames);
            if (inWindow == Qwen3TtsCodec.WINDOW) {
                long d0 = System.nanoTime();
                float[] chunk = codec.decodeChunk(ds, window, inWindow);
                decodeNs += System.nanoTime() - d0;
                pcm.add(chunk);
                total += chunk.length;
                inWindow = 0;
            }
            for (int i = 0; i < dim; i++) next[i] += padRow[i];
            engine.forwardEmbedding(state, next, pos++);
            headLogits(state, hidden, logits);
        }
        if (inWindow > 0) {
            long d0 = System.nanoTime();
            float[] chunk = codec.decodeChunk(ds, window, inWindow);
            decodeNs += System.nanoTime() - d0;
            pcm.add(chunk);
            total += chunk.length;
        }
        float[] out = new float[total];
        int off = 0;
        for (float[] c : pcm) {
            System.arraycopy(c, 0, out, off, c.length);
            off += c.length;
        }
        double promptS = (tPrompt - t0) / 1e9, allS = (System.nanoTime() - t0) / 1e9;
        System.out.printf("  TTS: %d prompt rows in %.1f s, %d frames (%.2f s of audio) in %.1f s (decoder %.1f s), %.2fx real time%n",
            prompt.length, promptS, frames, total / (double) sampleRate(), allS, decodeNs / 1e9,
            (total / (double) sampleRate()) / allS);
        return out;
    }

    /** Output-normed hidden state and the codec-head logits of the token just processed. */
    private void headLogits(InferenceState state, float[] hidden, float[] logits) {
        engine.normedHidden(state, hidden);
        Arrays.fill(logits, 0f);
        talker.weights().output().matmulParallel(hidden, logits, codecVocab, dim);
    }

    private int sampleTalker(float[] logits, boolean[] seen, Options opt, Random rng) {
        float[] l = Arrays.copyOf(logits, codecVocab);
        for (int j = 0; j < codecVocab; j++) {
            if (suppressed[j]) {
                l[j] = Float.NEGATIVE_INFINITY;
            } else if (seen[j] && opt.repetitionPenalty != 1f) {
                l[j] = l[j] > 0 ? l[j] / opt.repetitionPenalty : l[j] * opt.repetitionPenalty;
            }
        }
        return Sampling.sample(l, codecVocab, opt.topK, opt.topP, opt.temperature, rng);
    }

    private float[] row(int token) {
        FloatTensor emb = talker.weights().tokenEmbedding();
        float[] r = new float[dim];
        long base = (long) token * dim;
        for (int i = 0; i < dim; i++) r[i] = emb.getFloat(base + i);
        return r;
    }

    private float[] sum(int a, int b) {
        float[] r = row(a), s = row(b);
        for (int i = 0; i < dim; i++) r[i] += s[i];
        return r;
    }

    private int special(String piece) {
        int[] t = tokenizer.encode(piece);
        if (t.length != 1) throw new IllegalArgumentException("Qwen3-TTS vocabulary has no token " + piece);
        return t[0];
    }

    @Override
    public void close() {
        talker.close();
    }
}
