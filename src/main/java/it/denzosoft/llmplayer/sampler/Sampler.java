package it.denzosoft.llmplayer.sampler;

public interface Sampler {
    int sample(float[] logits);
}
