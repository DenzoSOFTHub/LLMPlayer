package it.denzosoft.llmplayer.inference;

import java.util.Arrays;

/**
 * Rope positions of a multimodal sequence for Qwen-VL multi-axis RoPE, as llama.cpp mtmd assigns
 * them. Image token {@code i} of an image starting at logical position {@code p0} with a merged grid
 * of {@code nx} columns gets {@code (t, h, w, e) = (p0, p0 + i / nx, p0 + i % nx, 0)}; the image
 * advances the logical position by {@code max(nx, ny)} instead of its token count, so every later
 * text token has {@code t = h = w = kvIndex + delta} where delta accumulates
 * {@code max(nx, ny) - nTokens} over the preceding images.
 *
 * <p>Positions are looked up by KV index, which stays sequential; only the rotation angles change.
 */
public final class MRopePositions {

    private int[] start = new int[4], len = new int[4], nx = new int[4], ny = new int[4];
    private int count;

    /** Register an image occupying KV slots {@code [kvStart, kvStart + nTokens)}. Images must be added in order. */
    public void addImage(int kvStart, int nTokens, int gridX, int gridY) {
        if (count == start.length) {
            start = Arrays.copyOf(start, count * 2);
            len = Arrays.copyOf(len, count * 2);
            nx = Arrays.copyOf(nx, count * 2);
            ny = Arrays.copyOf(ny, count * 2);
        }
        start[count] = kvStart;
        len[count] = nTokens;
        nx[count] = gridX;
        ny[count] = gridY;
        count++;
    }

    /** Fill {@code out} with the (t, h, w, e) rope positions of KV slot {@code kv}. */
    public void get(int kv, int[] out) {
        int delta = 0;
        for (int s = 0; s < count; s++) {
            if (kv < start[s]) break;
            if (kv < start[s] + len[s]) {
                int p0 = start[s] + delta;
                int i = kv - start[s];
                out[0] = p0;
                out[1] = p0 + i / nx[s];
                out[2] = p0 + i % nx[s];
                out[3] = 0;
                return;
            }
            delta += Math.max(nx[s], ny[s]) - len[s];
        }
        int p = kv + delta;
        out[0] = p;
        out[1] = p;
        out[2] = p;
        out[3] = 0;
    }
}
