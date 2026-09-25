package it.denzosoft.llmplayer.vision;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.IOException;

/**
 * Image decoding and preprocessing for the Qwen-VL family of vision encoders, following llama.cpp
 * {@code mtmd-image.cpp}: the image is resized with its aspect ratio preserved so that both sides are
 * multiples of {@code alignSize} (patch size × spatial merge) and the pixel count lies between
 * {@code minPixels} and {@code maxPixels} ("smart resize" in transformers), using a Pillow-style
 * separable bicubic filter (a = -0.5, widened when downsampling), then normalised per channel with
 * {@code (v / 255 - mean) / std}.
 *
 * <p>Decoding uses {@code javax.imageio} (JPEG, PNG, BMP, GIF), so no dependency is needed.
 */
public final class ImagePreprocessor {

    private ImagePreprocessor() {}

    /** A preprocessed image: planar float channels {@code [3][height * width]}. */
    public static final class Image {
        public final int width, height;
        public final float[][] channels;

        Image(int width, int height, float[][] channels) {
            this.width = width;
            this.height = height;
            this.channels = channels;
        }
    }

    /** Decode an encoded image (JPEG, PNG, ...) into packed 0xRRGGBB pixels. */
    public static int[] decode(byte[] bytes, int[] sizeOut) throws IOException {
        BufferedImage img = ImageIO.read(new ByteArrayInputStream(bytes));
        if (img == null) throw new IOException("Unsupported or corrupt image data");
        int w = img.getWidth(), h = img.getHeight();
        int[] rgb = img.getRGB(0, 0, w, h, null, 0, w);
        sizeOut[0] = w;
        sizeOut[1] = h;
        return rgb;
    }

    /**
     * Target size for "smart resize" (llama.cpp {@code calc_size_preserved_ratio}): round each side
     * to the nearest multiple of {@code align}, then scale down (floor) or up (ceil) with the aspect
     * ratio kept when the pixel count is outside [minPixels, maxPixels].
     */
    public static int[] smartResize(int width, int height, int align, int minPixels, int maxPixels) {
        int wBar = Math.max(align, roundBy(width, align));
        int hBar = Math.max(align, roundBy(height, align));
        if (maxPixels > 0 && (long) hBar * wBar > maxPixels) {
            double beta = Math.sqrt((double) height * width / maxPixels);
            hBar = Math.max(align, floorBy(height / beta, align));
            wBar = Math.max(align, floorBy(width / beta, align));
        } else if (minPixels > 0 && (long) hBar * wBar < minPixels) {
            double beta = Math.sqrt((double) minPixels / ((double) height * width));
            hBar = ceilBy(height * beta, align);
            wBar = ceilBy(width * beta, align);
        }
        return new int[] {wBar, hBar};
    }

    private static int roundBy(double x, int f) { return (int) Math.round(x / f) * f; }
    private static int ceilBy(double x, int f) { return (int) Math.ceil(x / f) * f; }
    private static int floorBy(double x, int f) { return (int) Math.floor(x / f) * f; }

    /**
     * Decode, smart-resize and normalise. {@code mean}/{@code std} are per channel (R, G, B).
     */
    public static Image preprocess(byte[] encoded, int align, int minPixels, int maxPixels,
                                   float[] mean, float[] std) throws IOException {
        int[] size = new int[2];
        int[] rgb = decode(encoded, size);
        int srcW = size[0], srcH = size[1];
        int[] target = smartResize(srcW, srcH, align, minPixels, maxPixels);
        int dstW = target[0], dstH = target[1];

        // planar float copy of the source, 0..255
        float[][] src = new float[3][srcW * srcH];
        for (int i = 0; i < rgb.length; i++) {
            int p = rgb[i];
            src[0][i] = (p >> 16) & 0xFF;
            src[1][i] = (p >> 8) & 0xFF;
            src[2][i] = p & 0xFF;
        }
        float[][] out = new float[3][];
        for (int c = 0; c < 3; c++) {
            float[] resized = (dstW == srcW && dstH == srcH) ? src[c] : resizeBicubic(src[c], srcW, srcH, dstW, dstH);
            float[] n = new float[dstW * dstH];
            for (int i = 0; i < n.length; i++) {
                float v = Math.max(0f, Math.min(255f, Math.round(resized[i]))); // 8-bit like the reference
                n[i] = (v / 255f - mean[c]) / std[c];
            }
            out[c] = n;
        }
        return new Image(dstW, dstH, out);
    }

    // ==================== Pillow-style separable bicubic ====================

    private static final double BICUBIC_A = -0.5;

    private static double cubic(double x) {
        if (x < 0) x = -x;
        if (x < 1.0) return ((BICUBIC_A + 2.0) * x - (BICUBIC_A + 3.0)) * x * x + 1;
        if (x < 2.0) return (((x - 5) * x + 8) * x - 4) * BICUBIC_A;
        return 0.0;
    }

    private static float[] resizeBicubic(float[] src, int srcW, int srcH, int dstW, int dstH) {
        // horizontal pass: [srcH][dstW]
        float[] tmp = new float[srcH * dstW];
        Coeffs cx = coeffs(srcW, dstW);
        for (int y = 0; y < srcH; y++) {
            int row = y * srcW;
            for (int x = 0; x < dstW; x++) {
                double s = 0;
                int b = cx.min[x], n = cx.count[x], off = x * cx.ksize;
                for (int k = 0; k < n; k++) s += cx.w[off + k] * src[row + b + k];
                tmp[y * dstW + x] = (float) s;
            }
        }
        // vertical pass: [dstH][dstW]
        float[] dst = new float[dstH * dstW];
        Coeffs cy = coeffs(srcH, dstH);
        for (int y = 0; y < dstH; y++) {
            int b = cy.min[y], n = cy.count[y], off = y * cy.ksize;
            for (int x = 0; x < dstW; x++) {
                double s = 0;
                for (int k = 0; k < n; k++) s += cy.w[off + k] * tmp[(b + k) * dstW + x];
                dst[y * dstW + x] = (float) s;
            }
        }
        return dst;
    }

    private static final class Coeffs {
        int ksize;
        int[] min, count;
        double[] w;
    }

    /** Filter taps per output pixel, as Pillow's precompute_coeffs (support widened when shrinking). */
    private static Coeffs coeffs(int inSize, int outSize) {
        double scale = (double) inSize / outSize;
        double filterScale = Math.max(1.0, scale);
        double support = 2.0 * filterScale;
        int ksize = (int) Math.ceil(support) * 2 + 1;
        Coeffs c = new Coeffs();
        c.ksize = ksize;
        c.min = new int[outSize];
        c.count = new int[outSize];
        c.w = new double[outSize * ksize];
        for (int xx = 0; xx < outSize; xx++) {
            double center = (xx + 0.5) * scale;
            double ss = 1.0 / filterScale;
            int xmin = (int) Math.max(0, Math.floor(center - support + 0.5));
            int xmax = (int) Math.min(inSize, Math.floor(center + support + 0.5));
            int n = xmax - xmin;
            double total = 0;
            for (int x = 0; x < n; x++) {
                double w = cubic((x + xmin - center + 0.5) * ss);
                c.w[xx * ksize + x] = w;
                total += w;
            }
            if (total != 0) {
                for (int x = 0; x < n; x++) c.w[xx * ksize + x] /= total;
            }
            c.min[xx] = xmin;
            c.count[xx] = n;
        }
        return c;
    }
}
