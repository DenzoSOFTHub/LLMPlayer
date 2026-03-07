package it.denzosoft.llmplayer.tuning.train;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;

/**
 * Mutable F32 tensor for LoRA weights, gradients, and optimizer states.
 * Simple 2D matrix (rows x cols) stored in row-major order.
 */
public class TrainableTensor {

    private final float[] data;
    private final int rows;
    private final int cols;

    public TrainableTensor(int rows, int cols) {
        this.rows = rows;
        this.cols = cols;
        this.data = new float[rows * cols];
    }

    /** Create from existing data (not copied). */
    public TrainableTensor(float[] data, int rows, int cols) {
        this.data = data;
        this.rows = rows;
        this.cols = cols;
    }

    public int rows() { return rows; }
    public int cols() { return cols; }
    public int size() { return data.length; }
    public float[] data() { return data; }

    public float get(int row, int col) { return data[row * cols + col]; }
    public void set(int row, int col, float v) { data[row * cols + col] = v; }

    /** Initialize with Kaiming/He uniform: U(-sqrt(6/fan_in), sqrt(6/fan_in)). */
    public void initKaiming(java.util.Random rng) {
        float bound = (float) Math.sqrt(6.0 / cols);
        for (int i = 0; i < data.length; i++) {
            data[i] = (rng.nextFloat() * 2 - 1) * bound;
        }
    }

    /** Initialize to zeros. */
    public void zero() { Arrays.fill(data, 0f); }

    /** Scale all values: data *= scale. */
    public void scale(float s) {
        for (int i = 0; i < data.length; i++) data[i] *= s;
    }

    /** Accumulate: this += other * scale. */
    public void addScaled(TrainableTensor other, float scale) {
        for (int i = 0; i < data.length; i++) {
            data[i] += other.data[i] * scale;
        }
    }

    /**
     * Matrix multiply: C = this(rows x cols) * B(bRows x bCols).
     * Requires this.cols == B.rows.
     * @return new tensor of shape (this.rows, B.cols)
     */
    public TrainableTensor matmul(TrainableTensor b) {
        assert cols == b.rows : "Dimension mismatch: " + cols + " vs " + b.rows;
        TrainableTensor c = new TrainableTensor(rows, b.cols);
        for (int i = 0; i < rows; i++) {
            for (int k = 0; k < cols; k++) {
                float aik = data[i * cols + k];
                if (aik == 0f) continue;
                for (int j = 0; j < b.cols; j++) {
                    c.data[i * b.cols + j] += aik * b.data[k * b.cols + j];
                }
            }
        }
        return c;
    }

    /**
     * Transpose this matrix.
     * @return new tensor of shape (cols, rows)
     */
    public TrainableTensor transpose() {
        TrainableTensor t = new TrainableTensor(cols, rows);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                t.data[j * rows + i] = data[i * cols + j];
            }
        }
        return t;
    }

    /**
     * Compute dot product of row i of this tensor with the given vector.
     * Vector must have length == cols.
     */
    public float dotRow(int row, float[] vec) {
        float sum = 0;
        int offset = row * cols;
        for (int j = 0; j < cols; j++) {
            sum += data[offset + j] * vec[j];
        }
        return sum;
    }

    /**
     * Matrix-vector multiply: y = this(rows x cols) * x(cols).
     * @return float array of length rows
     */
    public float[] matvec(float[] x) {
        float[] y = new float[rows];
        for (int i = 0; i < rows; i++) {
            y[i] = dotRow(i, x);
        }
        return y;
    }

    /** L2 norm of all values. */
    public float norm() {
        double sum = 0;
        for (float v : data) sum += (double) v * v;
        return (float) Math.sqrt(sum);
    }

    // --- Serialization ---

    public void save(Path file) throws IOException {
        try (DataOutputStream dos = new DataOutputStream(
                new BufferedOutputStream(Files.newOutputStream(file)))) {
            dos.writeInt(rows);
            dos.writeInt(cols);
            for (float v : data) dos.writeFloat(v);
        }
    }

    public static TrainableTensor load(Path file) throws IOException {
        try (DataInputStream dis = new DataInputStream(
                new BufferedInputStream(Files.newInputStream(file)))) {
            int rows = dis.readInt();
            int cols = dis.readInt();
            float[] data = new float[rows * cols];
            for (int i = 0; i < data.length; i++) data[i] = dis.readFloat();
            return new TrainableTensor(data, rows, cols);
        }
    }
}
