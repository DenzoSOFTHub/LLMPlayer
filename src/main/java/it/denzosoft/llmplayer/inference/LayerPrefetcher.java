package it.denzosoft.llmplayer.inference;

import it.denzosoft.llmplayer.gguf.GGUFFile;
import it.denzosoft.llmplayer.gguf.GGUFTensorInfo;
import it.denzosoft.llmplayer.tensor.TensorData;

import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

/**
 * Asynchronous next-layer weight prefetcher for lazy (no-preload) mmap loads of dense models
 * larger than physical RAM.
 *
 * In that regime the per-token bottleneck is the SSD, not the matmul: the cyclic walk through
 * the layers defeats the page-cache LRU, so most layer pages must be re-read from disk every
 * token. While the compute thread works on layer N, this prefetcher touches the mmap pages of
 * layer N+1 on a background thread, overlapping disk I/O with compute (the AirLLM prefetch
 * pattern, ~10% end-to-end there).
 *
 * Per-layer file ranges are derived from the GGUF tensor directory ("blk.N.*" offsets), so no
 * per-tensor plumbing is needed. The queue holds a single pending request and discards on
 * overflow: when the disk lags behind compute, dropped requests are harmless — the compute
 * thread simply faults the pages itself. Created only when {@code mmap.advise=sequential}
 * (dense lazy load) unless forced via {@code -Dmmap.prefetch=true}; disable with
 * {@code -Dmmap.prefetch=false}.
 */
public final class LayerPrefetcher {

    private final TensorData mappedFile;
    private final long[] rangeStart;   // absolute file offset of each layer's first tensor byte
    private final long[] rangeEnd;     // absolute file offset past each layer's last tensor byte
    private final int blockCount;
    private final int startLayer;      // first CPU-resident layer (GPU layers need no page-in)
    private final ThreadPoolExecutor executor;
    private volatile int lastRequested = -1;

    public static LayerPrefetcher createIfEnabled(GGUFFile gguf, int blockCount, int startLayer) {
        String prefetch = System.getProperty("mmap.prefetch", "auto");
        if ("false".equals(prefetch)) return null;
        boolean denseLazyLoad = "sequential".equals(System.getProperty("mmap.advise", ""));
        if (!"true".equals(prefetch) && !denseLazyLoad) return null;
        if (gguf == null || blockCount <= 0 || startLayer >= blockCount) return null;
        try {
            LayerPrefetcher p = new LayerPrefetcher(gguf, blockCount, Math.max(0, startLayer));
            System.out.println("  mmap: next-layer prefetch enabled (overlap disk I/O with compute, "
                + "layers " + Math.max(0, startLayer) + ".." + (blockCount - 1) + ")");
            return p;
        } catch (Throwable t) {
            return null; // best-effort feature, never block a load
        }
    }

    private LayerPrefetcher(GGUFFile gguf, int blockCount, int startLayer) {
        this.mappedFile = gguf.getMappedFile();
        this.blockCount = blockCount;
        this.startLayer = startLayer;
        this.rangeStart = new long[blockCount];
        this.rangeEnd = new long[blockCount];
        java.util.Arrays.fill(rangeStart, Long.MAX_VALUE);
        long dataOffset = gguf.getTensorDataOffset();
        for (GGUFTensorInfo info : gguf.getTensorInfos()) {
            int layer = layerIndexOf(info.name());
            if (layer < 0 || layer >= blockCount) continue;
            long start = dataOffset + info.offset();
            long end = start + info.byteSize();
            if (start < rangeStart[layer]) rangeStart[layer] = start;
            if (end > rangeEnd[layer]) rangeEnd[layer] = end;
        }
        ThreadFactory daemon = new ThreadFactory() {
            @Override
            public Thread newThread(Runnable r) {
                Thread t = new Thread(r, "layer-prefetch");
                t.setDaemon(true);
                return t;
            }
        };
        this.executor = new ThreadPoolExecutor(1, 1, 30, TimeUnit.SECONDS,
            new ArrayBlockingQueue<Runnable>(1), daemon,
            new ThreadPoolExecutor.DiscardPolicy());
    }

    /** Parse N from "blk.N.xxx" tensor names; -1 for non-layer tensors. */
    private static int layerIndexOf(String name) {
        if (!name.startsWith("blk.")) return -1;
        int dot = name.indexOf('.', 4);
        if (dot < 0) return -1;
        try {
            return Integer.parseInt(name.substring(4, dot));
        } catch (NumberFormatException e) {
            return -1;
        }
    }

    /**
     * Request an asynchronous page-in of the given layer's weight range. Called by the forward
     * loop with {@code currentLayer + 1}; wraps past the last layer to the first CPU layer so
     * the next token's first read is already in flight. Never blocks the caller.
     */
    public void prefetchLayer(int layer) {
        if (layer >= blockCount) layer = startLayer;       // wrap to next token's first CPU layer
        if (layer < startLayer || layer == lastRequested) return;
        final int target = layer;
        final long start = rangeStart[target];
        final long end = rangeEnd[target];
        if (start >= end) return;                          // no mmap range known for this layer
        lastRequested = target;
        executor.execute(new Runnable() {
            @Override
            public void run() {
                try {
                    mappedFile.slice(start, end - start).preload();
                } catch (Throwable ignore) {
                    // best-effort: a failed prefetch just means the compute thread pages it in
                }
            }
        });
    }

    /** Stop the prefetch thread. Call before unmapping the model file. */
    public void stop() {
        executor.shutdownNow();
        try {
            executor.awaitTermination(2, TimeUnit.SECONDS);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }
}
