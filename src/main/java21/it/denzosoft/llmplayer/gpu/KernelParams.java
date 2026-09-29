package it.denzosoft.llmplayer.gpu;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * Pre-allocated CUDA kernel parameter block: one 8-byte slot per argument plus the {@code void**}
 * array cuLaunchKernel expects. The pointer array is built once; callers only overwrite slot
 * values before each launch, so a hot path that reuses one instance allocates nothing.
 */
public final class KernelParams {
    private final MemorySegment args;
    private final MemorySegment ptrs;

    public KernelParams(Arena arena, int numArgs) {
        args = arena.allocate(numArgs * 8L, 8);
        ptrs = arena.allocate(ValueLayout.ADDRESS, numArgs);
        for (int i = 0; i < numArgs; i++) {
            ptrs.setAtIndex(ValueLayout.ADDRESS, i, args.asSlice(i * 8L, 8));
        }
    }

    public KernelParams setLong(int i, long v) { args.set(ValueLayout.JAVA_LONG, i * 8L, v); return this; }
    public KernelParams setInt(int i, int v) { args.set(ValueLayout.JAVA_INT, i * 8L, v); return this; }
    public KernelParams setFloat(int i, float v) { args.set(ValueLayout.JAVA_FLOAT, i * 8L, v); return this; }

    /** The {@code void**} kernelParams array. */
    public MemorySegment ptrs() { return ptrs; }
}
