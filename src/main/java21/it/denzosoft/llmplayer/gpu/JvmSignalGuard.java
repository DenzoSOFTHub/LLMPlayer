package it.denzosoft.llmplayer.gpu;

import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;

/**
 * Protects the JVM's own signal handlers from native libraries that replace them.
 *
 * <p>The HotSpot JVM uses SIGSEGV, SIGBUS, SIGFPE and SIGILL internally: C2-compiled code turns
 * null checks into memory accesses that fault, and stack banging and safepoint polls rely on
 * faults too. The JVM's handler recognises these faults and resumes the thread. Enumerating
 * OpenCL devices loads every installed ICD, and PoCL (the CPU OpenCL driver) with its LLVM
 * runtime installs its own handlers for exactly these signals (plus SIGTRAP). From then on the
 * first ordinary JVM fault — an implicit null check that trips, for instance — terminates the
 * process with a bare "Segmentation fault" and no hs_err report. It happened reproducibly with a
 * GPU-mode run that called {@code LLMEngine.autoConfigureGpu} (which probes OpenCL) and then
 * decoded token by token; with the JDK's signal-chaining library preloaded the same run was fine.
 *
 * <p>Usage: {@link #save()} before the native call, {@link #restore()} after it. Only the device
 * probe is wrapped, not an OpenCL context used for computation: PoCL relies on its SIGFPE handler
 * when a kernel divides by zero. A process that computes on PoCL should preload the JDK's
 * {@code libjsig.so} instead (signal chaining). Linux x86-64/aarch64 only; a no-op elsewhere or
 * when {@code sigaction} cannot be bound.
 */
public final class JvmSignalGuard {

    // SIGILL, SIGTRAP, SIGBUS, SIGFPE, SIGSEGV on Linux
    private static final int[] SIGNALS = {4, 5, 7, 8, 11};
    private static final String[] NAMES = {"SIGILL", "SIGTRAP", "SIGBUS", "SIGFPE", "SIGSEGV"};
    /** sizeof(struct sigaction) is 152 bytes with glibc on x86-64; leave room. */
    private static final long SA_SIZE = 256;
    private static final MethodHandle SIGACTION = bind();
    private static volatile boolean reported;

    private final Arena arena;
    private final MemorySegment saved;

    private JvmSignalGuard(Arena arena, MemorySegment saved) {
        this.arena = arena;
        this.saved = saved;
    }

    private static MethodHandle bind() {
        try {
            if (!System.getProperty("os.name", "").toLowerCase().contains("linux")) return null;
            Linker l = Linker.nativeLinker();
            return l.downcallHandle(l.defaultLookup().find("sigaction").orElseThrow(),
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS));
        } catch (Throwable e) {
            return null;
        }
    }

    /** Snapshot the current handlers (the JVM's, when called before any foreign library ran). */
    public static JvmSignalGuard save() {
        if (SIGACTION == null) return new JvmSignalGuard(null, null);
        Arena a = Arena.ofConfined();
        MemorySegment s = a.allocate(SA_SIZE * SIGNALS.length);
        try {
            for (int i = 0; i < SIGNALS.length; i++) {
                int rc = (int) SIGACTION.invokeExact(SIGNALS[i], MemorySegment.NULL, s.asSlice(i * SA_SIZE, SA_SIZE));
                if (rc != 0) { a.close(); return new JvmSignalGuard(null, null); }
            }
        } catch (Throwable e) {
            a.close();
            return new JvmSignalGuard(null, null);
        }
        return new JvmSignalGuard(a, s);
    }

    /** Re-install every snapshotted handler that a native call replaced; returns how many. */
    public int restore() {
        if (saved == null) return 0;
        int restored = 0;
        StringBuilder which = new StringBuilder();
        try (Arena t = Arena.ofConfined()) {
            MemorySegment cur = t.allocate(SA_SIZE);
            for (int i = 0; i < SIGNALS.length; i++) {
                MemorySegment mine = saved.asSlice(i * SA_SIZE, SA_SIZE);
                if ((int) SIGACTION.invokeExact(SIGNALS[i], MemorySegment.NULL, cur) != 0) continue;
                if (cur.get(ValueLayout.JAVA_LONG, 0) == mine.get(ValueLayout.JAVA_LONG, 0)) continue;
                if ((int) SIGACTION.invokeExact(SIGNALS[i], mine, MemorySegment.NULL) == 0) {
                    restored++;
                    which.append(which.length() == 0 ? "" : " ").append(NAMES[i]);
                }
            }
        } catch (Throwable ignored) {
            // best effort
        } finally {
            arena.close();
        }
        if (restored > 0 && !reported) {
            reported = true;
            System.err.println("OpenCL device probe replaced the JVM's signal handlers (" + which
                + ", PoCL/LLVM); restored them");
        }
        return restored;
    }
}
