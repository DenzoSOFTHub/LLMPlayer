package it.denzosoft.llmplayer.inference;

/**
 * Thrown when a GPU-resident forward pass fails in the middle of a sequence. The pass owned the
 * device-side KV cache (and recurrent state) for every earlier position, so the CPU cannot take
 * over without prefilling the sequence again. The engine has already dropped the GPU pass; a new
 * generation will run on the CPU from position 0.
 */
public class GpuFailureException extends RuntimeException {
    public GpuFailureException(String message, Throwable cause) {
        super(message + ": " + describe(cause), cause);
    }

    /**
     * Root cause as "Class: message": GPU passes are invoked reflectively, so the exception that
     * reaches the engine is usually an InvocationTargetException whose own message is null.
     * Prints the stack trace too when -Dcuda.debug=true.
     */
    public static String describe(Throwable e) {
        Throwable c = e;
        while (c.getCause() != null && c.getCause() != c) c = c.getCause();
        if ("true".equals(System.getProperty("cuda.debug"))) c.printStackTrace(System.err);
        return c.getClass().getSimpleName() + (c.getMessage() != null ? ": " + c.getMessage() : "");
    }
}
