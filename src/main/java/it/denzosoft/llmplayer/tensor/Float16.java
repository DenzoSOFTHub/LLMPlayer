package it.denzosoft.llmplayer.tensor;

/**
 * Portable FP16 to FP32 conversion, compatible with Java 8+.
 * Equivalent to Float.float16ToFloat() added in Java 20.
 */
public final class Float16 {

    private Float16() {}

    public static float toFloat(short bits) {
        int bin16 = Short.toUnsignedInt(bits);
        int sign = bin16 >>> 15;
        int exponent = (bin16 >> 10) & 0x1F;
        int significand = bin16 & 0x03FF;

        if (exponent == 0) {
            if (significand == 0) {
                return Float.intBitsToFloat(sign << 31); // +0 or -0
            }
            // Subnormal: normalize
            while ((significand & 0x0400) == 0) {
                significand <<= 1;
                exponent--;
            }
            significand &= 0x03FF;
            exponent++;
        } else if (exponent == 0x1F) {
            // Infinity or NaN
            return Float.intBitsToFloat((sign << 31) | 0x7F800000 | (significand << 13));
        }

        return Float.intBitsToFloat((sign << 31) | ((exponent + (127 - 15)) << 23) | (significand << 13));
    }
}
