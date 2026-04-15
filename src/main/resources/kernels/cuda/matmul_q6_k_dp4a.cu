/**
 * Q6_K × Q8_1 matmul using __dp4a int8 dot product.
 *
 * Q6_K block layout (210 bytes per 256-element super-block):
 *   ql:   bytes 0..127   (low 4-bit nibbles, 256 nibbles)
 *   qh:   bytes 128..191 (high 2-bit pieces, 256 × 2 bits)
 *   sc:   bytes 192..207 (16 signed int8 scales, one per 16 elements)
 *   d:    bytes 208..209 (FP16 super-block scale)
 *
 * Element-position layout (matches `Q6_KFloatTensor.getFloat` and `matmul_q6_k.cu`):
 *   For position j in [0..255]:
 *     half=j/128, jLocal=j%128, quadrant=jLocal/32, l=jLocal%32
 *     qlBase = bo + half*64;  qhBase = bo + 128 + half*32
 *     quadrant 0: ql_low  at ql[qlBase+l],     qh bits[1:0] of qh[qhBase+l]
 *     quadrant 1: ql_low  at ql[qlBase+32+l],  qh bits[3:2]
 *     quadrant 2: ql_high at ql[qlBase+l],     qh bits[5:4]
 *     quadrant 3: ql_high at ql[qlBase+32+l],  qh bits[7:6]
 *   Scale index: subBlock = j/16  (16 scales per super-block, one per 16 consecutive elements).
 *
 * dp4a strategy: each lane processes 4 consecutive elements at a time. 4 consecutive
 * elements all live in the same (half, quadrant), and either the same scale (if l aligned)
 * or two scales (if straddling the 16-element boundary). We choose chunk granularity = 4 so
 * that l + 0..3 always falls within one scale (since 16 % 4 == 0).
 *
 * Per chunk: read 4 ql bytes + 4 qh bytes, extract 4 q6 values, dp4a against 4 Q8_1 input bytes.
 *
 * Threading: 1 warp = 1 row, blockDim.x = 256 (8 rows per block) — same shape as
 * matmul_q4_k_dp4a so getMatmulGridDim/BlockDim defaults work unchanged.
 *
 * Each lane handles one (super-block, half, quadrant, k=0..7) tuple per outer iteration.
 * Total work units per row: numSuperBlocks * 2 * 4 * 8 = numSuperBlocks * 64 chunks.
 * For Llama-1B wv (cols=2048, numSB=8): 512 chunks per row → 16 per lane.
 */
extern "C" __global__ void matmul_q6_k_dp4a(
    const unsigned char* __restrict__ W,
    const unsigned char* __restrict__ input,    // Q8_1 quantized: (cols/32) × 40 bytes
    float* __restrict__ output,
    const int rows, const int cols, const int addToOutput)
{
    int warpId = threadIdx.x / 32;
    int lane   = threadIdx.x & 31;
    int rowsPerBlock = blockDim.x / 32;
    int row = blockIdx.x * rowsPerBlock + warpId;
    if (row >= rows) return;

    int numSuperBlocks = cols / 256;
    long rowOffset = (long)row * numSuperBlocks * 210;
    int totalChunks = numSuperBlocks * 64;  // 8 quadrants/SB × 8 chunks/quadrant
    float sum = 0.0f;

    // d (FP16 → float) — same per super-block; reload inside loop is OK (L1 caches it).

    for (int chunk = lane; chunk < totalChunks; chunk += 32) {
        // Decompose chunk index
        int sb       = chunk >> 6;       // super-block (0..numSB-1)
        int inSB     = chunk & 63;       // chunk within super-block (0..63)
        int half     = inSB >> 5;        // half (0 or 1) — 32 chunks per half
        int quadrant = (inSB >> 3) & 3;  // quadrant 0..3 — 8 chunks per quadrant
        int kIdx     = inSB & 7;         // chunk within quadrant (0..7); l = kIdx*4

        long bo = rowOffset + (long)sb * 210;

        // Super-block scale d (FP16) — packed at offset 208 (2 bytes)
        unsigned short dRaw = *(const unsigned short*)(W + bo + 208);
        // Inline FP16→FP32 (same as half2float in our kernels)
        unsigned int dSign = (dRaw >> 15) & 1;
        unsigned int dExp  = (dRaw >> 10) & 0x1F;
        unsigned int dMant = dRaw & 0x3FF;
        float d;
        if (dExp == 0) {
            if (dMant == 0) {
                d = dSign ? -0.0f : 0.0f;
            } else {
                while (!(dMant & 0x400)) { dMant <<= 1; dExp = (unsigned int)((int)dExp - 1); }
                dExp++; dMant &= 0x3FF;
                unsigned int f = (dSign << 31) | ((dExp + 112) << 23) | (dMant << 13);
                d = *(float*)&f;
            }
        } else if (dExp == 31) {
            unsigned int f = (dSign << 31) | 0x7F800000 | (dMant << 13);
            d = *(float*)&f;
        } else {
            unsigned int f = (dSign << 31) | ((dExp + 112) << 23) | (dMant << 13);
            d = *(float*)&f;
        }

        long qlBase = bo + (long)half * 64;
        long qhBase = bo + 128 + (long)half * 32;
        int l = kIdx * 4;  // position within quadrant: 0,4,8,12,16,20,24,28

        // Element positions: half*128 + quadrant*32 + l + (0..3)
        int elemStart = half * 128 + quadrant * 32 + l;

        // Scale index: each scale covers 16 consecutive elements. (elemStart / 16) is constant
        // for our 4-element chunk because elemStart % 16 ∈ {0, 4, 8, 12}.
        int scIdx = elemStart >> 4;
        int sc = (signed char)__ldg(&W[bo + 192 + scIdx]);

        // Read 4 ql bytes (consecutive in storage). Q6_K block size is 210 (not a multiple of 4),
        // so super-block bases are NOT always 4-byte aligned — must use byte loads, not uint32.
        int qlOff = ((quadrant & 1) ? 32 : 0) + l;
        unsigned int ql4 =
              ((unsigned int)__ldg(&W[qlBase + qlOff    ])      )
            | ((unsigned int)__ldg(&W[qlBase + qlOff + 1]) <<  8)
            | ((unsigned int)__ldg(&W[qlBase + qlOff + 2]) << 16)
            | ((unsigned int)__ldg(&W[qlBase + qlOff + 3]) << 24);

        // Read 4 qh bytes (consecutive)
        unsigned int qh4 =
              ((unsigned int)__ldg(&W[qhBase + l    ])      )
            | ((unsigned int)__ldg(&W[qhBase + l + 1]) <<  8)
            | ((unsigned int)__ldg(&W[qhBase + l + 2]) << 16)
            | ((unsigned int)__ldg(&W[qhBase + l + 3]) << 24);

        // Extract nibbles per byte (low if quadrant ∈ {0,1}, high if {2,3})
        // and high-bit pairs at shift 2*quadrant
        bool hiNibble = (quadrant & 2) != 0;
        unsigned int qlMask = 0x0F0F0F0Fu;
        unsigned int qlNibbles = hiNibble ? ((ql4 >> 4) & qlMask) : (ql4 & qlMask);
        unsigned int qhBits = (qh4 >> (2 * quadrant)) & 0x03030303u;
        unsigned int qPacked6bit = qlNibbles | (qhBits << 4);  // 4 × 6-bit values, each in low 6 bits of byte

        // Subtract 32 to make signed Q6 (range -32..31). __vsubss4 is per-byte saturating sub.
        int qSigned = __vsubss4((int)qPacked6bit, 0x20202020);

        // Read 4 Q8_1 input bytes at element offset (sb*256 + elemStart)
        int q8BlockIdx = (sb * 256 + elemStart) >> 5;          // Q8_1 block index (32 elems per block)
        int posInBlock = (sb * 256 + elemStart) & 31;          // byte offset within Q8_1 block
        long q8Off = (long)q8BlockIdx * 40;
        float inScale = *(const float*)(input + q8Off);
        // We don't need inSum separately for Q6_K because the offset (-32) is already baked into qSigned.

        int in4 = *(const int*)(input + q8Off + 8 + posInBlock);

        int dp = __dp4a(qSigned, in4, 0);

        // Per-element value: d * sc * (q - 32) * input[i] = d * sc * inScale * dp
        sum += d * (float)sc * inScale * (float)dp;
    }

    // Warp reduction
    for (int off = 16; off > 0; off >>= 1)
        sum += __shfl_down_sync(0xFFFFFFFF, sum, off);

    if (lane == 0) {
        if (addToOutput) output[row] += sum;
        else             output[row]  = sum;
    }
}
