// src/kernels/cpu/iq3s_neon.cpp - the IQ3_S gate/up rows on Arm64 with dotprod: the 32-value decode
// (the iq3s_grid magnitudes 10-bit-indexed from qs+qh, the 16-bit sign words, the nibble scales)
// runs once per verify window and every token applies it with four SDOTs; ggml-cpu's
// ggml_vec_dot_iq3_s_q8_K (ggml-cpu/arch/arm/quants.c) decodes for every token.  The accumulation
// reproduces that dot exactly: sumi1/sumi2 INT accumulate over the block (order-free), one float
// step per block sumf += d*(sumi1+sumi2) with d = f16(d)*y.d - so every token's rows are
// bit-identical to the ggml dot at any nt and to each other.  native_expert.cpp dispatches here
// when cpu_neon_dot_ok() says the CPU has vdotq_s32; STRATA_NO_IQ3S falls back to ggml-cpu's dot.
#include "strata/kernels/cpu/iq_avx2.hpp"

#include "ggml.h"

#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"

#include <arm_neon.h>

#include <cmath>
#include <cstdint>
#include <cstring>

namespace strata::kernels::cpu {
namespace {

// ggml-common.h's ggml_vld1q_* helpers are C-section macros; load through plain C++ intrinsics.
inline uint32x4_t vld_grid_u32(uint32_t w, uint32_t x, uint32_t y, uint32_t z) {
    const uint32_t v[4] = { w, x, y, z };
    return vld1q_u32(v);
}
struct u8x2 { uint8x16_t val[2]; };
inline u8x2 vld_u8_x2(const uint8_t* p) {
    u8x2 r; r.val[0] = vld1q_u8(p); r.val[1] = vld1q_u8(p + 16); return r;
}
struct s8x4 { int8x16_t val[4]; };
inline s8x4 vld_s8_x4(const int8_t* p) {
    s8x4 r;
    r.val[0] = vld1q_s8(p); r.val[1] = vld1q_s8(p + 16);
    r.val[2] = vld1q_s8(p + 32); r.val[3] = vld1q_s8(p + 48);
    return r;
}

template <int NT>
__attribute__((target("+dotprod")))
void iq3s_gu_kernel(const uint8_t* blob, size_t gu_row, size_t up_off, int n,
                    const block_q8_K* const* y, float* const* ff, int r0, int r1) {
    static const uint8_t k_mask1[32] = { 0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1, 2,2,2,2,2,2,2,2, 3,3,3,3,3,3,3,3 };
    static const uint8_t k_mask2[16] = { 0x01,0x02,0x04,0x08,0x10,0x20,0x40,0x80, 0x01,0x02,0x04,0x08,0x10,0x20,0x40,0x80 };
    static const int16_t k_shift[8]  = { 8,7,6,5,4,3,2,1 };
    const u8x2 mask1 = vld_u8_x2(k_mask1);
    const uint8x16_t mask2 = vld1q_u8(k_mask2);
    const int16x8_t hshift = vld1q_s16(k_shift);
    const uint16x8_t m256 = vdupq_n_u16(256);
    const uint8x16_t m1 = vdupq_n_u8(1);

    const int nb = n / QK_K;
    float g[NT], u[NT];
    for (int r = r0; r < r1; ++r) {
        const block_iq3_s* gx = (const block_iq3_s*) (blob + (size_t) r * gu_row);
        const block_iq3_s* ux = (const block_iq3_s*) (blob + up_off + (size_t) r * gu_row);
        for (int t = 0; t < NT; ++t) { g[t] = 0.f; u[t] = 0.f; }
        for (int side = 0; side < 2; ++side) {   // the gate rows first, the up rows: ggml's two dots
            const block_iq3_s* x = side ? ux : gx;
            float* dst = side ? u : g;
            for (int i = 0; i < nb; ++i) {
                uint32_t scales32[2];
                std::memcpy(scales32, x[i].scales, 4);
                scales32[1] = (((scales32[0] >> 4) & 0x0f0f0f0f) << 1) | 0x01010101;
                scales32[0] = ((scales32[0] & 0x0f0f0f0f) << 1) | 0x01010101;
                const uint8_t* scales8 = (const uint8_t*) scales32;
                const uint8_t* qs = x[i].qs;
                const uint8_t* qh = x[i].qh;
                const uint16_t* signs = (const uint16_t*) x[i].signs;
                int32_t sumi1[NT], sumi2[NT];
                for (int t = 0; t < NT; ++t) { sumi1[t] = 0; sumi2[t] = 0; }
                for (int ib32 = 0; ib32 < QK_K / 32; ib32 += 2) {  // 64 values per step, the ggml order
                    const uint8x16_t idx_l = vld1q_u8(qs); qs += 16;
                    uint16x8_t il = vorrq_u16(vmovl_u8(vget_low_u8(idx_l)),
                                              vandq_u16(vshlq_u16(vdupq_n_u16(qh[ib32 + 0]), hshift), m256));
                    uint16x8_t ih = vorrq_u16(vmovl_u8(vget_high_u8(idx_l)),
                                              vandq_u16(vshlq_u16(vdupq_n_u16(qh[ib32 + 1]), hshift), m256));
                    uint16_t idx[16];
                    vst1q_u16(idx, il); vst1q_u16(idx + 8, ih);
                    const int8x16_t a0 = vreinterpretq_s8_u32(vld_grid_u32(iq3s_grid[idx[0]], iq3s_grid[idx[1]],
                                                                            iq3s_grid[idx[2]], iq3s_grid[idx[3]]));
                    const int8x16_t a1 = vreinterpretq_s8_u32(vld_grid_u32(iq3s_grid[idx[4]], iq3s_grid[idx[5]],
                                                                            iq3s_grid[idx[6]], iq3s_grid[idx[7]]));
                    const int8x16_t a2 = vreinterpretq_s8_u32(vld_grid_u32(iq3s_grid[idx[8]], iq3s_grid[idx[9]],
                                                                            iq3s_grid[idx[10]], iq3s_grid[idx[11]]));
                    const int8x16_t a3 = vreinterpretq_s8_u32(vld_grid_u32(iq3s_grid[idx[12]], iq3s_grid[idx[13]],
                                                                            iq3s_grid[idx[14]], iq3s_grid[idx[15]]));
                    // the ±1 sign vectors from the four 16-bit sign words, the ggml table trick verbatim
                    uint32_t sw[4];
                    for (int k = 0; k < 4; ++k) sw[k] = signs[k];
                    uint8x16_t vA0 = vandq_u8(vqtbl1q_u8(vreinterpretq_u8_u32(vdupq_n_u32(sw[0] | ((uint32_t)sw[1] << 16))), mask1.val[0]), mask2);
                    uint8x16_t vA1 = vandq_u8(vqtbl1q_u8(vreinterpretq_u8_u32(vdupq_n_u32(sw[0] | ((uint32_t)sw[1] << 16))), mask1.val[1]), mask2);
                    uint8x16_t vB0 = vandq_u8(vqtbl1q_u8(vreinterpretq_u8_u32(vdupq_n_u32(sw[2] | ((uint32_t)sw[3] << 16))), mask1.val[0]), mask2);
                    uint8x16_t vB1 = vandq_u8(vqtbl1q_u8(vreinterpretq_u8_u32(vdupq_n_u32(sw[2] | ((uint32_t)sw[3] << 16))), mask1.val[1]), mask2);
                    const int8x16_t sA0 = vreinterpretq_s8_u8(vorrq_u8(vceqq_u8(vA0, mask2), m1));
                    const int8x16_t sA1 = vreinterpretq_s8_u8(vorrq_u8(vceqq_u8(vA1, mask2), m1));
                    const int8x16_t sB0 = vreinterpretq_s8_u8(vorrq_u8(vceqq_u8(vB0, mask2), m1));
                    const int8x16_t sB1 = vreinterpretq_s8_u8(vorrq_u8(vceqq_u8(vB1, mask2), m1));
                    const int8x16_t q0 = vmulq_s8(sA0, a0), q1 = vmulq_s8(sA1, a1);
                    const int8x16_t q2 = vmulq_s8(sB0, a2), q3 = vmulq_s8(sB1, a3);
                    signs += 4;
                    for (int t = 0; t < NT; ++t) {
                        const s8x4 q8b = vld_s8_x4(y[t][i].qs + 64 * (ib32 / 2));
                        const int32x4_t p1 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), q0, q8b.val[0]), q1, q8b.val[1]);
                        const int32x4_t p2 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), q2, q8b.val[2]), q3, q8b.val[3]);
                        sumi1[t] += vaddvq_s32(p1) * scales8[ib32 / 2 + 0];
                        sumi2[t] += vaddvq_s32(p2) * scales8[ib32 / 2 + 4];
                    }
                }
                for (int t = 0; t < NT; ++t)
                    dst[t] += (ggml_fp16_to_fp32(x[i].d) * y[t][i].d) * (float) (sumi1[t] + sumi2[t]);
            }
        }
        for (int t = 0; t < NT; ++t)
            ff[t][r] = (g[t] / (1.f + std::exp(-g[t]))) * u[t];   // the dispatch's SwiGLU, same shape
    }
}

}  // namespace

void iq3s256_gu_rows(const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act,
                     int nt, float* const* ff, int r0, int r1) {
    const block_q8_K* y[4];
    float* outp[4];
    for (int t0 = 0; t0 < nt; t0 += 4) {
        const int m = nt - t0 >= 4 ? 4 : nt - t0;
        for (int t = 0; t < m; ++t) { y[t] = (const block_q8_K*) act[t0 + t]; outp[t] = ff[t0 + t]; }
        switch (m) {
            case 1: iq3s_gu_kernel<1>(blob, gu_row, up_off, n, y, outp, r0, r1); break;
            case 2: iq3s_gu_kernel<2>(blob, gu_row, up_off, n, y, outp, r0, r1); break;
            case 3: iq3s_gu_kernel<3>(blob, gu_row, up_off, n, y, outp, r0, r1); break;
            default: iq3s_gu_kernel<4>(blob, gu_row, up_off, n, y, outp, r0, r1); break;
        }
    }
}

}  // namespace strata::kernels::cpu
