// src/kernels/cpu/iq2s_neon.cpp - the IQ2_S gate/up rows on Arm64 with dotprod: the 8-value grid
// decode (iq2s_grid, the 2 high bits from qh, the sign words in the upper half of qs), the nibble
// scales, four SDOTs per token per step; ggml-cpu's ggml_vec_dot_iq2_s_q8_K (ggml-cpu/arch/arm/
// quants.c) decodes for every token.  The accumulation reproduces that dot exactly: per-block INT
// sums sumi1/sumi2 (order-free), one float step d*(sumi1+sumi2) per block, the 0.125f applied once
// per side - every token's rows bit-identical to the ggml dot at any nt.  Dispatched by
// native_expert.cpp under cpu_neon_dot_ok(); STRATA_NO_IQ2S falls back to ggml-cpu's dot.
#include "strata/kernels/cpu/iq_avx2.hpp"

#include "ggml.h"

#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"

#include <arm_neon.h>

#include <cmath>
#include <cstdint>

namespace strata::kernels::cpu {
namespace {

struct s8x4 { int8x16_t val[4]; };
inline s8x4 vld_s8_x4(const int8_t* p) {
    s8x4 r;
    r.val[0] = vld1q_s8(p); r.val[1] = vld1q_s8(p + 16);
    r.val[2] = vld1q_s8(p + 32); r.val[3] = vld1q_s8(p + 48);
    return r;
}

template <int NT>
__attribute__((target("+dotprod")))
void iq2s_gu_kernel(const uint8_t* blob, size_t gu_row, size_t up_off, int n,
                    const block_q8_K* const* y, float* const* ff, int r0, int r1) {
    static const uint8_t k_mask1[32] = { 0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1, 2,2,2,2,2,2,2,2, 3,3,3,3,3,3,3,3 };
    static const uint8_t k_mask2[16] = { 0x01,0x02,0x04,0x08,0x10,0x20,0x40,0x80, 0x01,0x02,0x04,0x08,0x10,0x20,0x40,0x80 };
    const uint8x16_t m1v0 = vld1q_u8(k_mask1), m1v1 = vld1q_u8(k_mask1 + 16);
    const uint8x16_t mask2 = vld1q_u8(k_mask2);
    const uint8x16_t m1 = vdupq_n_u8(1);

    const int nb = n / QK_K;
    float g[NT], u[NT];
    for (int r = r0; r < r1; ++r) {
        const block_iq2_s* gx = (const block_iq2_s*) (blob + (size_t) r * gu_row);
        const block_iq2_s* ux = (const block_iq2_s*) (blob + up_off + (size_t) r * gu_row);
        for (int t = 0; t < NT; ++t) { g[t] = 0.f; u[t] = 0.f; }
        for (int side = 0; side < 2; ++side) {   // the gate rows first, the up rows: ggml's two dots
            const block_iq2_s* x = side ? ux : gx;
            float* dst = side ? u : g;
            for (int i = 0; i < nb; ++i) {
                const uint8_t* qs = x[i].qs;
                const uint8_t* qh = x[i].qh;
                const uint16_t* signs = (const uint16_t*) (x[i].qs + QK_K / 8);
                int32_t sumi1[NT], sumi2[NT];
                for (int t = 0; t < NT; ++t) { sumi1[t] = 0; sumi2[t] = 0; }
                for (int ib32 = 0; ib32 < QK_K / 32; ib32 += 2) {  // 64 values per step, the ggml order
                    // q2s.val[k] = two 8-value iq2s_grid rows, the ggml combines verbatim
                    int8x16_t q0 = vcombine_s8(vld1_s8((const int8_t*) (iq2s_grid + (qs[0] | ((qh[ib32 + 0] << 8) & 0x300)))),
                                                vld1_s8((const int8_t*) (iq2s_grid + (qs[1] | ((qh[ib32 + 0] << 6) & 0x300)))));
                    int8x16_t q1 = vcombine_s8(vld1_s8((const int8_t*) (iq2s_grid + (qs[2] | ((qh[ib32 + 0] << 4) & 0x300)))),
                                                vld1_s8((const int8_t*) (iq2s_grid + (qs[3] | ((qh[ib32 + 0] << 2) & 0x300)))));
                    int8x16_t q2 = vcombine_s8(vld1_s8((const int8_t*) (iq2s_grid + (qs[4] | ((qh[ib32 + 1] << 8) & 0x300)))),
                                                vld1_s8((const int8_t*) (iq2s_grid + (qs[5] | ((qh[ib32 + 1] << 6) & 0x300)))));
                    int8x16_t q3 = vcombine_s8(vld1_s8((const int8_t*) (iq2s_grid + (qs[6] | ((qh[ib32 + 1] << 4) & 0x300)))),
                                                vld1_s8((const int8_t*) (iq2s_grid + (qs[7] | ((qh[ib32 + 1] << 2) & 0x300)))));
                    qs += 8;
                    // the sign words in the upper half of qs: the ggml table trick verbatim
                    uint8x16_t v0 = vreinterpretq_u8_u32(vdupq_n_u32(signs[0] | ((uint32_t) signs[1] << 16)));
                    uint8x16_t v1 = vandq_u8(vqtbl1q_u8(v0, m1v1), mask2);
                    v0 = vandq_u8(vqtbl1q_u8(v0, m1v0), mask2);
                    q0 = vmulq_s8(vreinterpretq_s8_u8(vorrq_u8(vceqq_u8(v0, mask2), m1)), q0);
                    q1 = vmulq_s8(vreinterpretq_s8_u8(vorrq_u8(vceqq_u8(v1, mask2), m1)), q1);
                    v0 = vreinterpretq_u8_u32(vdupq_n_u32(signs[2] | ((uint32_t) signs[3] << 16)));
                    v1 = vandq_u8(vqtbl1q_u8(v0, m1v1), mask2);
                    v0 = vandq_u8(vqtbl1q_u8(v0, m1v0), mask2);
                    q2 = vmulq_s8(vreinterpretq_s8_u8(vorrq_u8(vceqq_u8(v0, mask2), m1)), q2);
                    q3 = vmulq_s8(vreinterpretq_s8_u8(vorrq_u8(vceqq_u8(v1, mask2), m1)), q3);
                    signs += 4;
                    const int s0lo = 1 + 2 * (x[i].scales[ib32 + 0] & 0xf), s0hi = 1 + 2 * (x[i].scales[ib32 + 0] >> 4);
                    const int s1lo = 1 + 2 * (x[i].scales[ib32 + 1] & 0xf), s1hi = 1 + 2 * (x[i].scales[ib32 + 1] >> 4);
                    for (int t = 0; t < NT; ++t) {
                        const s8x4 q8b = vld_s8_x4(y[t][i].qs + 64 * (ib32 / 2));
                        sumi1[t] += vaddvq_s32(vdotq_s32(vdupq_n_s32(0), q0, q8b.val[0])) * s0lo;
                        sumi2[t] += vaddvq_s32(vdotq_s32(vdupq_n_s32(0), q1, q8b.val[1])) * s0hi;
                        sumi1[t] += vaddvq_s32(vdotq_s32(vdupq_n_s32(0), q2, q8b.val[2])) * s1lo;
                        sumi2[t] += vaddvq_s32(vdotq_s32(vdupq_n_s32(0), q3, q8b.val[3])) * s1hi;
                    }
                }
                const float df = ggml_fp16_to_fp32(x[i].d);
                for (int t = 0; t < NT; ++t)
                    dst[t] += (df * y[t][i].d) * (float) (sumi1[t] + sumi2[t]);
            }
            for (int t = 0; t < NT; ++t) dst[t] *= 0.125f;   // the dot's final 0.125f
        }
        for (int t = 0; t < NT; ++t)
            ff[t][r] = (g[t] / (1.f + std::exp(-g[t]))) * u[t];   // the dispatch's SwiGLU, same shape
    }
}

}  // namespace

void iq2s256_gu_rows(const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act,
                     int nt, float* const* ff, int r0, int r1) {
    const block_q8_K* y[4];
    float* outp[4];
    for (int t0 = 0; t0 < nt; t0 += 4) {
        const int m = nt - t0 >= 4 ? 4 : nt - t0;
        for (int t = 0; t < m; ++t) { y[t] = (const block_q8_K*) act[t0 + t]; outp[t] = ff[t0 + t]; }
        switch (m) {
            case 1: iq2s_gu_kernel<1>(blob, gu_row, up_off, n, y, outp, r0, r1); break;
            case 2: iq2s_gu_kernel<2>(blob, gu_row, up_off, n, y, outp, r0, r1); break;
            case 3: iq2s_gu_kernel<3>(blob, gu_row, up_off, n, y, outp, r0, r1); break;
            default: iq2s_gu_kernel<4>(blob, gu_row, up_off, n, y, outp, r0, r1); break;
        }
    }
}

}  // namespace strata::kernels::cpu
