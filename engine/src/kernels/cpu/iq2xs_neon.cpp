// src/kernels/cpu/iq2xs_neon.cpp - the IQ2_XS and IQ2_XXS gate/up rows on Arm64 with dotprod.
// ggml-cpu's ggml_vec_dot_iq2_xs_q8_K / ggml_vec_dot_iq2_xxs_q8_K (ggml-cpu/arch/arm/quants.c)
// decode for every token; here the 512-entry iq2xs_grid / 256-entry iq2xxs_grid row, the
// keven_signs_q2xs sign vector and the nibble scales are produced once per step and reused for
// all tokens of the group.  The accumulation reproduces each dot exactly - IQ2_XS an INT vmla over
// the 4-bit scales (order-free) with one float step d*vaddvq(sumi) per block and a final 0.125f;
// IQ2_XXS the dot's float shape, the 0.5f+scale float weights and the sumf1/sumf2 float order per
// block kept verbatim (float addition is not associative) - so every token's rows are bit-identical
// to the ggml dot at any nt.  Dispatched by native_expert.cpp under cpu_neon_dot_ok();
// STRATA_NO_IQ2XS falls back to ggml-cpu's dot.
#include "strata/kernels/cpu/iq_avx2.hpp"
#include "iq2_keven_signs.hpp"

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

struct s8x4 { int8x16_t val[4]; };
inline s8x4 vld_s8_x4(const int8_t* p) {
    s8x4 r;
    r.val[0] = vld1q_s8(p); r.val[1] = vld1q_s8(p + 16);
    r.val[2] = vld1q_s8(p + 32); r.val[3] = vld1q_s8(p + 48);
    return r;
}

template <int NT>
__attribute__((target("+dotprod")))
void iq2xs_gu_kernel(const uint8_t* blob, size_t gu_row, size_t up_off, int n,
                     const block_q8_K* const* y, float* const* ff, int r0, int r1) {
    const uint64_t* signs64 = (const uint64_t*) keven_signs_q2xs;
    const int nb = n / QK_K;
    float g[NT], u[NT];
    for (int r = r0; r < r1; ++r) {
        const block_iq2_xs* gx = (const block_iq2_xs*) (blob + (size_t) r * gu_row);
        const block_iq2_xs* ux = (const block_iq2_xs*) (blob + up_off + (size_t) r * gu_row);
        for (int t = 0; t < NT; ++t) { g[t] = 0.f; u[t] = 0.f; }
        for (int side = 0; side < 2; ++side) {   // the gate rows first, the up rows: ggml's two dots
            const block_iq2_xs* x = side ? ux : gx;
            float* dst = side ? u : g;
            for (int i = 0; i < nb; ++i) {
                const float d = ggml_fp16_to_fp32(x[i].d);
                const uint16_t* q2 = x[i].qs;
                // the nibble scales, expanded exactly as ggml does (token-independent)
                const uint8x8_t scales8 = vld1_u8(x[i].scales);
                const uint8x8_t slo = vand_u8(scales8, vdup_n_u8(0xf));
                const uint8x8_t shi = vshr_n_u8(scales8, 4);
                uint8x16_t scales = vcombine_u8(vzip1_u8(slo, shi), vzip2_u8(slo, shi));
                scales = vaddq_u8(vshlq_n_u8(scales, 1), vdupq_n_u8(1));
                const uint16x8_t sc1 = vmovl_u8(vget_low_u8(scales));
                const uint16x8_t sc2 = vmovl_u8(vget_high_u8(scales));
                const int32x4_t sc[4] = {
                    vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(sc1))),
                    vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(sc1))),
                    vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(sc2))),
                    vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(sc2))) };
                int32x4_t sumi[NT];
                for (int t = 0; t < NT; ++t) sumi[t] = vdupq_n_s32(0);
                for (int ib64 = 0; ib64 < QK_K / 64; ++ib64) {
                    int8x16_t q2u[4], q2s[4];
                    for (int k = 0; k < 4; ++k) {   // the grid rows and their sign vectors, decoded once
                        q2u[k] = vcombine_s8(vld1_s8((const int8_t*) (iq2xs_grid + (q2[2 * k] & 511))),
                                              vld1_s8((const int8_t*) (iq2xs_grid + (q2[2 * k + 1] & 511))));
                        q2s[k] = vcombine_s8(vld1_s8((const int8_t*) (signs64 + (q2[2 * k] >> 9))),
                                              vld1_s8((const int8_t*) (signs64 + (q2[2 * k + 1] >> 9))));
                        q2u[k] = vmulq_s8(q2u[k], q2s[k]);
                    }
                    q2 += 8;
                    for (int t = 0; t < NT; ++t) {
                        const s8x4 q8b = vld_s8_x4(y[t][i].qs + 64 * ib64);
                        const int32x4_t p1 = vdotq_s32(vdupq_n_s32(0), q2u[0], q8b.val[0]);
                        const int32x4_t p2 = vdotq_s32(vdupq_n_s32(0), q2u[1], q8b.val[1]);
                        const int32x4_t p3 = vdotq_s32(vdupq_n_s32(0), q2u[2], q8b.val[2]);
                        const int32x4_t p4 = vdotq_s32(vdupq_n_s32(0), q2u[3], q8b.val[3]);
                        const int32x4_t p = vpaddq_s32(vpaddq_s32(p1, p2), vpaddq_s32(p3, p4));
                        sumi[t] = vmlaq_s32(sumi[t], p, sc[ib64]);
                    }
                }
                for (int t = 0; t < NT; ++t)
                    dst[t] += (d * y[t][i].d) * (float) vaddvq_s32(sumi[t]);
            }
            for (int t = 0; t < NT; ++t) dst[t] *= 0.125f;   // the dot's final 0.125f
        }
        for (int t = 0; t < NT; ++t)
            ff[t][r] = (g[t] / (1.f + std::exp(-g[t]))) * u[t];   // the dispatch's SwiGLU, same shape
    }
}

template <int NT>
__attribute__((target("+dotprod")))
void iq2xxs_gu_kernel(const uint8_t* blob, size_t gu_row, size_t up_off, int n,
                      const block_q8_K* const* y, float* const* ff, int r0, int r1) {
    const uint64_t* signs64 = (const uint64_t*) keven_signs_q2xs;
    const int nb = n / QK_K;
    float g[NT], u[NT];
    for (int r = r0; r < r1; ++r) {
        const block_iq2_xxs* gx = (const block_iq2_xxs*) (blob + (size_t) r * gu_row);
        const block_iq2_xxs* ux = (const block_iq2_xxs*) (blob + up_off + (size_t) r * gu_row);
        for (int t = 0; t < NT; ++t) { g[t] = 0.f; u[t] = 0.f; }
        for (int side = 0; side < 2; ++side) {   // the gate rows first, the up rows: ggml's two dots
            const block_iq2_xxs* x = side ? ux : gx;
            float* dst = side ? u : g;
            float sumf[NT];
            for (int t = 0; t < NT; ++t) sumf[t] = 0.f;
            for (int i = 0; i < nb; ++i) {
                const uint16_t* q2 = x[i].qs;
                float sumf1[NT], sumf2[NT];
                for (int t = 0; t < NT; ++t) { sumf1[t] = 0.f; sumf2[t] = 0.f; }
                for (int ib32 = 0; ib32 < QK_K / 32; ib32 += 2) {   // 64 values per step, the ggml order
                    uint32_t aux32[4];
                    std::memcpy(aux32, q2, 4 * sizeof(uint32_t)); q2 += 8;
                    const uint8_t* aux8 = (const uint8_t*) aux32;
                    int8x16_t q2u[4];
                    q2u[0] = vcombine_s8(vld1_s8((const int8_t*) (iq2xxs_grid + aux8[0])), vld1_s8((const int8_t*) (iq2xxs_grid + aux8[1])));
                    q2u[1] = vcombine_s8(vld1_s8((const int8_t*) (iq2xxs_grid + aux8[2])), vld1_s8((const int8_t*) (iq2xxs_grid + aux8[3])));
                    q2u[2] = vcombine_s8(vld1_s8((const int8_t*) (iq2xxs_grid + aux8[8])), vld1_s8((const int8_t*) (iq2xxs_grid + aux8[9])));
                    q2u[3] = vcombine_s8(vld1_s8((const int8_t*) (iq2xxs_grid + aux8[10])), vld1_s8((const int8_t*) (iq2xxs_grid + aux8[11])));
                    q2u[0] = vmulq_s8(q2u[0], vcombine_s8(vld1_s8((const int8_t*) (signs64 + ((aux32[1] >> 0) & 127))), vld1_s8((const int8_t*) (signs64 + ((aux32[1] >> 7) & 127)))));
                    q2u[1] = vmulq_s8(q2u[1], vcombine_s8(vld1_s8((const int8_t*) (signs64 + ((aux32[1] >> 14) & 127))), vld1_s8((const int8_t*) (signs64 + ((aux32[1] >> 21) & 127)))));
                    q2u[2] = vmulq_s8(q2u[2], vcombine_s8(vld1_s8((const int8_t*) (signs64 + ((aux32[3] >> 0) & 127))), vld1_s8((const int8_t*) (signs64 + ((aux32[3] >> 7) & 127)))));
                    q2u[3] = vmulq_s8(q2u[3], vcombine_s8(vld1_s8((const int8_t*) (signs64 + ((aux32[3] >> 14) & 127))), vld1_s8((const int8_t*) (signs64 + ((aux32[3] >> 21) & 127)))));
                    const float w1 = 0.5f + (float) (aux32[1] >> 28), w2 = 0.5f + (float) (aux32[3] >> 28);
                    for (int t = 0; t < NT; ++t) {
                        const s8x4 q8b = vld_s8_x4(y[t][i].qs + 64 * (ib32 / 2));
                        const int32x4_t pa = vdotq_s32(vdotq_s32(vdupq_n_s32(0), q2u[0], q8b.val[0]), q2u[1], q8b.val[1]);
                        const int32x4_t pb = vdotq_s32(vdotq_s32(vdupq_n_s32(0), q2u[2], q8b.val[2]), q2u[3], q8b.val[3]);
                        sumf1[t] += (float) vaddvq_s32(pa) * w1;   // the dot's float accumulation, same order
                        sumf2[t] += (float) vaddvq_s32(pb) * w2;
                    }
                }
                for (int t = 0; t < NT; ++t)
                    sumf[t] += (ggml_fp16_to_fp32(x[i].d) * y[t][i].d) * (sumf1[t] + sumf2[t]);
            }
            for (int t = 0; t < NT; ++t) dst[t] = sumf[t] * 0.25f;   // the dot's final 0.25f
        }
        for (int t = 0; t < NT; ++t)
            ff[t][r] = (g[t] / (1.f + std::exp(-g[t]))) * u[t];   // the dispatch's SwiGLU, same shape
    }
}

}  // namespace

void iq2xs256_gu_rows(const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act,
                      int nt, float* const* ff, int r0, int r1) {
    const block_q8_K* y[4];
    float* outp[4];
    for (int t0 = 0; t0 < nt; t0 += 4) {
        const int m = nt - t0 >= 4 ? 4 : nt - t0;
        for (int t = 0; t < m; ++t) { y[t] = (const block_q8_K*) act[t0 + t]; outp[t] = ff[t0 + t]; }
        switch (m) {
            case 1: iq2xs_gu_kernel<1>(blob, gu_row, up_off, n, y, outp, r0, r1); break;
            case 2: iq2xs_gu_kernel<2>(blob, gu_row, up_off, n, y, outp, r0, r1); break;
            case 3: iq2xs_gu_kernel<3>(blob, gu_row, up_off, n, y, outp, r0, r1); break;
            default: iq2xs_gu_kernel<4>(blob, gu_row, up_off, n, y, outp, r0, r1); break;
        }
    }
}

void iq2xxs256_gu_rows(const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act,
                       int nt, float* const* ff, int r0, int r1) {
    const block_q8_K* y[4];
    float* outp[4];
    for (int t0 = 0; t0 < nt; t0 += 4) {
        const int m = nt - t0 >= 4 ? 4 : nt - t0;
        for (int t = 0; t < m; ++t) { y[t] = (const block_q8_K*) act[t0 + t]; outp[t] = ff[t0 + t]; }
        switch (m) {
            case 1: iq2xxs_gu_kernel<1>(blob, gu_row, up_off, n, y, outp, r0, r1); break;
            case 2: iq2xxs_gu_kernel<2>(blob, gu_row, up_off, n, y, outp, r0, r1); break;
            case 3: iq2xxs_gu_kernel<3>(blob, gu_row, up_off, n, y, outp, r0, r1); break;
            default: iq2xxs_gu_kernel<4>(blob, gu_row, up_off, n, y, outp, r0, r1); break;
        }
    }
}

}  // namespace strata::kernels::cpu
