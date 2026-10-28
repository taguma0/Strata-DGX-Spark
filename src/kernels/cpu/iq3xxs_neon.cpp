// src/kernels/cpu/iq3xxs_neon.cpp - the IQ3_XXS gate/up rows on Arm64 with dotprod: the 32-value
// decode (the iq3xxs_grid magnitudes, the keven_signs_q2xs signs, the 4-bit scales) runs once per
// verify window and every token applies it with four SDOTs; ggml-cpu's ggml_vec_dot_iq3_xxs_q8_K
// decodes for every token (ggml-cpu/arch/arm/quants.c).  The per-token accumulation follows that
// dot's order exactly - sumf1/sumf2 per 64 values, sumf += d*(sumf1+sumf2) per 256-value block,
// the final 0.5 - so every token's rows are bit-identical to the ggml dot at any nt (#152 needs
// no rule here) and to each other.  native_expert.cpp dispatches here when cpu_neon_dot_ok()
// (portable_stubs.cpp, the base-ISA probe) says this CPU has vdotq_s32; STRATA_NO_IQ3XXS falls
// back to ggml-cpu's single-token dot.
//
// The x86 maddubs path absolutises the weights (maddubs' first operand is unsigned); the signed
// SDOT takes the grid words times the signs as they are (vmulq_s8), so no |w| / sign split here.
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

// The sign table from ggml-cpu/arch/arm/quants.c, bit for bit (128 rows of 8 int8 signs, indexed
// by the 7-bit sign fields of the IQ3_XXS aux words, exactly as the ggml dot reads it).
static const int8_t keven_signs_q2xs[1024] = {
     1,  1,  1,  1,  1,  1,  1,  1, -1,  1,  1,  1,  1,  1,  1, -1,  1, -1,  1,  1,  1,  1,  1, -1, -1, -1,  1,  1,  1,  1,  1,  1,
     1,  1, -1,  1,  1,  1,  1, -1, -1,  1, -1,  1,  1,  1,  1,  1,  1, -1, -1,  1,  1,  1,  1,  1, -1, -1, -1,  1,  1,  1,  1, -1,
     1,  1,  1, -1,  1,  1,  1, -1, -1,  1,  1, -1,  1,  1,  1,  1,  1, -1,  1, -1,  1,  1,  1,  1, -1, -1,  1, -1,  1,  1,  1, -1,
     1,  1, -1, -1,  1,  1,  1,  1, -1,  1, -1, -1,  1,  1,  1, -1,  1, -1, -1, -1,  1,  1,  1, -1, -1, -1, -1, -1,  1,  1,  1,  1,
     1,  1,  1,  1, -1,  1,  1, -1, -1,  1,  1,  1, -1,  1,  1,  1,  1, -1,  1,  1, -1,  1,  1,  1, -1, -1,  1,  1, -1,  1,  1, -1,
     1,  1, -1,  1, -1,  1,  1,  1, -1,  1, -1,  1, -1,  1,  1, -1,  1, -1, -1,  1, -1,  1,  1, -1, -1, -1, -1,  1, -1,  1,  1,  1,
     1,  1,  1, -1, -1,  1,  1,  1, -1,  1,  1, -1, -1,  1,  1, -1,  1, -1,  1, -1, -1,  1,  1, -1, -1, -1,  1, -1, -1,  1,  1,  1,
     1,  1, -1, -1, -1,  1,  1, -1, -1,  1, -1, -1, -1,  1,  1,  1,  1, -1, -1, -1, -1,  1,  1,  1, -1, -1, -1, -1, -1,  1,  1, -1,
     1,  1,  1,  1,  1, -1,  1, -1, -1,  1,  1,  1,  1, -1,  1,  1,  1, -1,  1,  1,  1, -1,  1,  1, -1, -1,  1,  1,  1, -1,  1, -1,
     1,  1, -1,  1,  1, -1,  1,  1, -1,  1, -1,  1,  1, -1,  1, -1,  1, -1, -1,  1,  1, -1,  1, -1, -1, -1, -1,  1,  1, -1,  1,  1,
     1,  1,  1, -1,  1, -1,  1,  1, -1,  1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1, -1, -1,  1, -1,  1, -1,  1,  1,
     1,  1, -1, -1,  1, -1,  1, -1, -1,  1, -1, -1,  1, -1,  1,  1,  1, -1, -1, -1,  1, -1,  1,  1, -1, -1, -1, -1,  1, -1,  1, -1,
     1,  1,  1,  1, -1, -1,  1,  1, -1,  1,  1,  1, -1, -1,  1, -1,  1, -1,  1,  1, -1, -1,  1, -1, -1, -1,  1,  1, -1, -1,  1,  1,
     1,  1, -1,  1, -1, -1,  1, -1, -1,  1, -1,  1, -1, -1,  1,  1,  1, -1, -1,  1, -1, -1,  1,  1, -1, -1, -1,  1, -1, -1,  1, -1,
     1,  1,  1, -1, -1, -1,  1, -1, -1,  1,  1, -1, -1, -1,  1,  1,  1, -1,  1, -1, -1, -1,  1,  1, -1, -1,  1, -1, -1, -1,  1, -1,
     1,  1, -1, -1, -1, -1,  1,  1, -1,  1, -1, -1, -1, -1,  1, -1,  1, -1, -1, -1, -1, -1,  1, -1, -1, -1, -1, -1, -1, -1,  1,  1,
     1,  1,  1,  1,  1,  1, -1, -1, -1,  1,  1,  1,  1,  1, -1,  1,  1, -1,  1,  1,  1,  1, -1,  1, -1, -1,  1,  1,  1,  1, -1, -1,
     1,  1, -1,  1,  1,  1, -1,  1, -1,  1, -1,  1,  1,  1, -1, -1,  1, -1, -1,  1,  1,  1, -1, -1, -1, -1, -1,  1,  1,  1, -1,  1,
     1,  1,  1, -1,  1,  1, -1,  1, -1,  1,  1, -1,  1,  1, -1, -1,  1, -1,  1, -1,  1,  1, -1, -1, -1, -1,  1, -1,  1,  1, -1,  1,
     1,  1, -1, -1,  1,  1, -1, -1, -1,  1, -1, -1,  1,  1, -1,  1,  1, -1, -1, -1,  1,  1, -1,  1, -1, -1, -1, -1,  1,  1, -1, -1,
     1,  1,  1,  1, -1,  1, -1,  1, -1,  1,  1,  1, -1,  1, -1, -1,  1, -1,  1,  1, -1,  1, -1, -1, -1, -1,  1,  1, -1,  1, -1,  1,
     1,  1, -1,  1, -1,  1, -1, -1, -1,  1, -1,  1, -1,  1, -1,  1,  1, -1, -1,  1, -1,  1, -1,  1, -1, -1, -1,  1, -1,  1, -1, -1,
     1,  1,  1, -1, -1,  1, -1, -1, -1,  1,  1, -1, -1,  1, -1,  1,  1, -1,  1, -1, -1,  1, -1,  1, -1, -1,  1, -1, -1,  1, -1, -1,
     1,  1, -1, -1, -1,  1, -1,  1, -1,  1, -1, -1, -1,  1, -1, -1,  1, -1, -1, -1, -1,  1, -1, -1, -1, -1, -1, -1, -1,  1, -1,  1,
     1,  1,  1,  1,  1, -1, -1,  1, -1,  1,  1,  1,  1, -1, -1, -1,  1, -1,  1,  1,  1, -1, -1, -1, -1, -1,  1,  1,  1, -1, -1,  1,
     1,  1, -1,  1,  1, -1, -1, -1, -1,  1, -1,  1,  1, -1, -1,  1,  1, -1, -1,  1,  1, -1, -1,  1, -1, -1, -1,  1,  1, -1, -1, -1,
     1,  1,  1, -1,  1, -1, -1, -1, -1,  1,  1, -1,  1, -1, -1,  1,  1, -1,  1, -1,  1, -1, -1,  1, -1, -1,  1, -1,  1, -1, -1, -1,
     1,  1, -1, -1,  1, -1, -1,  1, -1,  1, -1, -1,  1, -1, -1, -1,  1, -1, -1, -1,  1, -1, -1, -1, -1, -1, -1, -1,  1, -1, -1,  1,
     1,  1,  1,  1, -1, -1, -1, -1, -1,  1,  1,  1, -1, -1, -1,  1,  1, -1,  1,  1, -1, -1, -1,  1, -1, -1,  1,  1, -1, -1, -1, -1,
     1,  1, -1,  1, -1, -1, -1,  1, -1,  1, -1,  1, -1, -1, -1, -1,  1, -1, -1,  1, -1, -1, -1, -1, -1, -1, -1,  1, -1, -1, -1,  1,
     1,  1,  1, -1, -1, -1, -1,  1, -1,  1,  1, -1, -1, -1, -1, -1,  1, -1,  1, -1, -1, -1, -1, -1, -1, -1,  1, -1, -1, -1, -1,  1,
     1,  1, -1, -1, -1, -1, -1, -1, -1,  1, -1, -1, -1, -1, -1,  1,  1, -1, -1, -1, -1, -1, -1,  1, -1, -1, -1, -1, -1, -1, -1, -1,
};

// ggml-common.h's ggml_vld1q_u32 is the C section's macro; the C++ path loads the four grid words from
// a stack quad, the same four u32 into the same vector.
inline uint32x4_t vld_grid_u32(uint32_t w, uint32_t x, uint32_t y, uint32_t z) {
    const uint32_t v[4] = { w, x, y, z };
    return vld1q_u32(v);
}

template <int NT>
__attribute__((target("+dotprod")))
void iq3xxs_gu_kernel(const uint8_t* blob, size_t gu_row, size_t up_off, int n,
                      const block_q8_K* const* y, float* const* ff, int r0, int r1) {
    const uint64_t* signs64 = (const uint64_t*) keven_signs_q2xs;
    const int nb = n / QK_K;
    float g[NT], u[NT];
    for (int r = r0; r < r1; ++r) {
        const block_iq3_xxs* gx = (const block_iq3_xxs*) (blob + (size_t) r * gu_row);
        const block_iq3_xxs* ux = (const block_iq3_xxs*) (blob + up_off + (size_t) r * gu_row);
        for (int t = 0; t < NT; ++t) { g[t] = 0.f; u[t] = 0.f; }
        for (int side = 0; side < 2; ++side) {   // the gate rows first, the up rows: ggml's two dots
            const block_iq3_xxs* x = side ? ux : gx;
            float* dst = side ? u : g;
            for (int i = 0; i < nb; ++i) {
                const uint8_t* q3 = x[i].qs;
                const uint8_t* gas = x[i].qs + QK_K / 4;
                const float dx = ggml_fp16_to_fp32(x[i].d);
                float sf1[NT], sf2[NT];
                for (int t = 0; t < NT; ++t) { sf1[t] = 0.f; sf2[t] = 0.f; }
                for (int ib32 = 0; ib32 < QK_K / 32; ib32 += 2) {   // 64 values per step
                    uint32_t aux32[2];
                    std::memcpy(aux32, gas, 2 * sizeof(uint32_t));
                    gas += 2 * sizeof(uint32_t);
                    // one signed grid vector per 16 values, shared by every token
                    const int8x16_t q3s0 = vmulq_s8(
                        vcombine_s8(vld1_s8((const int8_t*) (signs64 + ((aux32[0] >>  0) & 127))),
                                    vld1_s8((const int8_t*) (signs64 + ((aux32[0] >>  7) & 127)))),
                        vreinterpretq_s8_u32(vld_grid_u32(iq3xxs_grid[q3[0]], iq3xxs_grid[q3[1]],
                                                             iq3xxs_grid[q3[2]], iq3xxs_grid[q3[3]])));
                    const int8x16_t q3s1 = vmulq_s8(
                        vcombine_s8(vld1_s8((const int8_t*) (signs64 + ((aux32[0] >> 14) & 127))),
                                    vld1_s8((const int8_t*) (signs64 + ((aux32[0] >> 21) & 127)))),
                        vreinterpretq_s8_u32(vld_grid_u32(iq3xxs_grid[q3[4]], iq3xxs_grid[q3[5]],
                                                             iq3xxs_grid[q3[6]], iq3xxs_grid[q3[7]])));
                    const int8x16_t q3s2 = vmulq_s8(
                        vcombine_s8(vld1_s8((const int8_t*) (signs64 + ((aux32[1] >>  0) & 127))),
                                    vld1_s8((const int8_t*) (signs64 + ((aux32[1] >>  7) & 127)))),
                        vreinterpretq_s8_u32(vld_grid_u32(iq3xxs_grid[q3[8]], iq3xxs_grid[q3[9]],
                                                             iq3xxs_grid[q3[10]], iq3xxs_grid[q3[11]])));
                    const int8x16_t q3s3 = vmulq_s8(
                        vcombine_s8(vld1_s8((const int8_t*) (signs64 + ((aux32[1] >> 14) & 127))),
                                    vld1_s8((const int8_t*) (signs64 + ((aux32[1] >> 21) & 127)))),
                        vreinterpretq_s8_u32(vld_grid_u32(iq3xxs_grid[q3[12]], iq3xxs_grid[q3[13]],
                                                             iq3xxs_grid[q3[14]], iq3xxs_grid[q3[15]])));
                    q3 += 16;
                    const float sc1 = 0.5f + (aux32[0] >> 28);
                    const float sc2 = 0.5f + (aux32[1] >> 28);
                    for (int t = 0; t < NT; ++t) {
                        const int8_t* q8 = y[t][i].qs + 32 * ib32;
                        const int8x16_t qa = vld1q_s8(q8), qb = vld1q_s8(q8 + 16);
                        const int8x16_t qc = vld1q_s8(q8 + 32), qd = vld1q_s8(q8 + 48);
                        const int32x4_t p1 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), q3s0, qa), q3s1, qb);
                        const int32x4_t p2 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), q3s2, qc), q3s3, qd);
                        sf1[t] += (float) vaddvq_s32(p1) * sc1;
                        sf2[t] += (float) vaddvq_s32(p2) * sc2;
                    }
                }
                for (int t = 0; t < NT; ++t)
                    dst[t] += (dx * y[t][i].d) * (sf1[t] + sf2[t]);   // ggml: sumf += d*(sumf1+sumf2)
            }
        }
        for (int t = 0; t < NT; ++t) {
            g[t] *= 0.5f;   // ggml's final 0.5 per dot
            u[t] *= 0.5f;
            ff[t][r] = (g[t] / (1.f + std::exp(-g[t]))) * u[t];   // the dispatch's SwiGLU, same shape
        }
    }
}

}  // namespace

void iq3xxs256_gu_rows(const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act,
                       int nt, float* const* ff, int r0, int r1) {
    const block_q8_K* y[4];
    float* outp[4];
    for (int t0 = 0; t0 < nt; t0 += 4) {
        const int m = nt - t0 >= 4 ? 4 : nt - t0;
        for (int t = 0; t < m; ++t) { y[t] = (const block_q8_K*) act[t0 + t]; outp[t] = ff[t0 + t]; }
        switch (m) {
            case 1: iq3xxs_gu_kernel<1>(blob, gu_row, up_off, n, y, outp, r0, r1); break;
            case 2: iq3xxs_gu_kernel<2>(blob, gu_row, up_off, n, y, outp, r0, r1); break;
            case 3: iq3xxs_gu_kernel<3>(blob, gu_row, up_off, n, y, outp, r0, r1); break;
            default: iq3xxs_gu_kernel<4>(blob, gu_row, up_off, n, y, outp, r0, r1); break;
        }
    }
}

}  // namespace strata::kernels::cpu
