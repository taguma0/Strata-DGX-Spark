// src/kernels/cpu/iq4nl_neon.cpp - the IQ4_NL down rows on Arm64 with dotprod: the one-token kernel's
// per-block decode (the 16-entry kvalues_iq4nl table through TBL, the dot through SDOT) done once per
// block and shared by nt tokens; ggml-cpu's ggml_vec_dot_iq4_nl_q8_0 decodes for every token
// (arch/arm/quants.c). The rows are bit-identical to that ggml dot - the same block order, the same
// pair-then-tail float shape, the same table - and so to each other at every nt (#152 needs no rule
// here: QK4_NL == QK8_0, the scales pair one-to-one and there is no group to straddle).
// native_expert.cpp dispatches here when cpu_neon_dot_ok() (portable_stubs.cpp, the base-ISA probe)
// says this CPU has vdotq_s32; STRATA_NO_IQ4NL falls back to ggml-cpu's single-token dot.
//
// QK4_NL == QK8_0 == 32: a block's 16 nibble bytes decode to 32 int8 weights (2 TBL + and/shift per
// 16 bytes) and pair with the token's 32 activation bytes: 2 SDOTs per block per token. The x86
// maddubs path absolutises the weights (maddubs' first operand is unsigned); the signed SDOT takes
// the table's int8 codes as they are, so the |w| / sign split is not needed.
#include "strata/kernels/cpu/iq_avx2.hpp"

#include "ggml.h"

#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"

#include <arm_neon.h>

#include <cstdint>

namespace strata::kernels::cpu {
namespace {

// A lane group of up to 4 tokens: each token's row is its own accumulation in the same order, so
// tokens past the group run their own group and stay bit-identical.
template <int NT>
__attribute__((target("+dotprod")))
void iq4nl_neon_rows(const uint8_t* w, size_t row_bytes, int n, const block_q8_0* const* y,
                     float* const* out, int r0, int r1) {
    const int8x16_t values = vld1q_s8(kvalues_iq4nl);
    const uint8x16_t m4b = vdupq_n_u8(0x0f);
    const int nb = n / QK4_NL;
    for (int r = r0; r < r1; ++r) {
        const block_iq4_nl* x = (const block_iq4_nl*) (w + (size_t) r * row_bytes);
        float sumf[NT];
        for (int t = 0; t < NT; ++t) sumf[t] = 0.f;
        int ib = 0;
        for (; ib + 1 < nb; ib += 2) {
            const uint8x16_t q4a = vld1q_u8(x[ib + 0].qs);
            const uint8x16_t q4b = vld1q_u8(x[ib + 1].qs);
            const int8x16_t lo0 = vqtbl1q_s8(values, vandq_u8(q4a, m4b));
            const int8x16_t hi0 = vqtbl1q_s8(values, vshrq_n_u8(q4a, 4));
            const int8x16_t lo1 = vqtbl1q_s8(values, vandq_u8(q4b, m4b));
            const int8x16_t hi1 = vqtbl1q_s8(values, vshrq_n_u8(q4b, 4));
            const float d0 = ggml_fp16_to_fp32(x[ib + 0].d);
            const float d1 = ggml_fp16_to_fp32(x[ib + 1].d);
            for (int t = 0; t < NT; ++t) {
                const block_q8_0* yb = &y[t][ib];
                const int32x4_t p0 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), lo0, vld1q_s8(yb[0].qs)),
                                               hi0, vld1q_s8(yb[0].qs + 16));
                const int32x4_t p1 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), lo1, vld1q_s8(yb[1].qs)),
                                               hi1, vld1q_s8(yb[1].qs + 16));
                sumf[t] += d0 * ggml_fp16_to_fp32(yb[0].d) * (float) vaddvq_s32(p0) +
                           d1 * ggml_fp16_to_fp32(yb[1].d) * (float) vaddvq_s32(p1);
            }
        }
        if (ib < nb) {   // ggml's own tail for a last odd block: the y scale first, sumi1 + sumi2 as one
            const block_iq4_nl* xb = &x[ib];
            for (int t = 0; t < NT; ++t) {
                const block_q8_0* yb = &y[t][ib];
                const float d = ggml_fp16_to_fp32(yb->d) * ggml_fp16_to_fp32(xb->d);
                int sumi1 = 0, sumi2 = 0;
                for (int j = 0; j < QK4_NL / 2; ++j) {
                    sumi1 += yb->qs[j] * kvalues_iq4nl[xb->qs[j] & 0xf];
                    sumi2 += yb->qs[j + QK4_NL / 2] * kvalues_iq4nl[xb->qs[j] >> 4];
                }
                sumf[t] += d * (float) (sumi1 + sumi2);
            }
        }
        for (int t = 0; t < NT; ++t) out[t][r] = sumf[t];
    }
}

}  // namespace

void iq4nl256_down_rows(const uint8_t* w, size_t row_bytes, int n, const void* const* hq, int nt,
                        float* const* out, int r0, int r1) {
    const block_q8_0* y[4];
    float* outp[4];
    for (int t0 = 0; t0 < nt; t0 += 4) {
        const int m = nt - t0 >= 4 ? 4 : nt - t0;
        for (int t = 0; t < m; ++t) { y[t] = (const block_q8_0*) hq[t0 + t]; outp[t] = out[t0 + t]; }
        switch (m) {
            case 1: iq4nl_neon_rows<1>(w, row_bytes, n, y, outp, r0, r1); break;
            case 2: iq4nl_neon_rows<2>(w, row_bytes, n, y, outp, r0, r1); break;
            case 3: iq4nl_neon_rows<3>(w, row_bytes, n, y, outp, r0, r1); break;
            default: iq4nl_neon_rows<4>(w, row_bytes, n, y, outp, r0, r1); break;
        }
    }
}

}  // namespace strata::kernels::cpu
