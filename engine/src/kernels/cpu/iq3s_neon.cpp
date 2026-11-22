// src/kernels/cpu/iq3s_neon.cpp - the IQ3_S gate/up rows on Arm64 (types 21).  WIP PHASE-1 SCAFFOLD:
// the frame (row walk, NT batching, the SwiGLU of the dispatch's fallback) is the iq3xxs_neon.cpp
// one, but the inner product is STILL ggml's own per-token vec_dot - it exists to bring up the
// build wiring, the parity harness and the blob/activation plumbing with a bit-exact reference.
// PHASE 2 replaces the inner loop with the decode-then-SDOT form (the iq3s_grid magnitudes, the
// 2-bit qs low parts, the signs byte's sign/dmin fields, ggml_vec_dot_iq3_s_q8_K's float shape
// reproduced in its exact order) so every token's rows stay bit-identical to the ggml dot at any
// nt.  native_expert.cpp will gate this on cpu_neon_dot_ok(); STRATA_NO_IQ3S falls back to ggml.
#include "strata/kernels/cpu/iq_avx2.hpp"

#include "ggml.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdint>

namespace strata::kernels::cpu {

void iq3s256_gu_rows(const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act,
                     int nt, float* const* ff, int r0, int r1) {
    const ggml_vec_dot_t dn = ggml_get_type_traits_cpu(GGML_TYPE_IQ3_S)->vec_dot;  // TEMP: Phase 2 -> SDOT
    for (int r = r0; r < r1; ++r) {
        const uint8_t* gx = blob + (size_t) r * gu_row;
        const uint8_t* ux = blob + up_off + (size_t) r * gu_row;
        for (int t = 0; t < nt; ++t) {
            float g = 0.f, u = 0.f;
            dn(n, &g, 0, gx, 0, act[t], 0, 1);
            dn(n, &u, 0, ux, 0, act[t], 0, 1);
            ff[t][r] = (g / (1.f + std::exp(-g))) * u;   // the dispatch's SwiGLU, same shape
        }
    }
}

}  // namespace strata::kernels::cpu
