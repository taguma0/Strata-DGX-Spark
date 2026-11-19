// src/kernels/cpu/iq3xxs_neon_parity.cpp - the Arm64 SDOT IQ3_XXS gate/up kernel (iq3xxs_neon.cpp)
// against ggml-cpu's own ggml_vec_dot_iq3_xxs_q8_K + the dispatch's SwiGLU, BIT FOR BIT.  No GPU,
// no model: random blocks (every grid index, sign pattern and scale nibble occurs) and random
// activations quantized with ggml's q8_K.
//
//     iq3xxs_neon_parity                     nt = 1..8, every row bit-identical to the ggml dot
//     iq3xxs_neon_parity --bench [--mb 64]   ms per expert on one thread: ggml's two per-token dots
//                                            vs the SDOT kernel over gate+up blobs of ~--mb MB
//
// Exits 77 (the repo's ctest skip code) on a CPU without dotprod: the harness calls the kernel
// directly, past the engine's cpu_neon_dot_ok() dispatch gate.
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/iq_avx2.hpp"

#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#if !defined(_WIN32)
#include <sched.h>
#endif

namespace cpu = strata::kernels::cpu;
using Clock = std::chrono::steady_clock;

namespace {

constexpr int kN = 2560;       // n_embd (10 blocks of QK_K)
constexpr int kFF = 640;       // gate/up rows per expert (n_ff)
constexpr int kBS = 98;        // sizeof(block_iq3_xxs)
constexpr int kMaxT = 8;

uint64_t mix(uint64_t& s) {   // splitmix64
    uint64_t z = (s += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

// random bytes (all grid / sign / scale patterns), then a sane fp16 scale at the front of every block
void fill_blocks(uint8_t* p, size_t bytes, uint64_t& seed) {
    for (size_t i = 0; i < bytes; i += 8) {
        const uint64_t v = mix(seed);
        std::memcpy(p + i, &v, std::min<size_t>(8, bytes - i));
    }
    for (size_t o = 0; o + kBS <= bytes; o += kBS) {
        const float d = (0.5f + (float) (mix(seed) >> 40) * (1.f / 16777216.f)) * 1e-3f;
        const ggml_fp16_t h = ggml_fp32_to_fp16(d);
        std::memcpy(p + o, &h, 2);
    }
}

std::vector<uint8_t> q8k_of(const std::vector<float>& h) {
    std::vector<uint8_t> q((size_t) (h.size() / 256) * (4 + 256 + 2 * 16));
    ggml_get_type_traits_cpu(GGML_TYPE_Q8_K)->from_float(h.data(), q.data(), (int64_t) h.size());
    return q;
}

int failures = 0;

void run_checks() {
    const size_t row_bytes = (size_t) (kN / 256) * kBS;
    const size_t gu_bytes = (size_t) kFF * row_bytes;
    uint64_t seed = 0x1C33A5EDull;
    std::vector<uint8_t> blob(2 * gu_bytes);
    fill_blocks(blob.data(), blob.size(), seed);
    std::vector<std::vector<uint8_t>> act(kMaxT);
    for (int t = 0; t < kMaxT; ++t) {
        std::vector<float> x(kN);
        for (int i = 0; i < kN; ++i) x[i] = (float) ((mix(seed) >> 40) * 0x1p-24 * 2.0 - 1.0);
        act[t] = q8k_of(x);
    }
    const void* actp[kMaxT];
    std::vector<float> out[kMaxT];
    float* outp[kMaxT];
    for (int t = 0; t < kMaxT; ++t) { actp[t] = act[t].data(); out[t].assign(kFF, 0.f); outp[t] = out[t].data(); }

    const ggml_vec_dot_t dn = ggml_get_type_traits_cpu(GGML_TYPE_IQ3_XXS)->vec_dot;
    for (int nt = 1; nt <= kMaxT; ++nt) {
        cpu::iq3xxs256_gu_rows(blob.data(), row_bytes, gu_bytes, kN, actp, nt, outp, 0, kFF);
        for (int t = 0; t < nt; ++t) {
            size_t d = 0;
            for (int r = 0; r < kFF; ++r) {
                float g = 0.f, u = 0.f;
                dn(kN, &g, 0, blob.data() + (size_t) r * row_bytes, 0, actp[t], 0, 1);
                dn(kN, &u, 0, blob.data() + gu_bytes + (size_t) r * row_bytes, 0, actp[t], 0, 1);
                const float ref = (g / (1.f + std::exp(-g))) * u;
                d += std::memcmp(&out[t][r], &ref, sizeof(float)) != 0;
            }
            if (d) {
                std::printf("  FAIL nt=%d token %d: %zu of %d rows differ bit-for-bit vs ggml\n",
                            nt, t, d, kFF);
                ++failures;
            }
        }
        std::printf("iq3xxs_neon_parity: nt=1..%d: %d gate/up rows checked against ggml_vec_dot_iq3_xxs_q8_K\n",
                    kMaxT, kFF);
    }
}

void run_bench(int mb) {
#if !defined(_WIN32)
    cpu_set_t s;
    CPU_ZERO(&s);
    CPU_SET(0, &s);
    sched_setaffinity(0, sizeof s, &s);
#endif
    const size_t row_bytes = (size_t) (kN / 256) * kBS;
    const int rows = (int) ((size_t) mb * 1024 * 1024 / 2 / row_bytes);
    const size_t gu_bytes = (size_t) rows * row_bytes;
    uint64_t seed = 0xB105ull;
    std::vector<uint8_t> blob(2 * gu_bytes);
    fill_blocks(blob.data(), blob.size(), seed);
    std::vector<std::vector<uint8_t>> act(kMaxT);
    for (int t = 0; t < kMaxT; ++t) {
        std::vector<float> x(kN);
        for (int i = 0; i < kN; ++i) x[i] = (float) ((mix(seed) >> 40) * 0x1p-24 * 2.0 - 1.0);
        act[t] = q8k_of(x);
    }
    const void* actp[kMaxT];
    std::vector<float> out[kMaxT];
    float* outp[kMaxT];
    for (int t = 0; t < kMaxT; ++t) { actp[t] = act[t].data(); out[t].assign(rows, 0.f); outp[t] = out[t].data(); }
    const ggml_vec_dot_t dn = ggml_get_type_traits_cpu(GGML_TYPE_IQ3_XXS)->vec_dot;
    std::printf("iq3xxs_neon bench: %d gate/up rows x n=%d, one thread, %d MB of gate+up blobs\n", rows, kN, mb);
    for (const int nt : {1, 2, 4, 8}) {
        double ms_g = 1e300, ms_k = 1e300;
        for (int rep = 0; rep < 3; ++rep) {
            auto t0 = Clock::now();
            for (int r = 0; r < rows; ++r)
                for (int t = 0; t < nt; ++t) {
                    float g = 0.f, u = 0.f;
                    dn(kN, &g, 0, blob.data() + (size_t) r * row_bytes, 0, actp[t], 0, 1);
                    dn(kN, &u, 0, blob.data() + gu_bytes + (size_t) r * row_bytes, 0, actp[t], 0, 1);
                    out[t][r] = (g / (1.f + std::exp(-g))) * u;
                }
            auto t1 = Clock::now();
            if (rep) ms_g = std::min(ms_g, std::chrono::duration<double, std::milli>(t1 - t0).count());
            t0 = Clock::now();
            cpu::iq3xxs256_gu_rows(blob.data(), row_bytes, gu_bytes, kN, actp, nt, outp, 0, rows);
            t1 = Clock::now();
            if (rep) ms_k = std::min(ms_k, std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
        std::printf("  nt=%d  ggml %8.2f ms  sdot %8.2f ms  %.2fx\n", nt, ms_g, ms_k, ms_g / ms_k);
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (!cpu::cpu_neon_dot_ok()) {
        std::printf("iq3xxs_neon_parity: no dotprod (HWCAP_ASIMDDOT) on this CPU: skip\n");
        return 77;
    }
    int mb = 64;
    bool bench = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--bench")) bench = true;
        else if (i + 1 < argc && !std::strcmp(argv[i], "--mb")) mb = std::atoi(argv[++i]);
    }
    if (bench) { run_bench(mb); return 0; }
    run_checks();
    if (failures) { std::printf("iq3xxs_neon_parity: %d FAILURES\n", failures); return 1; }
    std::printf("iq3xxs_neon_parity: all rows bit-identical to ggml-cpu's dot (dotprod: yes)\n");
    return 0;
}
