// src/kernels/cpu/iq4nl_neon_parity.cpp - the Arm64 SDOT IQ4_NL down-row kernel (iq4nl_neon.cpp) against
// ggml-cpu's own ggml_vec_dot_iq4_nl_q8_0, BIT FOR BIT.  No GPU, no model: random blocks (every grid
// index and sign pattern occurs) and random activations quantized the way the engine quantizes them.
//
//     iq4nl_neon_parity                     nt = 1..8, every row bit-identical to the ggml dot
//     iq4nl_neon_parity --bench [--mb 128]  ms per expert on one thread: ggml's per-token dot vs the
//                                           SDOT kernel over --mb MB of down blobs (weights stream
//                                           from DRAM as in decode)
//
// Exits 77 (the repo's ctest skip code) on a CPU without dotprod: the harness calls the kernel
// directly, past the engine's cpu_neon_dot_ok() dispatch gate.
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/iq_avx2.hpp"

#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <chrono>
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

constexpr int kQK = 32;        // QK4_NL == QK8_0 == 32 (ggml-common.h)
constexpr int kBS = 18;        // sizeof(block_iq4_nl): fp16 scale + 16 nibble bytes
constexpr int kRows = 2560;    // down rows per expert (n_embd), as iq_avx1_parity's kH
constexpr int kMaxT = 8;

uint64_t mix(uint64_t& s) {   // splitmix64
    uint64_t z = (s += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

// random bytes, then a sane fp16 scale at the front of every block (iq_avx1_parity's)
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

std::vector<uint8_t> q8_of(const std::vector<float>& h) {
    std::vector<uint8_t> q((size_t) (h.size() / kQK) * (2 + 32));
    ggml_get_type_traits_cpu(GGML_TYPE_Q8_0)->from_float(h.data(), q.data(), (int64_t) h.size());
    return q;
}

int failures = 0;

void run_checks() {
    const int kFFs[] = {640, 672};   // 20 blocks: whole pairs; and 21: pairs + ggml's tail block
    const ggml_vec_dot_t dn = ggml_get_type_traits_cpu(GGML_TYPE_IQ4_NL)->vec_dot;
    for (const int ff : kFFs) {
        uint64_t seed = 0x5EED1234u + (uint64_t) ff;
        const size_t row_bytes = (size_t) (ff / kQK) * kBS;
        std::vector<uint8_t> blob((size_t) kRows * row_bytes);
        fill_blocks(blob.data(), blob.size(), seed);
        std::vector<std::vector<uint8_t>> hq(kMaxT);
        for (int t = 0; t < kMaxT; ++t) {
            std::vector<float> x(ff);
            for (int i = 0; i < ff; ++i) x[i] = (float) ((mix(seed) >> 40) * 0x1p-24 * 2.0 - 1.0);
            hq[t] = q8_of(x);
        }
        const void* hqp[kMaxT];
        std::vector<float> out[kMaxT];
        float* outp[kMaxT];
        for (int t = 0; t < kMaxT; ++t) { hqp[t] = hq[t].data(); out[t].assign(kRows, 0.f); outp[t] = out[t].data(); }
        for (int nt = 1; nt <= kMaxT; ++nt) {
            cpu::iq4nl256_down_rows(blob.data(), row_bytes, ff, hqp, nt, outp, 0, kRows);
            for (int t = 0; t < nt; ++t) {
                size_t d = 0;
                for (int r = 0; r < kRows; ++r) {
                    float s = 0.f;
                    dn(ff, &s, 0, blob.data() + (size_t) r * row_bytes, 0, hqp[t], 0, 1);
                    d += std::memcmp(&out[t][r], &s, sizeof(float)) != 0;
                }
                if (d) {
                    std::printf("  FAIL ff=%d nt=%d token %d: %zu of %d rows differ bit-for-bit vs ggml\n",
                                ff, nt, t, d, kRows);
                    ++failures;
                }
            }
        }
        std::printf("iq4nl_neon_parity: ff=%d: %d rows x nt=1..%d checked against ggml_vec_dot_iq4_nl_q8_0\n",
                    ff, kRows, kMaxT);
    }
}

void run_bench(int mb) {
#if !defined(_WIN32)
    cpu_set_t s;
    CPU_ZERO(&s);
    CPU_SET(0, &s);
    sched_setaffinity(0, sizeof s, &s);
#endif
    const int ff = 640;
    const size_t row_bytes = (size_t) (ff / kQK) * kBS;
    const int rows = (int) ((size_t) mb * 1024 * 1024 / row_bytes);
    uint64_t seed = 0xB105ull;
    std::vector<uint8_t> blob((size_t) rows * row_bytes);
    fill_blocks(blob.data(), blob.size(), seed);
    std::vector<std::vector<uint8_t>> hq(kMaxT);
    for (int t = 0; t < kMaxT; ++t) {
        std::vector<float> x(ff);
        for (int i = 0; i < ff; ++i) x[i] = (float) ((mix(seed) >> 40) * 0x1p-24 * 2.0 - 1.0);
        hq[t] = q8_of(x);
    }
    const void* hqp[kMaxT];
    std::vector<float> out[kMaxT];
    float* outp[kMaxT];
    for (int t = 0; t < kMaxT; ++t) { hqp[t] = hq[t].data(); out[t].assign(rows, 0.f); outp[t] = out[t].data(); }
    const ggml_vec_dot_t dn = ggml_get_type_traits_cpu(GGML_TYPE_IQ4_NL)->vec_dot;
    std::printf("iq4nl_neon bench: %d rows x ff=%d, one thread, %d MB of blobs\n", rows, ff, mb);
    for (const int nt : {1, 2, 4, 8}) {
        double ms_g = 1e300, ms_k = 1e300;
        for (int rep = 0; rep < 3; ++rep) {
            auto t0 = Clock::now();
            for (int r = 0; r < rows; ++r)
                for (int t = 0; t < nt; ++t) {
                    float s = 0.f;
                    dn(ff, &s, 0, blob.data() + (size_t) r * row_bytes, 0, hqp[t], 0, 1);
                    out[t][r] = s;
                }
            auto t1 = Clock::now();
            if (rep) ms_g = std::min(ms_g, std::chrono::duration<double, std::milli>(t1 - t0).count());
            t0 = Clock::now();
            cpu::iq4nl256_down_rows(blob.data(), row_bytes, ff, hqp, nt, outp, 0, rows);
            t1 = Clock::now();
            if (rep) ms_k = std::min(ms_k, std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
        std::printf("  nt=%d  ggml %8.2f ms (%7.0f tok/s)  sdot %8.2f ms (%7.0f tok/s)  %.2fx\n",
                    nt, ms_g, rows * (double) nt / (ms_g * 1e-3), ms_k, rows * (double) nt / (ms_k * 1e-3),
                    ms_g / ms_k);
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (!cpu::cpu_neon_dot_ok()) {
        std::printf("iq4nl_neon_parity: no dotprod (HWCAP_ASIMDDOT) on this CPU: skip\n");
        return 77;
    }
    int mb = 128;
    bool bench = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--bench")) bench = true;
        else if (i + 1 < argc && !std::strcmp(argv[i], "--mb")) mb = std::atoi(argv[++i]);
    }
    if (bench) { run_bench(mb); return 0; }
    run_checks();
    if (failures) { std::printf("iq4nl_neon_parity: %d FAILURES\n", failures); return 1; }
    std::printf("iq4nl_neon_parity: all rows bit-identical to ggml-cpu's dot (dotprod: yes)\n");
    return 0;
}
