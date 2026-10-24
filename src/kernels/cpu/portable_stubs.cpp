// src/kernels/cpu/portable_stubs.cpp - the CPU kernels' entry points on a CPU that is not x86-64 (Arm64).
//
// Strata's own expert kernels (expert.cpp, q2_avx2.cpp, iq_avx512.cpp, iq_avx2.cpp, kq_avx2.cpp) are AVX-512 / AVX-2
// intrinsics and are built on x86-64 only.  Here a native (i-quant) pack runs on ggml-cpu's own dot products - its
// NEON / dotprod code - which native_expert.cpp reaches whenever the "supported" probes below say no.  What has no
// portable form (the canonical Q2_0 pack's VNNI kernel, the multi-token i-quant kernels) refuses by name instead of
// computing something else.
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/iq_avx1.hpp"
#include "strata/kernels/cpu/iq_avx2.hpp"
#include "strata/kernels/cpu/iq_avx512.hpp"
#include "strata/kernels/cpu/kq_avx1.hpp"
#include "strata/kernels/cpu/kq_avx2.hpp"
#include "strata/kernels/cpu/nvfp4_avx2.hpp"
#include "strata/kernels/cpu/nvfp4_avx512.hpp"
#include "strata/kernels/cpu/tq_cpu.hpp"

#include <arm_neon.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace strata::kernels::cpu {
namespace {

[[noreturn]] void x86_only(const char* what) {
    std::fprintf(stderr,
                 "strata: %s is an x86-64 (AVX) kernel and is not built for this CPU.\n"
                 "        On Arm64 use a native i-quant pack (IQ2_XS, IQ3_XXS, IQ3_S, ...), whose experts run on ggml-cpu.\n",
                 what);
    std::abort();
}

inline float bf16_to_f32(uint16_t b) {
    const uint32_t u = (uint32_t) b << 16;
    float f;
    std::memcpy(&f, &u, sizeof f);
    return f;
}

}  // namespace

// ---- the i-quant multi-token kernels: none; native_expert.cpp falls back to ggml-cpu's vec_dot
bool iq512_supported(int) noexcept { return false; }
void iq512_gu_rows(int, const uint8_t*, size_t, size_t, int, const void* const*, int, float* const*, int, int) {
    x86_only("the AVX-512 i-quant kernel");
}
void iq512_rows(int, const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int) {
    x86_only("the AVX-512 i-quant kernel");
}
bool iq256_supported(int) noexcept { return false; }
void iq256_gu_rows_exact_one(int, const uint8_t*, size_t, size_t, int, const void*, float*, int, int) {
    x86_only("the AVX-2 i-quant exact-one kernel");  // the exact1 call is behind the avx2 probe (native_expert.cpp:179)
}
void iq256_gu_rows(int, const uint8_t*, size_t, size_t, int, const void* const*, int, float* const*, int, int) {
    x86_only("the AVX-2 i-quant kernel");
}
void iq256_rows(int, const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int) {
    x86_only("the AVX-2 i-quant kernel");
}
void iq4nl256_down_rows(const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int) {
    x86_only("the AVX-2 IQ4_NL kernel");
}
bool kq256_supported(int) noexcept { return false; }
void kq256_gu_rows(int, const uint8_t*, size_t, size_t, int, const void* const*, int, float* const*, int, int) {
    x86_only("the AVX-2 K-quant kernel");
}
void kq256_rows(int, const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int) {
    x86_only("the AVX-2 K-quant kernel");
}

// ---- the routing-aware prefetch's router estimate: plain loops (the compiler vectorizes them for NEON)
void bf16_rows_dot(const uint16_t* w, int rows, int cols, const float* x, float* out) {
    for (int r = 0; r < rows; ++r) {
        const uint16_t* row = w + (size_t) r * cols;
        float s = 0.f;
        for (int c = 0; c < cols; ++c) s += bf16_to_f32(row[c]) * x[c];
        out[r] = s;
    }
}

void bf16_rows_dot_multi(const uint16_t* w, int rows, int cols, const float* x, int nt, float* out) {
    for (int t = 0; t < nt; ++t) bf16_rows_dot(w, rows, cols, x + (size_t) t * cols, out + (size_t) t * rows);
}

void bf16_rows_dot_multi_avx1(const uint16_t* w, int rows, int cols, const float* x, int nt, float* out) {
    bf16_rows_dot_multi(w, rows, cols, x, nt, out);   // never selected (cpu_avx1_ok is false here)
}

// ---- the canonical Q2_0 pack's CPU expert kernel (AVX-512 VNNI + VBMI): x86-64 only
const char* CpuFeatures::reason() const { return "not an x86-64 CPU"; }
CpuFeatures cpu_features() { return CpuFeatures{}; }

void cpu_require_expert_support() {
    std::fprintf(stderr,
                 "strata: this CPU cannot run the canonical Q2_0 pack's expert kernel (AVX-512 VNNI + VBMI, x86-64 only).\n"
                 "        On Arm64 use a native i-quant pack: IQ2_XS, IQ3_XXS or IQ3_S.\n");
    std::exit(1);
}

void expert_set_oracle_q8_0(bool) {}
bool expert_oracle_q8_0_enabled() { return false; }
// Called for every layer even with a native pack, whose kernels never read this ActQ (expert_source.cpp): a no-op here.
// The Q2_0 kernels that would read it are the stubs below and refuse.
void act_quant_q8_1(const float*, int, ActQ& a) { a.nchunks = 0; }
void s2_expert_vnni(const uint8_t*, const float*, float*, ExpertScratch&) { x86_only("the Q2_0 expert kernel"); }
void s2_expert_vnni_q(const uint8_t*, const ActQ&, float*, ExpertScratch&) { x86_only("the Q2_0 expert kernel"); }
void s2_expert_gu_rows(const uint8_t*, const ActQ&, float*, int, int) { x86_only("the Q2_0 expert kernel"); }
void s2_expert_down_rows(const uint8_t*, const ActQ&, float*, int, int) { x86_only("the Q2_0 expert kernel"); }
void s2_expert_gu_rows_multi(const uint8_t*, const ActQ* const*, int, float* const*, int, int) {
    x86_only("the Q2_0 expert kernel");
}
void s2_expert_down_rows_multi(const uint8_t*, const ActQ* const*, int, float* const*, int, int) {
    x86_only("the Q2_0 expert kernel");
}
void s2_expert_vnni_multi(const uint8_t*, const ActQ* const*, int, float* const*, ExpertScratchMulti&) {
    x86_only("the Q2_0 expert kernel");
}
void s2_expert_scalar(const uint8_t*, const float*, float*, bool) { x86_only("the Q2_0 expert kernel"); }
void q2_0_gguf_rows_multi(const uint8_t*, size_t, int, const ActQ* const*, int, float* const*, int, int) {
    x86_only("the Q2_0 row kernel");
}


// ---- Q2_0 rows in the GGUF block layout (18 bytes per 64 weights: fp16 d, 16 bytes of 2-bit codes) against `nt`
// activations - q2_avx2.cpp's arithmetic on NEON: codes 0..3 times the int8 activation per 32-value chunk (exact in
// int32), times the weight scale and the chunk scale, minus the weight scale times the chunk's hx (the -1 offset).
namespace {

inline float h2f(const uint8_t* p) {
    __fp16 h;
    std::memcpy(&h, p, 2);
    return (float) h;
}

// 16 code bytes -> 64 codes in value order (value i is in byte i/4, bits 2*(i%4)); vst4q interleaves them
inline void unpack64(const uint8_t* codes, uint8_t* out64) {
    const uint8x16_t b = vld1q_u8(codes), m3 = vdupq_n_u8(3);
    uint8x16x4_t c;
    c.val[0] = vandq_u8(b, m3);
    c.val[1] = vandq_u8(vshrq_n_u8(b, 2), m3);
    c.val[2] = vandq_u8(vshrq_n_u8(b, 4), m3);
    c.val[3] = vshrq_n_u8(b, 6);
    vst4q_u8(out64, c);
}

// sum of 32 codes (0..3) * int8 activations, exactly
inline int32_t dot32(const uint8_t* codes32, const int8_t* q) {
    int32x4_t acc = vdupq_n_s32(0);
    for (int i = 0; i < 32; i += 16) {
        const int8x16_t c = vreinterpretq_s8_u8(vld1q_u8(codes32 + i)), x = vld1q_s8(q + i);
        acc = vpadalq_s16(acc, vmull_s8(vget_low_s8(c), vget_low_s8(x)));
        acc = vpadalq_s16(acc, vmull_high_s8(c, x));
    }
    return vaddvq_s32(acc);
}

template <int NT>
inline void row_multi(const uint8_t* row, const ActQ* const* a, int nblocks, float* res) {
    float acc[NT], corr[NT];
    for (int t = 0; t < NT; ++t) { acc[t] = 0.f; corr[t] = 0.f; }
    alignas(16) uint8_t codes[64];
    for (int b = 0; b < nblocks; ++b) {
        const uint8_t* blk = row + (size_t) b * 18;
        const float d = h2f(blk);
        unpack64(blk + 2, codes);
        for (int t = 0; t < NT; ++t) {
            const int8_t* q = a[t]->q + b * 64;
            acc[t] += d * a[t]->scale[2 * b] * (float) dot32(codes, q);
            acc[t] += d * a[t]->scale[2 * b + 1] * (float) dot32(codes + 32, q + 32);
            corr[t] += d * (a[t]->hx[2 * b] + a[t]->hx[2 * b + 1]);
        }
    }
    for (int t = 0; t < NT; ++t) res[t] = acc[t] - corr[t];
}

template <int NT>
void rows(const uint8_t* w, size_t row_bytes, int nblocks, const ActQ* const* a, float* const* out, int r0, int r1) {
    float res[NT];
    for (int r = r0; r < r1; ++r) {
        row_multi<NT>(w + (size_t) r * row_bytes, a, nblocks, res);
        for (int t = 0; t < NT; ++t) out[t][r] = res[t];
    }
}

}  // namespace

void q2_0_gguf_rows_multi_avx2(const uint8_t* w, size_t row_bytes, int nblocks, const ActQ* const* a, int nt,
                               float* const* out, int r0, int r1) {
    switch (nt) {
        case 1: rows<1>(w, row_bytes, nblocks, a, out, r0, r1); break;
        case 2: rows<2>(w, row_bytes, nblocks, a, out, r0, r1); break;
        case 3: rows<3>(w, row_bytes, nblocks, a, out, r0, r1); break;
        case 4: rows<4>(w, row_bytes, nblocks, a, out, r0, r1); break;
        default:
            for (int t0 = 0; t0 < nt; t0 += 4) {
                const int k = nt - t0 < 4 ? nt - t0 : 4;
                q2_0_gguf_rows_multi_avx2(w, row_bytes, nblocks, a + t0, k, out + t0, r0, r1);
            }
    }
}

// the legacy activation quantizer (the scalar rule of expert.cpp: half away from zero, clamp to +-127)
void act_quant_q8_1_avx2(const float* x, int n, ActQ& a) {
    a.nchunks = n / QKA;
    for (int k = 0; k < a.nchunks; ++k) {
        const float* xb = x + k * QKA;
        float amax = 0.f;
        for (int j = 0; j < QKA; ++j) amax = std::fmax(amax, std::fabs(xb[j]));
        const float s = amax > 0.f ? amax / 127.f : 0.f;
        const float inv = s > 0.f ? 1.f / s : 0.f;
        int32_t sum = 0;
        int8_t* q = a.q + k * QKA;
        for (int j = 0; j < QKA; ++j) {
            const float t = xb[j] * inv;
            int v = (int) (t + (t >= 0.f ? 0.5f : -0.5f));
            v = v < -127 ? -127 : (v > 127 ? 127 : v);
            q[j] = (int8_t) v;
            sum += v;
        }
        a.scale[k] = s;
        a.sum[k] = sum;
        a.hx[k] = s * (float) sum;
    }
}


// ---- v0.1.43 additions: the AVX-1 rung (issue #1699) and the IQ AVX-1 kernels ----
// On Arm64 cpu_avx512_ok() and cpu_avx2_ok() are false, so q2_rows_any, q2_rows_any_legacy, and
// act_quant_any dispatch to the avx1 rungs at runtime; they delegate to the NEON stand-ins above
// (same arithmetic).  The IQ AVX-1 kernels and the Q8_K quantizers are never selected here -
// native_expert.cpp reaches ggml-cpu's NEON dots when the "supported" probes say no - so they
// refuse by name.

void q2_0_gguf_rows_multi_avx1(const uint8_t* w, size_t row_bytes, int nblocks, const ActQ* const* a, int nt,
                               float* const* out, int r0, int r1) {
    q2_0_gguf_rows_multi_avx2(w, row_bytes, nblocks, a, nt, out, r0, r1);
}
void q2_0_gguf_rows_multi_avx1_legacy(const uint8_t* w, size_t row_bytes, int nblocks, const ActQ* const* a, int nt,
                                      float* const* out, int r0, int r1) {
    q2_0_gguf_rows_multi_avx2(w, row_bytes, nblocks, a, nt, out, r0, r1);
}
void q2_0_gguf_rows_multi_avx2_legacy(const uint8_t* w, size_t row_bytes, int nblocks, const ActQ* const* a, int nt,
                                      float* const* out, int r0, int r1) {
    q2_0_gguf_rows_multi_avx2(w, row_bytes, nblocks, a, nt, out, r0, r1);
}
void act_quant_q8_1_avx1(const float* x, int n, ActQ& a) { act_quant_q8_1_avx2(x, n, a); }

bool iq128_supported(int) noexcept { return false; }
void iq128_gu_rows(int, const uint8_t*, size_t, size_t, int, const void* const*, int, float* const*, int, int) {
    x86_only("the IQ AVX-1 gate/up kernel");
}
bool iq128_down_supported(int) noexcept { return false; }
void iq128_down_rows(int, const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int) {
    x86_only("the IQ AVX-1 down kernel");
}
void q8k_quant_avx1(const float*, void*, int64_t) { x86_only("the Q8_K quantizer"); }
void q8k_quant_avx2(const float*, void*, int64_t) { x86_only("the Q8_K quantizer"); }

// 0.1.44: the NVFP4 expert rows (nvfp4_avx512.cpp / nvfp4_avx2.cpp).  native_expert.cpp reaches them behind its
// AVX probes, which say no on Arm64: the fits() probes fall through to ggml-cpu's NVFP4 dot, the row kernels refuse.
bool nvfp4_512_fits(int) { return false; }
void nvfp4_512_gu_rows(const uint8_t*, size_t, size_t, int, const void* const*, int, float* const*, int, int, float,
                       float) {
    x86_only("the NVFP4 AVX-512 gate/up kernel");
}
void nvfp4_512_rows(const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int, float) {
    x86_only("the NVFP4 AVX-512 row kernel");
}
bool nvfp4_256_fits(int) { return false; }
void nvfp4_256_gu_rows(const uint8_t*, size_t, size_t, int, const void* const*, int, float* const*, int, int, float,
                       float) {
    x86_only("the NVFP4 AVX-2 gate/up kernel");
}
void nvfp4_256_rows(const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int, float) {
    x86_only("the NVFP4 AVX-2 row kernel");
}

// 0.2.1: the trellis (TQ) expert rows (tq_cpu.cpp / tq_avx2.cpp / tq_avx512.cpp).  native_expert.cpp reaches them
// behind the tq_supported() / tq_kernels_enabled() probes: on Arm64 the probes say no (upstream ggml here has no
// trellis type, ggml-cpu's vec_dot runs the expert), the row kernels refuse by name.  tq_cpu_parity links the same
// entry points; it reports and skips on this host, as on a CPU without the trellis packs.
bool tq_supported(int) noexcept { return false; }
int tq_kbits(int) noexcept { return 0; }   // the inline tq_block_bytes in the header calls this; the probes gate it off
TqIsa tq_isa() { return TqIsa::scalar; }
bool tq_isa_ok(TqIsa isa) { return isa == TqIsa::scalar; }
bool tq_kernels_enabled() { return false; }
const char* tq_isa_name(TqIsa isa) { return isa == TqIsa::scalar ? "scalar" : (isa == TqIsa::avx2 ? "avx2" : "avx512"); }
void tq_rows(TqIsa, int, const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int) {
    x86_only("the trellis (TQ) row kernel");
}
void tq_gu_rows(TqIsa, int, const uint8_t*, size_t, size_t, int, const void* const*, int, float* const*, int, int) {
    x86_only("the trellis (TQ) gate/up kernel");
}

}  // namespace strata::kernels::cpu
