// include/strata/platform/cpu_relax.hpp - the spin-wait hint and the store fence, per architecture.
//
// x86: `_mm_pause` and `_mm_sfence` (the fence orders the write-combining stores into pinned host memory before
// the doorbell the GPU polls).  aarch64: `yield` and `dmb sy` - the full system barrier, so host stores reach the
// GPU (GB10 / Grace NVLink-C2C, and any PCIe endpoint) before a later store.  Apple silicon (the
// Metal build) keeps `dmb ishst`, the barrier its compat header (metal/include/metal_compat/immintrin.h) has always used.
// Local port of the approach in https://github.com/Niko1221/Strata/pull/1077 onto main e8ca9af.
#pragma once

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#define STRATA_X86 1
// Macros on x86, so the code the compiler sees is exactly the intrinsic that stood there before (the objects of the
// files that use them are identical before and after this header).  x86 does not reorder a load before an older load:
// the load fence has nothing to emit.
#define strata_cpu_pause() _mm_pause()
#define strata_store_fence() _mm_sfence()
#define strata_load_fence() ((void) 0)
#elif defined(__aarch64__)
inline void strata_cpu_pause() { __asm__ __volatile__("yield" ::: "memory"); }
#if defined(__APPLE__)
inline void strata_store_fence() { __asm__ __volatile__("dmb ishst" ::: "memory"); }
#define strata_load_fence() ((void) 0)   // the Metal build has never had one
#else
inline void strata_store_fence() { __asm__ __volatile__("dmb sy" ::: "memory"); }
// After a spin saw a flag the GPU raised, the data the GPU wrote before it must be read after the flag: an ARM core
// may otherwise load that data early.  Both fences are the FULL system barrier `dmb sy` (the calvinvette Jetson-Orin
// fork's choice, 2026-11-20): oshst/ishld is the tuned pair for a C2C-coherent GPU, but sy additionally orders
// against every system-scope observer (non-coherent PCIe endpoints included) for a few tens of cycles once per
// doorbell transition - insurance priced at under 1e-4% of CPU time.  The fences sit outside the spin loops.
inline void strata_load_fence() { __asm__ __volatile__("dmb sy" ::: "memory"); }
#endif
#else
#include <atomic>
inline void strata_cpu_pause() {}
inline void strata_store_fence() { std::atomic_thread_fence(std::memory_order_seq_cst); }
inline void strata_load_fence() { std::atomic_thread_fence(std::memory_order_acquire); }
#endif
