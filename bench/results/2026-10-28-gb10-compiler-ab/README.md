# gcc-15 `-mcpu=gb10` rebuild on GB10: dotprod kernels linked, no speedup

Measured 2026-10-28 on the same GB10 box, engine 0.1.52, as
[the affinity A/B](../2026-10-28-gb10-pool-affinity-ab/README.md). The baseline is that A/B's `-4` round
(gcc 13.3 build, `--pool-affinity auto`). `tools/strata_bench.py`, suite 1, 3 runs per test, medians with ranges.

## What was rebuilt

The stock build uses gcc 13.3, whose native detection does not know the GB10: `HAVE_DOTPROD` empty, 0 dotprod
instructions in `libggml-cpu.a`. gcc 15.2 does know it (`-mcpu=gb10` = Cortex-X925: dotprod, i8mm, SVE2). The box
was rebuilt in an isolated copy with `CC=gcc-15 CXX=g++-15 CFLAGS/CXXFLAGS="-mcpu=gb10"` (the CUDA host compiler
kept at gcc 13.3, nvcc 13.3 unchanged). Result: `HAVE_DOTPROD=1`, `HAVE_MATMUL_INT8=1`, **402** `vdotq`/`smmla`
instructions in `libggml-cpu.a` (0 in the stock build). Engine version identical (0.1.52), so the two binaries
differ only by compiler and target flags - a pure compiler A/B.

## Same answers

Greedy (T=0) completions through the two servers side by side (stock build on port 8000, rebuild on 8010, same
pack, same engine args): every test prompt byte-identical (two-sentence explanation, haiku, 17x23 arithmetic).
The dotprod kernels replace code, not math.

## Results (medians)

| Metric | gcc 13.3 (`-4`, `auto`) | gcc-15 `-mcpu=gb10` r1 (`-5`) | r2 (`-6`) |
| --- | ---: | ---: | ---: |
| chat decode tok/s | 68.0 (57.1-85.6) | 61.4 (54.4-84.7) | 63.5 (57.6-84.1) |
| agent decode tok/s | 64.9 (31.6-80.9) | 62.0 (41.6-75.1) | 62.8 (53.6-70.8) |
| chat first token s | 0.43 | 0.43 | 0.43 |
| agent turn first token s | 2.28 | 2.44 | 2.29 |
| 4K prompt tok/s | 1427.3 | 1435.4 | 1427.1 |
| 16K prompt tok/s | 1675.7 | 1672.9 | 1666.4 |
| cache hit % | 70.5 | 70.5 | 70.5 |

Full suite tables: adjacent folders `-5` and `-6`.

## Reading

- The dotprod kernels speed the CPU's i-quant row dots. On this install that path is a minority of the work - the
  GPU computes the cached experts and prefill is DRAM-bound - and the numbers show it: prompt reading and TTFT
  match to the third digit, decode medians sit inside the baseline's run-to-run spread (n=9, 54-86 tok/s).
- Two rounds, no signal above noise. 402 new instructions buy nothing measurable at any measured point.
- **Not adopted**: the production engine stays the gcc 13.3 build. The rebuild is kept for reference at
  `/home/taguma/Strata-gb10-build` (engine 0.1.52, same args, `--pool-affinity auto`).
- Timing note for this box: the server's start-up warm-up aborts the engine (exit -6, "the Q8_K quantizer is an
  x86-64 (AVX) kernel...") on some starts, in both builds alike (13 of 101 starts in the production log; none in
  the timed rounds, whose starts warm differently). The server restarts the engine and serving continues; warm-up
  starts are not timed.
