# IQ3_XXS NEON SDOT kernel A/B (gx10 GB10, engine 0.1.52, 2026-10-29)

Same engine binary (branch `iq3xxs-neon`, commit f2e485f2), same pack/args/vision; the only
difference is the dispatch kill switch `STRATA_NO_IQ3XXS=1` (A = ggml-cpu NEON dot fallback)
versus the new `iq3xxs256_gu_rows` SDOT kernel (B). Suite = `bench/results/2026-10-05-dgx-spark-gb10/bench.py`
against :8000 (random.seed(1), identical prompts both runs), model window closed between runs.

| prompt_tok | A prefill tok/s | B prefill tok/s | delta |
|---|---|---|---|
| 431 | 521.8 | 518.7 | -0.6% |
| 1582 | 1024.9 | 1019.5 | -0.5% |
| 6190 | 1484.1 | 1477.8 | -0.4% |
| 12334 | 1581.1 | 1580.4 | -0.0% |
| 24622 | 1655.5 | 1659.4 | +0.2% |
| 49199 | 1681.1 | 1684.4 | +0.2% |
| 75047 | 1667.5 | 1669.9 | +0.1% |

Decode (gen 128, MTP spec 4): A 55.1 / 52.6 / 56.6 / 50.5 / 53.2 / 54.8 / 48.7 vs
B 55.1 / 52.4 / 56.7 / 50.4 / 53.4 / 54.8 / 48.6 tok/s — within noise.

**Verdict: no measurable end-to-end gain** (all deltas within run noise), despite the kernel
being bit-identical to the ggml dot (parity nt=1..8) and 1.26-3.11x faster than the ggml dot
in isolation. The expert rows are not dot-kernel-bound in the engine pipeline: the windows are
DRAM/mmap-traffic- and GPU-latency-bound, so replacing the dot does not move the wall clock.
Matches the 2026-10-28 `-mcpu=gb10` dotprod A/B (838fefc1): linking dotprod kernels also
changed nothing. The kernel stays gated behind `cpu_neon_dot_ok() && f.gu_type == 18` with the
`STRATA_NO_IQ3XXS` kill switch; branch not merged to main.
