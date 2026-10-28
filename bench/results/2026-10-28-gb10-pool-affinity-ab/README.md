# GB10 CPU pool affinity A/B: `all` vs `p-cores` vs `auto`

Measured on 2026-10-28 on the same GB10 box as [the community record](../2026-10-28-community-gb10-swift-1-5-iq3-xxs/README.md),
with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speed only: no correctness check was run.

## Setup (identical in all three rounds)

- NVIDIA GB10 (10 Cortex-X925 P-cores + 10 Cortex-A725 E-cores, no SMT), 121.6 GiB unified memory; engine 0.1.52;
  model swift-1.5-iq3_xxs, context 262,144
- Same serve config in every round (pack, native, PLE, expert cache, prefill 8192, spec 4, MTP, kv int8, ...) with
  one extra engine arg per round: `--pool-affinity all` / `p-cores` / `auto`
- Each round: fresh server, one warm-up request (model load not timed), then the suite; one warm-up chat and a 1K
  prompt are discarded first
- Placements from the engine's own start line:
  - `all`: 18 pool workers on 15,16,17,18,5,...,14,0,1,2,3 (9 X925 + 9 A725), host on 19 (X925)
  - `p-cores`: 9 pool workers on 6,7,8,9,15,16,17,18,19 (all X925), host on 5 (X925)
  - `auto`: the same placement as `p-cores` on this CPU (no SMT siblings; the default count is P-cores minus one)

## Results (medians)

| Metric | `all` (18 workers, P+E) | `p-cores` (9 workers, P) | `auto` (9 workers, P) |
| --- | ---: | ---: | ---: |
| chat decode tok/s | 61.4 (58.4-86.1) | 62.8 (56.0-84.1) | 68.0 (57.1-85.6) |
| agent decode tok/s | 61.2 (51.4-82.4) | 65.0 (55.4-74.1) | 64.9 (31.6-80.9) |
| chat first token s | 0.43 | 0.44 | 0.43 |
| agent turn first token s | 2.29 | 2.28 | 2.28 |
| 4K prompt tok/s | 1423.6 (1367.4-1424.4) | 1429.1 (1380.3-1431.5) | 1427.3 (1374.8-1427.8) |
| 16K prompt tok/s | 1667.8 (1659.4-1672.6) | 1677.9 (1667.2-1678.1) | 1675.7 (1663.1-1677.8) |
| agent cache hit % | 70.38 | 70.39 | 70.48 |

The full suite tables are in the adjacent folders `-2` (`all`), `-3` (`p-cores`) and `-4` (`auto`).

## Reading

- Halving the pool to performance cores only is not a loss: agent decode +6.2% (61.2 -> 65.0, matched by `auto`),
  chat decode median +2.3 to +10.8% within its run-to-run spread.
- Prompt reading does not move (±0.6%): the prefill path is DRAM-bound and core choice does not reach it.
- First-token times match across layouts.
- On GB10, `auto` and `p-cores` settle on the same placement, so on this box they are the same configuration.
- Decode variance is wide (n=9, 56-86 tok/s); the agent decode +6.2% (n=15, matched by two layouts) is the
  stable effect.

Adopted: this install's serve config runs `--pool-affinity auto`.
