# GB10 bench records added by this fork

Every folder here is new against upstream `main` - community runs and A/B experiments that carry the numbers the
README and [the fork page](DGX_SPARK_FORK.md) cite. They live where they were measured; this is only the index.

Regenerate the list (after `git fetch upstream`):

```
git diff --name-only upstream/main main -- bench docs/bench-results | cut -d/ -f1-3 | sort -u
```

Titles are the first heading of each folder's RESULTS.md or README.md, as written on the day (A/B work folders that carry only results.json and engine logs are labeled from their slug).

- [`bench/results/2026-10-05-dgx-spark-gb10/`](../bench/results/2026-10-05-dgx-spark-gb10/) - Benchmark: NVIDIA GB10 (DGX Spark, Arm64), IQ3_XXS with images on
- [`bench/results/2026-10-28-community-gb10-swift-1-5-iq3-xxs/`](../bench/results/2026-10-28-community-gb10-swift-1-5-iq3-xxs/) - Community benchmark on NVIDIA GB10 (10-28-swift)
- [`bench/results/2026-10-28-community-gb10-swift-1-5-iq3-xxs-2/`](../bench/results/2026-10-28-community-gb10-swift-1-5-iq3-xxs-2/) - Community benchmark on NVIDIA GB10 (10-28-swift-2)
- [`bench/results/2026-10-28-community-gb10-swift-1-5-iq3-xxs-3/`](../bench/results/2026-10-28-community-gb10-swift-1-5-iq3-xxs-3/) - Community benchmark on NVIDIA GB10 (10-28-swift-3)
- [`bench/results/2026-10-28-community-gb10-swift-1-5-iq3-xxs-4/`](../bench/results/2026-10-28-community-gb10-swift-1-5-iq3-xxs-4/) - Community benchmark on NVIDIA GB10 (10-28-swift-4)
- [`bench/results/2026-10-28-community-gb10-swift-1-5-iq3-xxs-5/`](../bench/results/2026-10-28-community-gb10-swift-1-5-iq3-xxs-5/) - Community benchmark on NVIDIA GB10 (10-28-swift-5)
- [`bench/results/2026-10-28-community-gb10-swift-1-5-iq3-xxs-6/`](../bench/results/2026-10-28-community-gb10-swift-1-5-iq3-xxs-6/) - Community benchmark on NVIDIA GB10 (10-28-swift-6)
- [`bench/results/2026-10-28-gb10-compiler-ab/`](../bench/results/2026-10-28-gb10-compiler-ab/) - gcc-15 `-mcpu=gb10` rebuild on GB10: dotprod kernels linked, no speedup
- [`bench/results/2026-10-28-gb10-pool-affinity-ab/`](../bench/results/2026-10-28-gb10-pool-affinity-ab/) - GB10 CPU pool affinity A/B: `all` vs `p-cores` vs `auto`
- [`bench/results/2026-10-29-community-gb10-swift-1-5-iq3-xxs/`](../bench/results/2026-10-29-community-gb10-swift-1-5-iq3-xxs/) - Community benchmark on NVIDIA GB10 (10-29-swift)
- [`bench/results/2026-10-29-community-gb10-swift-1-5-iq3-xxs-2/`](../bench/results/2026-10-29-community-gb10-swift-1-5-iq3-xxs-2/) - Community benchmark on NVIDIA GB10 (10-29-swift-2)
- [`bench/results/2026-10-29-community-gb10-swift-1-5-iq3-xxs-3/`](../bench/results/2026-10-29-community-gb10-swift-1-5-iq3-xxs-3/) - Community benchmark on NVIDIA GB10 (10-29-swift-3)
- [`bench/results/2026-10-29-iq3xxs-neon-ab/`](../bench/results/2026-10-29-iq3xxs-neon-ab/) - IQ3_XXS NEON SDOT kernel A/B (gx10 GB10, engine 0.1.52, 2026-10-29)
- [`bench/results/2026-10-29-standard-gb10-bca52b7b/`](../bench/results/2026-10-29-standard-gb10-bca52b7b/) - Strata standard benchmark: NVIDIA GB10
- [`bench/results/2026-11-04-community-gb10-swift-1-5-iq3-xxs/`](../bench/results/2026-11-04-community-gb10-swift-1-5-iq3-xxs/) - Community benchmark on NVIDIA GB10 (11-04-swift)
- [`bench/results/2026-11-04-community-gb10-swift-1-5-iq3-xxs-2/`](../bench/results/2026-11-04-community-gb10-swift-1-5-iq3-xxs-2/) - Community benchmark on NVIDIA GB10 (11-04-swift-2)
- [`bench/results/2026-11-04-community-gb10-swift-1-5-iq3-xxs-df2/`](../bench/results/2026-11-04-community-gb10-swift-1-5-iq3-xxs-df2/) - Community benchmark on NVIDIA GB10 (11-04-swift-df2)
- [`bench/results/2026-11-05-e2-profile-fill-advise/`](../bench/results/2026-11-05-e2-profile-fill-advise/) - E2: per-layer profile fill advised (GB10, 2026-11-05) — cold fill 2.9x, default on
- [`bench/results/2026-11-05-ple-queue-depth-p0/`](../bench/results/2026-11-05-ple-queue-depth-p0/) - PLE direct queue-depth P0 (GB10, 2026-11-05) — deep-queue AIO arm: neutral stop, not built
- [`bench/results/2026-11-05-serve-startup-time/`](../bench/results/2026-11-05-serve-startup-time/) - Strata serve startup time on GB10 (2026-11-05) — warm restart 8.3 s; cold boot 73 s is mmap-fault-bound
- [`bench/results/2026-11-07-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs/`](../bench/results/2026-11-07-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs/) - Community benchmark on NVIDIA GB10 (11-07)
- [`bench/results/2026-11-07-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs-2-stock/`](../bench/results/2026-11-07-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs-2-stock/) - Community benchmark on NVIDIA GB10 (11-07-2-stock)
- [`bench/results/2026-11-07-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs-2-tuned-VOID/`](../bench/results/2026-11-07-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs-2-tuned-VOID/) - Community benchmark on NVIDIA GB10 (11-07-2-tuned-VOID)
- [`bench/results/2026-11-07-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs-2-tuned2/`](../bench/results/2026-11-07-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs-2-tuned2/) - Community benchmark on NVIDIA GB10 (11-07-2-tuned2)
- [`bench/results/2026-11-07-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs-3-stock2/`](../bench/results/2026-11-07-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs-3-stock2/) - Community benchmark on NVIDIA GB10 (11-07-3-stock2)
- [`bench/results/2026-11-19-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs/`](../bench/results/2026-11-19-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs/) - Community benchmark on NVIDIA GB10 (11-19)
- [`bench/results/2026-11-22-df2-profile/`](../bench/results/2026-11-22-df2-profile/) - A/B work folder (results.json + engine logs; see the 11/22 A/B series note)
- [`bench/results/2026-11-22-q2-neon/`](../bench/results/2026-11-22-q2-neon/) - GB10 A/B: Arm64 SDOT IQ2/IQ3_S gate/up kernels — 中立（2026-11-22）
- [`docs/bench-results/2026-11-19-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs/`](bench-results/2026-11-19-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs/) - Community benchmark on NVIDIA GB10 (11-19)
- [`docs/bench-results/2026-11-19-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs-2/`](bench-results/2026-11-19-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs-2/) - Community benchmark on NVIDIA GB10 (11-19-2)
- [`docs/bench-results/2026-11-19-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs-3/`](bench-results/2026-11-19-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs-3/) - Community benchmark on NVIDIA GB10 (11-19-3)
- [`docs/bench-results/2026-11-19-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs-4/`](bench-results/2026-11-19-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs-4/) - Community benchmark on NVIDIA GB10 (11-19-4)
- [`docs/bench-results/2026-11-19-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs-4-gb1/`](bench-results/2026-11-19-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs-4-gb1/) - Community benchmark on NVIDIA GB10 (11-19-4-gb1)
- [`docs/bench-results/2026-11-19-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs-4-gb2/`](bench-results/2026-11-19-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs-4-gb2/) - Community benchmark on NVIDIA GB10 (11-19-4-gb2)
- [`docs/bench-results/2026-11-19-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs-4-gc1/`](bench-results/2026-11-19-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs-4-gc1/) - Community benchmark on NVIDIA GB10 (11-19-4-gc1)
- [`docs/bench-results/2026-11-19-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs-4-gc2/`](bench-results/2026-11-19-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs-4-gc2/) - Community benchmark on NVIDIA GB10 (11-19-4-gc2)
- [`docs/bench-results/2026-11-19-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs-4-np1a/`](bench-results/2026-11-19-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs-4-np1a/) - Community benchmark on NVIDIA GB10 (11-19-4-np1a)
- [`docs/bench-results/2026-11-19-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs-4-np1b/`](bench-results/2026-11-19-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs-4-np1b/) - Community benchmark on NVIDIA GB10 (11-19-4-np1b)
- [`docs/bench-results/2026-11-19-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs-4-s1/`](bench-results/2026-11-19-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs-4-s1/) - Community benchmark on NVIDIA GB10 (11-19-4-s1)
- [`docs/bench-results/2026-11-19-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs-4-s2/`](bench-results/2026-11-19-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs-4-s2/) - Community benchmark on NVIDIA GB10 (11-19-4-s2)
- [`docs/bench-results/2026-11-19-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs-4-t1/`](bench-results/2026-11-19-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs-4-t1/) - Community benchmark on NVIDIA GB10 (11-19-4-t1)
- [`docs/bench-results/2026-11-19-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs-4-t2/`](bench-results/2026-11-19-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs-4-t2/) - Community benchmark on NVIDIA GB10 (11-19-4-t2)
- [`docs/bench-results/2026-11-22-b1-polystrata-ab/`](bench-results/2026-11-22-b1-polystrata-ab/) - B1 adoption A/B on GB10 (PolyStrata-technique shortlist, 2026-11-22)
- [`docs/bench-results/2026-11-23-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs/`](bench-results/2026-11-23-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs/) - Community benchmark on NVIDIA GB10 (11-23)
- [`docs/bench-results/2026-11-23-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs-3/`](bench-results/2026-11-23-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs-3/) - Community benchmark on NVIDIA GB10 (11-23-3)
- [`docs/bench-results/2026-11-26-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs/`](bench-results/2026-11-26-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs/) - Community benchmark on NVIDIA GB10 (11-26)
- [`docs/bench-results/ab-aux-2026-11-22-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs-ab/`](bench-results/ab-aux-2026-11-22-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs-ab/) - Community benchmark on NVIDIA GB10 (11/22 A/B arm aux)
- [`docs/bench-results/ab-base-2026-11-22-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs-ab/`](bench-results/ab-base-2026-11-22-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs-ab/) - Community benchmark on NVIDIA GB10 (11/22 A/B arm base)
- [`docs/bench-results/ab-base2-2026-11-22-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs-ab/`](bench-results/ab-base2-2026-11-22-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs-ab/) - Community benchmark on NVIDIA GB10 (11/22 A/B arm base2)
- [`docs/bench-results/ab-batch2-2026-11-22-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs-ab/`](bench-results/ab-batch2-2026-11-22-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs-ab/) - Community benchmark on NVIDIA GB10 (11/22 A/B arm batch2)
- [`docs/bench-results/ab-batch2aux-2026-11-22-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs-ab/`](bench-results/ab-batch2aux-2026-11-22-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs-ab/) - Community benchmark on NVIDIA GB10 (11/22 A/B arm batch2aux)
- [`docs/bench-results/ab-expdec-2026-11-22-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs-ab/`](bench-results/ab-expdec-2026-11-22-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs-ab/) - Community benchmark on NVIDIA GB10 (11/22 A/B arm expdec)
- [`docs/bench-results/ab-msgb-2026-11-22-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs-ab/`](bench-results/ab-msgb-2026-11-22-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs-ab/) - Community benchmark on NVIDIA GB10 (11/22 A/B arm msgb)
- [`docs/bench-results/ab-qsa-2026-11-22-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs-ab/`](bench-results/ab-qsa-2026-11-22-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs-ab/) - Community benchmark on NVIDIA GB10 (11/22 A/B arm qsa)
- [`docs/bench-results/ab-tail-2026-11-22-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs-ab/`](bench-results/ab-tail-2026-11-22-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs-ab/) - Community benchmark on NVIDIA GB10 (11/22 A/B arm tail)
