# GB10 bench records added by this fork

Every folder here is new against upstream `main` - community runs and A/B experiments that carry the numbers the
README and [the fork page](DGX_SPARK_FORK.md) cite. They live where they were measured; this is only the index.

Regenerate the list (after `git fetch upstream`):

```
git diff --name-only upstream/main main -- bench docs/bench-results | cut -d/ -f1-3 | sort -u
```

Titles are the first heading of each folder's RESULTS.md or README.md, as written on the day.

- [`bench/results/2026-10-05-dgx-spark-gb10/`](../bench/results/2026-10-05-dgx-spark-gb10/) - Benchmark: NVIDIA GB10 (DGX Spark, Arm64), IQ3_XXS with images on
- [`bench/results/2026-10-28-community-gb10-swift-1-5-iq3-xxs/`](../bench/results/2026-10-28-community-gb10-swift-1-5-iq3-xxs/) - Community benchmark on NVIDIA GB10
- [`bench/results/2026-10-28-community-gb10-swift-1-5-iq3-xxs-2/`](../bench/results/2026-10-28-community-gb10-swift-1-5-iq3-xxs-2/) - Community benchmark on NVIDIA GB10
- [`bench/results/2026-10-28-community-gb10-swift-1-5-iq3-xxs-3/`](../bench/results/2026-10-28-community-gb10-swift-1-5-iq3-xxs-3/) - Community benchmark on NVIDIA GB10
- [`bench/results/2026-10-28-community-gb10-swift-1-5-iq3-xxs-4/`](../bench/results/2026-10-28-community-gb10-swift-1-5-iq3-xxs-4/) - Community benchmark on NVIDIA GB10
- [`bench/results/2026-10-28-community-gb10-swift-1-5-iq3-xxs-5/`](../bench/results/2026-10-28-community-gb10-swift-1-5-iq3-xxs-5/) - Community benchmark on NVIDIA GB10
- [`bench/results/2026-10-28-community-gb10-swift-1-5-iq3-xxs-6/`](../bench/results/2026-10-28-community-gb10-swift-1-5-iq3-xxs-6/) - Community benchmark on NVIDIA GB10
- [`bench/results/2026-10-28-gb10-compiler-ab/`](../bench/results/2026-10-28-gb10-compiler-ab/) - gcc-15 `-mcpu=gb10` rebuild on GB10: dotprod kernels linked, no speedup
- [`bench/results/2026-10-28-gb10-pool-affinity-ab/`](../bench/results/2026-10-28-gb10-pool-affinity-ab/) - GB10 CPU pool affinity A/B: `all` vs `p-cores` vs `auto`
- [`bench/results/2026-10-29-community-gb10-swift-1-5-iq3-xxs/`](../bench/results/2026-10-29-community-gb10-swift-1-5-iq3-xxs/) - Community benchmark on NVIDIA GB10
- [`bench/results/2026-10-29-community-gb10-swift-1-5-iq3-xxs-2/`](../bench/results/2026-10-29-community-gb10-swift-1-5-iq3-xxs-2/) - Community benchmark on NVIDIA GB10
- [`bench/results/2026-10-29-community-gb10-swift-1-5-iq3-xxs-3/`](../bench/results/2026-10-29-community-gb10-swift-1-5-iq3-xxs-3/) - Community benchmark on NVIDIA GB10
- [`bench/results/2026-10-29-iq3xxs-neon-ab/`](../bench/results/2026-10-29-iq3xxs-neon-ab/) - IQ3_XXS NEON SDOT kernel A/B (gx10 GB10, engine 0.1.52, 2026-10-29)
- [`bench/results/2026-10-29-standard-gb10-bca52b7b/`](../bench/results/2026-10-29-standard-gb10-bca52b7b/) - Strata standard benchmark: NVIDIA GB10
- [`bench/results/2026-11-04-community-gb10-swift-1-5-iq3-xxs/`](../bench/results/2026-11-04-community-gb10-swift-1-5-iq3-xxs/) - Community benchmark on NVIDIA GB10
- [`bench/results/2026-11-04-community-gb10-swift-1-5-iq3-xxs-2/`](../bench/results/2026-11-04-community-gb10-swift-1-5-iq3-xxs-2/) - Community benchmark on NVIDIA GB10
- [`bench/results/2026-11-04-community-gb10-swift-1-5-iq3-xxs-df2/`](../bench/results/2026-11-04-community-gb10-swift-1-5-iq3-xxs-df2/) - Community benchmark on NVIDIA GB10
- [`bench/results/2026-11-05-e2-profile-fill-advise/`](../bench/results/2026-11-05-e2-profile-fill-advise/) - E2: per-layer profile fill advised (GB10, 2026-11-05) — cold fill 2.9x, default on
- [`bench/results/2026-11-05-ple-queue-depth-p0/`](../bench/results/2026-11-05-ple-queue-depth-p0/) - PLE direct queue-depth P0 (GB10, 2026-11-05) — deep-queue AIO arm: neutral stop, not built
- [`bench/results/2026-11-05-serve-startup-time/`](../bench/results/2026-11-05-serve-startup-time/) - Strata serve startup time on GB10 (2026-11-05) — warm restart 8.3 s; cold boot 73 s is mmap-fault-bound
- [`bench/results/2026-11-07-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs/`](../bench/results/2026-11-07-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs/) - Community benchmark on NVIDIA GB10
- [`bench/results/2026-11-07-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs-2-stock/`](../bench/results/2026-11-07-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs-2-stock/) - Community benchmark on NVIDIA GB10
- [`bench/results/2026-11-07-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs-2-tuned-VOID/`](../bench/results/2026-11-07-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs-2-tuned-VOID/) - Community benchmark on NVIDIA GB10
- [`bench/results/2026-11-07-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs-2-tuned2/`](../bench/results/2026-11-07-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs-2-tuned2/) - Community benchmark on NVIDIA GB10
- [`bench/results/2026-11-07-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs-3-stock2/`](../bench/results/2026-11-07-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs-3-stock2/) - Community benchmark on NVIDIA GB10
- [`bench/results/2026-11-19-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs/`](../bench/results/2026-11-19-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs/) - Community benchmark on NVIDIA GB10
- [`docs/bench-results/2026-11-19-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs/`](bench-results/2026-11-19-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs) - Community benchmark on NVIDIA GB10
- [`docs/bench-results/2026-11-19-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs-2/`](bench-results/2026-11-19-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs-2) - Community benchmark on NVIDIA GB10
- [`docs/bench-results/2026-11-19-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs-3/`](bench-results/2026-11-19-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs-3) - Community benchmark on NVIDIA GB10
- [`docs/bench-results/2026-11-19-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs-4/`](bench-results/2026-11-19-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs-4) - Community benchmark on NVIDIA GB10
- [`docs/bench-results/2026-11-19-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs-4-gb1/`](bench-results/2026-11-19-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs-4-gb1) - Community benchmark on NVIDIA GB10
- [`docs/bench-results/2026-11-19-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs-4-gb2/`](bench-results/2026-11-19-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs-4-gb2) - Community benchmark on NVIDIA GB10
- [`docs/bench-results/2026-11-19-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs-4-gc1/`](bench-results/2026-11-19-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs-4-gc1) - Community benchmark on NVIDIA GB10
- [`docs/bench-results/2026-11-19-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs-4-gc2/`](bench-results/2026-11-19-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs-4-gc2) - Community benchmark on NVIDIA GB10
- [`docs/bench-results/2026-11-19-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs-4-np1a/`](bench-results/2026-11-19-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs-4-np1a) - Community benchmark on NVIDIA GB10
- [`docs/bench-results/2026-11-19-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs-4-np1b/`](bench-results/2026-11-19-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs-4-np1b) - Community benchmark on NVIDIA GB10
- [`docs/bench-results/2026-11-19-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs-4-s1/`](bench-results/2026-11-19-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs-4-s1) - Community benchmark on NVIDIA GB10
- [`docs/bench-results/2026-11-19-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs-4-s2/`](bench-results/2026-11-19-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs-4-s2) - Community benchmark on NVIDIA GB10
- [`docs/bench-results/2026-11-19-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs-4-t1/`](bench-results/2026-11-19-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs-4-t1) - Community benchmark on NVIDIA GB10
- [`docs/bench-results/2026-11-19-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs-4-t2/`](bench-results/2026-11-19-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs-4-t2) - Community benchmark on NVIDIA GB10
