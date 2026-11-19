# What this fork changes over upstream Strata

Upstream is [Niko1221/Strata](https://github.com/Niko1221/Strata), kept as the `upstream` remote. The
[DGX Spark section of the README](../README.md#what-is-different-from-the-x86-64-build) states the four
substantive deltas; this page is the file-by-file version of the same claim, so a reviewer can check them
against the code instead of trusting the prose.

152 files changed, 38108 insertions(+), 869 deletions(-) as measured on 2026-11-19. Of those, 86 are benchmark records
under `bench/results/` and `docs/bench-results/` (this fork's A/B and community-run evidence, listed at the
bottom); the code-and-docs delta is **66 files (9 added, 48 modified, 9 deleted)**.

The sync direction never reverses: every upstream release is merged on top of the port (first landed in
[`324b0ef`](https://github.com/taguma0/Strata-DGX-Spark/commit/324b0ef)); nothing from this fork is
assumed to go back. A commit-level log diff is deliberately *not* used as the ledger - releases arrive as
cherry-picks, so `upstream/main..main` is mostly upstream's own authors.

## Regenerating this page

```
git fetch upstream
git diff --shortstat upstream/main main
git diff --name-status upstream/main main -- ':(exclude)bench/*' ':(exclude)docs/bench-results/*'
```

Everything under "Bench records" below comes from the third command; the area lists below were written
from the same output and should be re-derived, not hand-extended, after a big upstream merge.

## Engine (14 files)

The Arm64 / GB10 kernel layer - the heart of the fork.

- **Added** `engine/src/kernels/cpu/iq3xxs_neon.cpp`, `iq4nl_neon.cpp` (+`*_parity.cpp` twins): the two tuned
  SDOT kernels for the IQ3_XXS and IQ4_NL gate/up rows. Bit-identical to the ggml dots they replace,
  1.26x (1 token) to 3.11x (4+) in isolation; end-to-end speed is unchanged because the expert windows are
  DRAM/GPU-bound ([A/B record](../bench/results/2026-10-29-iq3xxs-neon-ab/RESULTS.md)).
- **Added** `engine/src/kernels/cpu/portable_stubs.cpp`: `ggml_fp16_to_fp32` and friends, needed because the
  0.2.16 release engine no longer links `ggml-base`.
- **Modified** `engine/CMakeLists.txt`, `cmake/backends/cpu.cmake`: build the NEON sources on Arm64 and leave
  the x86-only AVX-512/AVX-2 kernels out of the Arm build.
- **Modified** `src/kernels/cpu/native_expert.cpp`, `expert_layout.cpp`, `include/strata/kernels/cpu/expert.hpp`,
  `iq_avx2.hpp`, `src/core/native_dense.cpp`, `src/core/verify.cpp`, `src/program/generate.cpp`: the i-quant
  rows' GB10 path - the NEON include propagation, the f16 scale decode, and the dense-layer GB10 fixes.

## Serve: IPv6 and the VRAM load gate (8 files)

- `terminal/serve/server.py`, `server_parts/http_api.py` (+`test_server.py`): IPv6 listening. An address with
  a colon binds the IPv6 family (`::` serves IPv4 too via `V6ONLY` off), URLs bracket IPv6 hosts, and the
  printed address skips loopback and link-local `fe80::`.
- `terminal/serve/server_parts/engines.py`, `terminal/strata_control/vram_fit.py` (+**added**
  `terminal/serve/test_vram_load_gate.py`, `telemetry.py`): `vram_need: "auto"` prices the model's own VRAM
  floor (`start_need_mib()`), `load_wait_s` lets a load wait for the card, and on GB10 - whose unified memory
  leaves NVML with no frame buffer - the free memory is read through the CUDA driver instead.
- `terminal/serve/server.py` also carries the fork's dual-stack default used by the chat page and API.

## Setup and tests (15 files)

- `terminal/setup.py`: ARM64 detection as a test-overridable global (golden-style tests set `setup.ARM64 = False`
  to describe an x86 PC on any host), NEON as the Arm baseline in place of the AVX checks, the machine model
  suffixed "(Arm64, NEON)", and `unified_vram_gb()` - the GB10 rule of a quarter of RAM, at most 32 GB, counted
  as VRAM so shared bytes are not counted twice.
- `terminal/tools/test_setup_*.py` (12) + `terminal/tests/terminal/test_drafting*.py`: the same tests adapted
  to the 0.2.16 `terminal/` layout, Arm hosts and the fork's config names.

## App (3 files)

`app/host/server/native-settings.mjs`, `src/lib/drafting.ts` (+test): the draft-layer sentences the fork's
serve prints, mirrored in the app.

## Docs (11 files)

`docs/DGX_SPARK_LINUX.md` (the upstream-official Spark path, retested here), `docs/DETAILS.md` and
`docs/VRAM_ELASTIC.md` (the load-gate additions), **added** `docs/SWIFT-DF2-GB10-2026-11-04.md` (the DF2
drafter record on GB10), and the 7 `docs/lang/` README translations for the fork's intro rewrite.

## Removed from upstream (9 files)

GitHub plumbing that does not run (or does not make sense) on a fork: the 4 `ISSUE_TEMPLATE` files,
`pull_request_template.md`, and the `backends.yml`, `docs.yml`, `snapshot.yml` workflows; plus
`tools/BENCH.bat`. Upstream remains the place to report issues.

## Root and tools (6 files)

`build.sh` (**added**, the fork's GB10 build entry), `.gitignore`, `README.md` (the DGX Spark section),
`SUPPORTERS.md`; `tools/requant_dense_gguf.py` (**added**, requantizing the dense layers) and
`tools/docker/build.sh`.

## Bench records (this fork's evidence, 86 folders)

New in the fork, kept because the README's numbers point at them:

- `bench/results/2026-10-05-dgx-spark-gb10/README.md/`
- `bench/results/2026-10-05-dgx-spark-gb10/bench.py/`
- `bench/results/2026-10-05-dgx-spark-gb10/results.json/`
- `bench/results/2026-10-28-community-gb10-swift-1-5-iq3-xxs-2/README.md/`
- `bench/results/2026-10-28-community-gb10-swift-1-5-iq3-xxs-2/results.json/`
- `bench/results/2026-10-28-community-gb10-swift-1-5-iq3-xxs-3/README.md/`
- `bench/results/2026-10-28-community-gb10-swift-1-5-iq3-xxs-3/results.json/`
- `bench/results/2026-10-28-community-gb10-swift-1-5-iq3-xxs-4/README.md/`
- `bench/results/2026-10-28-community-gb10-swift-1-5-iq3-xxs-4/results.json/`
- `bench/results/2026-10-28-community-gb10-swift-1-5-iq3-xxs-5/README.md/`
- `bench/results/2026-10-28-community-gb10-swift-1-5-iq3-xxs-5/results.json/`
- `bench/results/2026-10-28-community-gb10-swift-1-5-iq3-xxs-6/README.md/`
- `bench/results/2026-10-28-community-gb10-swift-1-5-iq3-xxs-6/results.json/`
- `bench/results/2026-10-28-community-gb10-swift-1-5-iq3-xxs/README.md/`
- `bench/results/2026-10-28-community-gb10-swift-1-5-iq3-xxs/results.json/`
- `bench/results/2026-10-28-gb10-compiler-ab/README.md/`
- `bench/results/2026-10-28-gb10-pool-affinity-ab/README.md/`
- `bench/results/2026-10-29-community-gb10-swift-1-5-iq3-xxs-2/README.md/`
- `bench/results/2026-10-29-community-gb10-swift-1-5-iq3-xxs-2/engine-log-excerpt.txt/`
- `bench/results/2026-10-29-community-gb10-swift-1-5-iq3-xxs-2/results.json/`
- `bench/results/2026-10-29-community-gb10-swift-1-5-iq3-xxs-3/README.md/`
- `bench/results/2026-10-29-community-gb10-swift-1-5-iq3-xxs-3/engine-log-excerpt.txt/`
- `bench/results/2026-10-29-community-gb10-swift-1-5-iq3-xxs-3/results.json/`
- `bench/results/2026-10-29-community-gb10-swift-1-5-iq3-xxs/README.md/`
- `bench/results/2026-10-29-community-gb10-swift-1-5-iq3-xxs/results.json/`
- `bench/results/2026-10-29-iq3xxs-neon-ab/RESULTS.md/`
- `bench/results/2026-10-29-iq3xxs-neon-ab/fallback-ggml-neon.txt/`
- `bench/results/2026-10-29-iq3xxs-neon-ab/sdot-kernel.txt/`
- `bench/results/2026-10-29-standard-gb10-bca52b7b/README.md/`
- `bench/results/2026-10-29-standard-gb10-bca52b7b/config.json/`
- `bench/results/2026-10-29-standard-gb10-bca52b7b/outputs.jsonl/`
- `bench/results/2026-10-29-standard-gb10-bca52b7b/provenance/chat_template.jinja/`
- `bench/results/2026-10-29-standard-gb10-bca52b7b/provenance/corpus-build.json/`
- `bench/results/2026-10-29-standard-gb10-bca52b7b/provenance/corpus-manifest.sha256/`
- `bench/results/2026-10-29-standard-gb10-bca52b7b/results.json/`
- `bench/results/2026-11-04-community-gb10-swift-1-5-iq3-xxs-2/README.md/`
- `bench/results/2026-11-04-community-gb10-swift-1-5-iq3-xxs-2/results.json/`
- `bench/results/2026-11-04-community-gb10-swift-1-5-iq3-xxs-df2/README.md/`
- `bench/results/2026-11-04-community-gb10-swift-1-5-iq3-xxs-df2/results.json/`
- `bench/results/2026-11-04-community-gb10-swift-1-5-iq3-xxs/README.md/`
- `bench/results/2026-11-04-community-gb10-swift-1-5-iq3-xxs/results.json/`
- `bench/results/2026-11-05-e2-profile-fill-advise/README.md/`
- `bench/results/2026-11-05-ple-queue-depth-p0/README.md/`
- `bench/results/2026-11-05-serve-startup-time/README.md/`
- `bench/results/2026-11-07-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs-2-stock/README.md/`
- `bench/results/2026-11-07-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs-2-stock/results.json/`
- `bench/results/2026-11-07-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs-2-tuned-VOID/README.md/`
- `bench/results/2026-11-07-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs-2-tuned-VOID/results.json/`
- `bench/results/2026-11-07-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs-2-tuned2/README.md/`
- `bench/results/2026-11-07-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs-2-tuned2/results.json/`
- `bench/results/2026-11-07-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs-3-stock2/README.md/`
- `bench/results/2026-11-07-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs-3-stock2/results.json/`
- `bench/results/2026-11-07-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs/README.md/`
- `bench/results/2026-11-07-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs/engine-log-excerpt.txt/`
- `bench/results/2026-11-07-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs/results.json/`
- `bench/results/2026-11-19-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs/README.md/`
- `bench/results/2026-11-19-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs/engine-log-excerpt.txt/`
- `bench/results/2026-11-19-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs/results.json/`
- `docs/bench-results/2026-11-19-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs-2/README.md/`
- `docs/bench-results/2026-11-19-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs-2/results.json/`
- `docs/bench-results/2026-11-19-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs-3/README.md/`
- `docs/bench-results/2026-11-19-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs-3/results.json/`
- `docs/bench-results/2026-11-19-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs-4-gb1/README.md/`
- `docs/bench-results/2026-11-19-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs-4-gb1/results.json/`
- `docs/bench-results/2026-11-19-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs-4-gb2/README.md/`
- `docs/bench-results/2026-11-19-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs-4-gb2/results.json/`
- `docs/bench-results/2026-11-19-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs-4-gc1/README.md/`
- `docs/bench-results/2026-11-19-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs-4-gc1/results.json/`
- `docs/bench-results/2026-11-19-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs-4-gc2/README.md/`
- `docs/bench-results/2026-11-19-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs-4-gc2/results.json/`
- `docs/bench-results/2026-11-19-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs-4-np1a/README.md/`
- `docs/bench-results/2026-11-19-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs-4-np1a/results.json/`
- `docs/bench-results/2026-11-19-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs-4-np1b/README.md/`
- `docs/bench-results/2026-11-19-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs-4-np1b/results.json/`
- `docs/bench-results/2026-11-19-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs-4-s1/README.md/`
- `docs/bench-results/2026-11-19-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs-4-s1/results.json/`
- `docs/bench-results/2026-11-19-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs-4-s2/README.md/`
- `docs/bench-results/2026-11-19-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs-4-s2/results.json/`
- `docs/bench-results/2026-11-19-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs-4-t1/README.md/`
- `docs/bench-results/2026-11-19-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs-4-t1/results.json/`
- `docs/bench-results/2026-11-19-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs-4-t2/README.md/`
- `docs/bench-results/2026-11-19-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs-4-t2/results.json/`
- `docs/bench-results/2026-11-19-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs-4/README.md/`
- `docs/bench-results/2026-11-19-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs-4/results.json/`
- `docs/bench-results/2026-11-19-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs/README.md/`
- `docs/bench-results/2026-11-19-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs/results.json/`
