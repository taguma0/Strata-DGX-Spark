# What this fork changes over upstream Strata

Upstream is [Niko1221/Strata](https://github.com/Niko1221/Strata), kept as the `upstream` remote (a local clone of it on the build host). The
[DGX Spark section of the README](../README.md#what-is-different-from-the-x86-64-build) states the
substantive deltas; this page is the file-by-file version of the same claim, so a reviewer can check them
against the code instead of trusting the prose.

155 files changed, 38228 insertions(+), 865 deletions(-) as measured on 2026-11-19 (the page measures itself, so a docs edit shifts the insertion count by its own size). The benchmark records under `bench/results/` and `docs/bench-results/` are indexed in [DGX_SPARK_BENCH_RECORDS.md](DGX_SPARK_BENCH_RECORDS.md); the code-and-docs delta is **69 files (11 added, 49 modified, 9 deleted)**.

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

The area lists below were written from the second command and the bench index from the third - re-derive
both, do not hand-extend them, after a big upstream merge.

## Engine (14 files)

The Arm64 / GB10 kernel layer - the heart of the fork.

- **Added** `engine/src/kernels/cpu/iq3xxs_neon.cpp`, `engine/src/kernels/cpu/iq4nl_neon.cpp` and their
  `engine/src/kernels/cpu/iq3xxs_neon_parity.cpp` / `iq4nl_neon_parity.cpp` twins: the two tuned
  SDOT kernels for the IQ3_XXS and IQ4_NL gate/up rows. Bit-identical to the ggml dots they replace,
  1.26x (1 token) to 3.11x (4+) in isolation; end-to-end speed is unchanged because the expert windows are
  DRAM/GPU-bound ([A/B record](../bench/results/2026-10-29-iq3xxs-neon-ab/RESULTS.md)). They were weighed
  against dropping them on 2026-11-19 and **kept** by the owner: the parity twins make them a free substitution
  (bit-identical by construction), and they are the only Arm i-quant tuning this build carries.
- **Added** `engine/src/kernels/cpu/portable_stubs.cpp`: `ggml_fp16_to_fp32` and friends, needed because the
  0.2.16 release engine no longer links `ggml-base`.
- **Modified** `engine/CMakeLists.txt`, `engine/cmake/backends/cpu.cmake`: build the NEON sources on Arm64 and leave
  the x86-only AVX-512/AVX-2 kernels out of the Arm build.
- **Modified** `engine/src/kernels/cpu/native_expert.cpp`, `engine/src/kernels/cpu/expert_layout.cpp`,
  `engine/include/strata/kernels/cpu/expert.hpp`, `engine/include/strata/kernels/cpu/iq_avx2.hpp`,
  `engine/src/core/native_dense.cpp`, `engine/src/core/verify.cpp`: the i-quant rows' GB10 path - the NEON
  include propagation, the f16 scale decode, and the dense-layer GB10 fixes.
- **Modified** `engine/src/program/generate.cpp` - **E2, model loading**: the mapped-read readahead
  (`advise_pairs`, `STRATA_FILL_AHEAD`, default 256) used to gate off for `--expert-cache-per-layer`, which is
  what the Spark config deploys, so the cold profile fill faulted 24,576 blobs serially at 587 MB/s (73.1 s;
  warm the same loop runs 9.3 GB/s - the fault stream is the whole cold boot). The advise now covers the
  per-layer fill, masked to the pairs a layer actually admits: cold fill 2.9x faster, default on
  ([E2 record](../bench/results/2026-11-05-e2-profile-fill-advise/README.md),
  [startup phases](../bench/results/2026-11-05-serve-startup-time/README.md): warm restart 8.3 s).

## Serve: the VRAM load gate (5 files)

- `terminal/serve/server.py` (the `vram_fit_start_need` wiring), `terminal/serve/server_parts/engines.py`,
  `terminal/strata_control/vram_fit.py` (+**added** `terminal/serve/test_vram_load_gate.py`, **modified**
  `terminal/serve/telemetry.py`): the model **load** gate. `vram_need`
  raises the gate line above `min_free_vram_mib` - as a number, or `"auto"`, which prices the model's own VRAM
  floor once per start (`vram_fit.start_need_mib()`: context, KV window, drafter, reserve; the higher line wins
  and a hand-set floor is never undercut); `load_wait_s` lets an unloaded model wait for the card to come back
  instead of failing (503 + `Retry-After` as before). On GB10 the "free VRAM" the gate waits on comes from the
  CUDA driver read `terminal/serve/telemetry.py` adds (`cuMemGetInfo` through the primary context, ~0.1 s per
  read, cached) because NVML reports no
  frame buffer at all on unified memory (`nvidia-smi` prints `[N/A]`).
  The **unload** side is deliberately unchanged from upstream - on this machine an unload is expensive (the
  46.84 GiB GPU expert cache must be refilled; see the E2 entry above), so the fork invests entirely on the
  load side: gate, price and wait, not new eviction.

## Serve: read-only run config (`server.py` + `test_server.py`)

- `STRATA_RUN_CONFIG_RO=1` makes the run config `strata-<model>.json` read-only: GET `/config` answers 404 (the
  web app already hides the Model settings card when the endpoint does not answer - upstream behavior the fork
  reuses), POST `/config` answers 403 for every client, the API-key holder included, and a plain line says the
  mode is on when serve starts. Upstream lets any device that can open the chat page save the model settings -
  on a server without an API key that is everyone on the LAN. The `RunConfigReadOnly` tests in
  `terminal/serve/test_server.py` pin the off/on pair (off: the card loads and saves as always; on: 404/403, the
  file byte-for-byte untouched). The README section "Keep the model settings read-only" is the user-facing story.
  The same file also carries two IPv6 behavior tests - upstream switches to `AF_INET6` on a colon address, so
  `::1` answering and `::` dual-stack are pins on upstream's mechanism (the fork had its own IPv6 stack until
  it converged to upstream's on 2026-11-19, `5a6736bfc`).

## Setup and tests (18 files)

- `terminal/setup.py`: ARM64 detection as a test-overridable global (golden-style tests set `setup.ARM64 = False`
  to describe an x86 PC on any host), NEON as the Arm baseline in place of the AVX checks, the machine model
  suffixed "(Arm64, NEON)". The GB10 VRAM count itself is upstream's (`gpus()` handles the `[N/A]` as unified
  memory, Strix-Halo style); an earlier fork rule counting a quarter of the RAM became dead code and was
  removed once upstream's own handling landed in the sync.
- `terminal/strata_control/drafting.py`: Swift (UkisAI's fine-tune of the same weights - same hidden size,
  vocabulary and experts layout) joins `EXPERIMENTAL_FAMILIES`: a start that NAMES `dflash2` runs the DFlash2
  drafter, Auto keeps the MTP. Upstream refuses Swift there (no acceptance numbers, MTP only); the fork's
  acceptance record is [docs/SWIFT-DF2-GB10-2026-11-04.md](SWIFT-DF2-GB10-2026-11-04.md) (gx10, 2026-11-04).
- `terminal/tools/test_setup_*.py` (12): eleven carry one pin - `setup.ARM64 = False`, or the
  `mock.patch.object` form in `test_setup_feed_engine` - "these tests describe an x86-64 PC whatever machine
  runs them" (the CUDA repo URL picks sbsa on Arm, `cpu_floor`'s gate and the feed-engine path read the
  global). The golden test also normalizes the host-derived `vision.threads` out of the recorded Unsloth
  UD-Q4_K_XL golden configs on non-WIN hosts; `test_setup_sycl`'s entire delta is dropping upstream's
  xe/i915 expert-profile ranking test.
- `terminal/tests/terminal/test_drafting*.py` (2): not adaptations - behavior tests of `EXPERIMENTAL_FAMILIES`
  above: naming `dflash2` on a swift model runs the drafter with no note, the auto path keeps the MTP, and
  `supported_model(experimental=True)` accepts the swift names.
- `terminal/tools/test_no_llama_download.py`, `terminal/tests/terminal/test_bench_standard.py`: fallout fixes
  for the Removed-from-upstream pass below - both read a deleted file directly (the `backends.yml` workflow,
  the `BENCH.bat` launcher) and failed with `FileNotFoundError` until the dead entries came out.

## App (3 files)

`app/host/server/native-settings.mjs`, `app/host/src/lib/drafting.ts` and `app/host/tests/drafting.test.mjs`:
the draft-layer sentences the fork's serve prints, mirrored in the app.

## Docs (13 files)

`docs/DGX_SPARK_LINUX.md` (the upstream-official Spark path, retested here), `docs/DETAILS.md` and
`docs/VRAM_ELASTIC.md` (the load-gate additions), **added** `docs/SWIFT-DF2-GB10-2026-11-04.md` (the DF2
drafter record on GB10), the 7 `docs/lang/` README translations for the fork's intro rewrite, and this page
with its bench index (`docs/DGX_SPARK_FORK.md`, `docs/DGX_SPARK_BENCH_RECORDS.md`).

## Removed from upstream (9 files)

GitHub plumbing that does not run (or does not make sense) on a fork: the 4 `ISSUE_TEMPLATE` files,
`pull_request_template.md`, and the `backends.yml`, `docs.yml`, `snapshot.yml` workflows; plus
`tools/BENCH.bat`. Upstream remains the place to report issues. The deletions had fallout the first pass
missed: two tests read the deleted files and broke (fixed as noted under Setup and tests), `setup.py`'s
benchmark prompt offered the removed `BENCH.bat` (it names `./bench.sh` only now), and `vram_fit.py` carried
a `ctx_label` helper nothing had ever called (removed with the rest).

## Root and tools (6 files)

`build.sh` (**added**, the fork's GB10 build entry), `.gitignore`, `README.md` (the DGX Spark section),
`SUPPORTERS.md`; `tools/requant_dense_gguf.py` (**added**, requantizing the dense layers) and
`tools/docker/build.sh`.

The bench records this fork adds (40 folders of A/B and community evidence) are indexed separately:
[docs/DGX_SPARK_BENCH_RECORDS.md](DGX_SPARK_BENCH_RECORDS.md).
