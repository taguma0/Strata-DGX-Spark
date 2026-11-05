# PLE direct queue-depth P0 (GB10, 2026-11-05) — deep-queue AIO arm: neutral stop, not built

Question: the single-DGX-Spark SGLang recipe runs its PLE reads at queue depth 512; the Strata PLE
`DirectFile` Linux pool is 16 blocking O_DIRECT preads (`direct_file.cpp`, "the thread count is the
queue depth"). Is 16 the binding constraint? Probe plan: measure the live PLE stream before building
an io_submit/io_uring arm (stop rule: if the PLE stall is under ~1% of the step, later phases are
not worth it).

Method: live serve `strata-swift-iq3_xxs` (`--ple-io direct`, dflash2 `--spec 6`, FIFO one
sequence), `/v1/chat/completions`, `iostat -x 1` sampled across the bench. Random high-entropy
prompts for the cold-row case (the repeated-fox prompt is a PLE row-cache artifact: aqu-sz 0.83).

| run | prompt | prefill tok/s | decode tok/s | note |
|---|---|---|---|---|
| dec512 (repetitive) | 1,258 | 1,211 | 66.9 | row cache absorbs everything |
| pf30k (repetitive) | 12,058 | 2,134 | 55.3 | PLE rows recur: disk idle (max aqu-sz 0.83) |
| randpf1 | 46,455 | 2,029 | 59.4 | fresh 4-grams, cold rows |
| randpf2 | 46,429 | 2,024 | 59.1 | fresh seed |
| randpfdec | 46,406 | 2,044 | 64.9 | 256 tokens in 46k context |

iostat over the random-prompt window (nvme0n1): **max r/s 124,899, max aqu-sz 14.0, avg aqu-sz 1.46,
max %util 72.5%, max r_await 13.8 ms**. Idle aqu-sz 0.27.

What the numbers show:

- Demand. Decode: 16 rows/token x ~65 tok/s = ~1,040 rows/s; spec-6 windows ~112 rows/step. Prefill:
  46.4k tokens x 16 rows = 743k rows read in 23 s (avg ~32k rows/s, bursts to the measured 125k).
- Supply. The 16-deep pool delivers 125k rows/s peaks (4 KiB, r_await ~0.13 ms), and the peak lands
  at aqu-sz 14.0 - the pool cap is touched only in the opening read burst of a chunk. The burst
  (a 32k chunk asks 524k rows ~ 4.2 s at peak rate) fits inside the ~16 s chunk compute, and avg
  aqu-sz 1.46 says the overlap works: reads finish long before collect blocks.
- Verdict. The PLE stall is far under 1% of the step in both decode and prefill, on this NVMe, for
  this serving shape (FIFO, one sequence). A 256-deep io_submit arm (the expert-tier precedent,
  `expert_source.cpp` io_setup kDepth 256) has no measurable seam to widen; building it would buy
  headroom the model cannot use: token rate would have to exceed ~7.8k tok/s before 125k rows/s
  binds. Stop rule fired: the deep-queue arm is NOT built. The recipe queue depth 512 answers a
  different shape (SGLang concurrent sessions, no per-token row cache).

Re-test triggers: serve concurrency > 1 sharing the PLE pool; prefill past ~5k tok/s (engine
progress); a SATA-class drive (the `generate.cpp` probe bar is 15k rows/s: a drive under the bar
WILL bind); prompts with unique 4-grams sustained past 1M rows/round.
