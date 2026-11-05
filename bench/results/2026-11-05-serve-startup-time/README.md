# Strata serve startup time on GB10 (2026-11-05) — warm restart 8.3 s; cold boot 73 s is mmap-fault-bound

Question: the vLLM recipe (roozich/Qwen3.8-Flash-Spark) shipped patches 14-18 to take weight
loading 450-541 s -> 35.5 s and boot ~11 min -> 2 min 08 s. What is Strata's startup on the same
class of GB10 hardware, and is there a load bottleneck worth patches?

Method: passive first (live log blocks, no restart): `strata-swift-iq3_xxs.log` startup blocks
(2026-10-25 arena run, 2026-11-05 09:22 mmap-experts+dflash2 run). Then one owner-approved
identical-config restart (engine SIGTERM at 10:19:58, auto_load re-spawn, page cache warm).
Production model window closed ~3 min total; recovered loaded=true at 10:21.

Startup phase table (same config, 39.97 GiB / 24,576 expert slots):

| phase | cold page cache (09:22) | warm page cache (10:20 restart) |
|---|---|---|
| pack weights 266 MiB | 0.6 s | 0.1 s |
| projections 2,418 MiB | 5.1 s | 1.9 s |
| head Q5_K 437 MB | 0.3 s | 0.1 s |
| PLE direct probe | 46,152 rows/s | 49,995 rows/s |
| **profile pre-fill 24,576 slots** | **73.1 s @ 587 MB/s** | **4.6 s @ 9,266 MB/s** |
| dflash2 pack + graphs | 11 ms | 2 ms |
| verify windows (6) | ~65 ms | ~65 ms |
| **request-to-response e2e** | ~85 s class | **8.3 s** |

What the numbers show:

- The profile fill loop (`generate.cpp` ~8043-8093) is a single-threaded serial
  `blob_stable()` (mmap page faults) + `fill_slot_blocking()` (sync H2D) per slot. Warm, the same
  loop runs at 9.3 GB/s - so the H2D/copy machinery is not the bottleneck: 587 MB/s cold is the
  mmap fault stream, ~94% of the cold 73.1 s.
- Strata's architecture makes vLLM's patch class mostly structural-overkill: vLLM re-copies
  76 GiB host->device every boot (hence patches 14-18: pread iterator, name index, chunked copy,
  per-expert H2D, MTP prefilter -> 35.5 s every boot). Strata pages the expert set into the page
  cache ONCE per OS boot; every later engine restart is a 4.6 s touch, not a copy. Warm-restart
  boot 8.3 s e2e vs the vLLM recipe's 128 s full boot.
- Oct-25 arena mode logged "loaded 39.97 GiB at 5.66 GiB/s" (bulk sequential = 7.1 s). The current
  mmap-experts + per-layer + profile config (the 2026-11-04 dflash2 gate config) pays serial
  per-slot faults cold: 73.1 s once per OS boot.

Verdict: no action needed for the serving rhythm (soft restarts dominate; 8.3 s). Two named,
unexercised levers for cold-boot (OS-restart) time, in priority order:

1. E2 parallel profile fill: stage blobs on the existing 8 I/O threads (whole-blob preads,
   FileExpertSource) ahead of the fill loop, keep the single verify-slot check. Expected ~4-5x on
   the fault stream (drive sequential ~5-6 GB/s), 73 s -> ~15-20 s. Small, local change.
2. E4 `vm.nr_hugepages=20464` (sysctl.d, needs reboot): only pays on the arena path (not current
   config); it removes the arena's "MAP_HUGETLB unavailable... compaction would stall the load"
   warning and its page-fault tax. Park until an arena-mode config is chosen.

Method notes: restart path is engine SIGTERM -> server.py auto_load re-spawns on the next request
(no server restart, no config edit); phase times are self-reported per startup block; quote
warm/cold page-cache state with any startup number (`Cached:` in /proc/meminfo: 50.2 GiB before,
54.9 GiB after).
