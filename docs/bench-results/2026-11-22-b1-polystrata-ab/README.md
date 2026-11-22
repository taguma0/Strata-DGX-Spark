# B1 adoption A/B on GB10 (PolyStrata-technique shortlist, 2026-11-22)

One campaign, 9 rounds, fresh engine per round through serve (clone of the live
`terminal/strata-unsloth-ud-iq3_xxs.json`, engine 0.1.64, driver `/tmp/b1_driver.sh`,
probe `/tmp/b1_probe.py`, per-round logs `strata-ab-*.log`, suite folders
`../ab-<arm>-2026-11-22-community-*`). Suite = `tools/strata_bench.py` quick run against
the live server; probe = solo 200-token decode, 2- and 3-concurrent aggregate tok/s, and
an edited-tail reuse scenario (46 KB shared prefix, last user message replaced; the first
iteration's pair is the clean read).

## Medians

| arm | suite decode | 4K prompt | 16K prompt | agent TTFT | probe solo | agg2 | agg3 | edited-tail re-read (s) |
|---|---|---|---|---|---|---|---|---|
| base | 59.3 | 1741 | 2189 | 1.25 | 70.7 | 57.4 | 59.0 | 7.8 |
| batch2 | 35.7 | 1978 | 2311 | 0.99 | 43.9 | 69.2 | 64.7 | 7.5 |
| aux | 57.5 | 1737 | 2188 | 1.26 | 70.9 | 58.6 | 59.8 | 7.9 |
| batch2aux | 59.8 | 2031 | 2330 | 0.98 | 71.3 | 67.1 | 64.4 | 7.5 |
| base2 (control) | 58.3 | 1733 | 2174 | 1.26 | 71.0 | 58.1 | 59.9 | 7.9 |
| tail (STRATA_ROUTE_TAIL_SKIP=7) | 59.3 | 1731 | 2173 | 1.26 | 71.0 | 58.5 | 59.4 | 8.0 |
| msgb (STRATA_CACHE_MESSAGE_BOUNDARY=1) | 59.1 | 1736 | 2167 | 1.26 | 69.0 | 56.7 | 59.3 | **0.7** |
| qsa (STRATA_QSA_ROWS=1) | 56.9 | 1745 | 2185 | 1.25 | 69.9 | 57.9 | 59.5 | 7.9 |
| expdec (STRATA_EXPDEC=1) | 58.3 | 1731 | 2181 | 1.26 | 70.7 | 57.3 | 59.6 | 7.9 |

base/base2 drift: decode 1.7%, prompts 0.5%, probe solo 0.4%.

## Verdicts

- **STRATA_CACHE_MESSAGE_BOUNDARY=1 (serve env): adopt.** The edited-tail re-read falls
  from 7.5-8.0 s to 0.7 s with everything else inside the noise band. The knob only adds
  a snapshot at the previous turn boundary; reuse is still decided by the existing checks.
- **--batch 2 --aux-cpus auto: adopt for concurrent use.** 2-concurrent aggregate
  +15-19%, 3-concurrent +7-8%, 4K/16K prompt reads +13-17%, agent-turn first token
  1.26 to 0.98 s, and solo paths unchanged (suite decode, probe solo, cache hits all
  within noise).
- **--batch 2 alone: do not use.** Solo suite decode collapsed to 35.7 (range down to 25)
  and one suite first-token hit 20.6 s. Adding --aux-cpus auto removes both, so the slot
  path is sensitive to aux-thread contention on GB10; treat as a fork behaviour to watch,
  not a reason to leave batch off.
- **--aux-cpus auto alone: neutral** for solo use (no gain, no harm), but required with
  --batch.
- **STRATA_ROUTE_TAIL_SKIP=7: neutral, not adopted.** No startup ON line was printed in
  the round log, and every metric equals the controls; on this box either the skip does
  not dispatch or the miss share is too small to pay. Re-test if the expert-hit profile
  changes.
- **STRATA_QSA_ROWS=1: not adopted** (-1 to -2.5% decode, prompts unchanged).
- **STRATA_EXPDEC=1: neutral, not adopted** (all metrics inside noise; it stays an
  AMD-gfx12 default upstream and off on CUDA builds).

## What the numbers do and do not show

Byte-identity was not gated this campaign (batch-slot arithmetic; the msgb arm changes
only checkpoint placement). The probe's second scenario iteration is cache-warm in every
arm and is not comparable; only the first iteration's pair is quoted. --batch 3 was not
run. One campaign, one host state: the batch2aux prompt-read gain deserves a second
confirmation before being quoted as a batch effect.

Regenerate: clone the live config per arm as `strata-ab-<arm>.json` (extra args for
batch2/aux/batch2aux, a distinct log per arm; env arms stay on the serve command), run
`/tmp/b1_driver.sh`.
