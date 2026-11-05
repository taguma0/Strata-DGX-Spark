# E2: per-layer profile fill advised (GB10, 2026-11-05) — cold fill 2.9x, default on

Question: the serve startup's profile fill is the whole cold-boot cost (see
2026-11-05-serve-startup-time: warm 4.6 s, true cold 73.1 s @ 587 MB/s). The mapped-read
readahead (`advise_pairs`, `STRATA_FILL_AHEAD` default 256) existed only for the non-per-layer
fill (`!per_layer && ...` gate) - the deployed config runs `--expert-cache-per-layer`, so the
fill loop faulted 24,576 blobs serially, unadvised.

Change (src/program/generate.cpp, the profile fill): the advise gate now covers per-layer
admission. R4.2g skips the pairs a layer cannot take, so the advise masks them: a per-layer
`cap`/`taken` scan over the profile (layers = highest pair's layer + 1; caps from
`layer_slot_range`) marks `fill_admit` and advises only admitted pairs - advising a pair the
fill will not admit reads its blob for nothing. Non-per-layer path byte-identical.
`STRATA_FILL_AHEAD=0` keeps the old behavior (the A/B arm).

Method: fadvise-DONTNEED on the model GGUF (clean read-only pages drop without root;
`os.posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED)` after `sync`), identical binary, arms =
env gate only, engine booted from the live config args, fill line collected, engine killed.
This is a light cold (other files stay cached); the 09:22 OS-class cold (73.1 s) is the
absolute reference.

| arm | gate | profile fill |
|---|---|---|
| A | `STRATA_FILL_AHEAD=0` (old behavior) | 16.4 s @ 2,610 MB/s |
| B | default 256, per-layer masked | **5.7 s @ 7,463 MB/s** |

`slot 0 verified` (the fill's built-in readback check) passed in both arms; the fill is a
pure I/O-path change, no compute touched. The light-cold ratio is 2.9x; the kernel-side
WILLNEED stream lands near the warm 9,266 MB/s ceiling. Extrapolated OS-class cold:
73 s -> ~15-20 s (E2's named target, met).

Verdict: the per-layer advise rides the existing default (256) - no new flag, no inversion
needed; the deployed config gets it on the next engine start. Startup numbers must quote
page-cache state; the light-cold protocol above is reproducible without root.

Production: engine 0.1.56 rebuilt (build.sh), serve restarted by auto_load; INFO line
records the config (expert_slots=24576, spec=6, conversation_cache 8192 MiB x 4).
