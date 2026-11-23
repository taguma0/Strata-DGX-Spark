# Community benchmark on NVIDIA GB10

Measured on 2026-11-23 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GB10, 122 GiB VRAM
- CPU: aarch64 (20 cores, 20 threads); installed RAM: 121.6 GiB; storage: nvme0n1 ESL01TBTLCZ-27J2-TYN nvme
- OS: Linux 7.0.0-1025-nvidia; GPU driver: 595.91.07; CUDA (driver supports up to): 13.2
- Strata: project 0.1.66, commit not a git checkout; engine 0.1.65
- GPU detail (after the run, near idle): power_limit_w=[N/A], power_draw_w=45.46, pcie=gen1 x1, pstate=P0, temp_c=72
- Background workloads: not recorded (GPU power limit/draw read above)

## Model and configuration

- Model and quantization: qwen3.8-flash-next-unsloth-ud-iq3_xxs
- Context: 262144 tokens; parallel slots: 2
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
strata --pack unsloth-ud-iq3_xxs --native Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00004.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto:32768 --spec 4 --mtp rt --max-context 262144 --kv int8 --mmap-experts --mtp-q4 all --mtp-hnorm stream --ple-io direct --kv-resident 32768 --vision --vram-reserve-mib 700 --spec-min-p 0.5 --adapt-every 1 --adapt-swaps 160 --adapt-decay 0.97 --pool-affinity auto --batch 2 --aux-cpus auto --expert-cache-per-layer --coupled-draft
```

Environment of the config: `STRATA_RUN_CONFIG_RO=1`, `STRATA_HC_Q8=1`, `STRATA_CACHE_MESSAGE_BOUNDARY=1`

## Method

Run by `python tools/strata_bench.py` against the HTTP API of a Strata server (started by the benchmark from the install configuration); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 56.6 (53.7-85.1) | 0.43 (0.40-0.49) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 1690.4 (1619.7-1700.1) | not measured | 2.41 (2.40-2.52) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 2151.9 (2114.4-2160.2) | not measured | 7.51 (7.47-7.63) |
| agent session, follow-up turns | 14375 (14360-14392) at the last turn | see below | up to 64 | 3 | see below | 57.1 (43.5-66.3) | 1.98 (1.26-2.62) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 70.2 % (70.2-70.2); earlier-turn tokens read again per session: 5 (5-5) (earlier turns that the cache did not hold; the 13511 (13511-13513) new tokens of the files are read once); first turn (nothing cached) first token: 0.96 s (0.93-0.98). Corpus: `repo_docs` (87f7a826b680) [README.md, docs/HOW_IT_WORKS.md, docs/DETAILS.md, docs/INSTALL.md, docs/MODELS.md].

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 677.0 | 0.934 | 41.5 |
| 1 | 4435 | 618 | 3817 | 1478.4 | 2.624 | 66.3 |
| 2 | 5971 | 4498 | 1473 | 1098.6 | 1.377 | 50.0 |
| 3 | 9241 | 6034 | 3207 | 1431.5 | 2.278 | 50.9 |
| 4 | 11327 | 9294 | 2033 | 1216.5 | 1.709 | 57.1 |
| 5 | 14360 | 11374 | 2986 | 1416.3 | 2.149 | 62.6 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load is not recorded. Total time of the suite: 102 s.
