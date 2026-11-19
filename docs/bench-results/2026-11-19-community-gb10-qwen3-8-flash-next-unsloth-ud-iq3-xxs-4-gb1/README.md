# Community benchmark on NVIDIA GB10

Measured on 2026-11-19 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GB10, VRAM not read
- CPU: aarch64 (? cores, 20 threads); installed RAM: 121.6 GiB; storage: nvme0n1 ESL01TBTLCZ-27J2-TYN nvme
- OS: Linux 7.0.0-1025-nvidia; GPU driver: 595.91.07; CUDA (driver supports up to): 13.2
- Strata: project 0.1.62, commit not a git checkout; engine not read
- GPU detail (after the run, near idle): power_limit_w=[N/A], power_draw_w=16.06, pcie=gen1 x1, pstate=P0, temp_c=66
- Background workloads: not recorded (GPU power limit/draw read above)

## Model and configuration

- Model and quantization: qwen3.8-flash-next-unsloth-ud-iq3_xxs
- Context: not read tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
strata --pack unsloth-ud-iq3_xxs --native Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00004.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto:32768 --spec 4 --mtp rt --max-context 262144 --kv int8 --mmap-experts --mtp-q4 all --mtp-hnorm stream --ple-io direct --kv-resident 32768 --vision --vram-reserve-mib 700 --spec-min-p 0.5 --adapt-every 1 --adapt-swaps 160 --adapt-decay 0.97 --pool-affinity auto
```

Environment of the config: `STRATA_RUN_CONFIG_RO=1`, `STRATA_HC_Q8=1`, `STRATA_ROUTE_TAIL_SKIP=0`

## Method

Run by `python tools/strata_bench.py` against the HTTP API of a Strata server (already running); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 60.4 (52.8-105.4) | 2.17 (1.46-2.46) |
| 4K prompt, fresh | 4019 (4019-4019) | 4012 (4012-4012) | 24 cap | 3 | 123.5 (111.5-126.6) | not measured | 2.93 (2.93-3.08) |
| 16K prompt, fresh | 16058 (16058-16058) | 16051 (16051-16051) | 24 cap | 3 | 127.7 (91.1-129.2) | not measured | 7.95 (7.93-8.11) |
| agent session, follow-up turns | 9009 (7379-9062) at the last turn | see below | up to 64 | 3 | see below | 56.9 (39.2-83.8) | 2.35 (1.67-2.67) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 99.9 % (99.9-99.9); earlier-turn tokens read again per session: 0 (0-0) (earlier turns that the cache did not hold; the 35 (28-35) new tokens of the files are read once); first turn (nothing cached) first token: 1.22 s (1.20-1.26).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 602 | 7 | 102.3 | 1.221 | 34.0 |
| 1 | 2258 | 2251 | 7 | 130.1 | 2.422 | 59.0 |
| 2 | 3964 | 3957 | 7 | 123.0 | 2.542 | 53.4 |
| 3 | 5670 | 5663 | 7 | 129.6 | 2.29 | 58.3 |
| 4 | 7366 | 7359 | 7 | 126.1 | 1.975 | 83.8 |
| 5 | 9062 | 9055 | 7 | 131.1 | 1.978 | 69.2 |

### Skipped and failed

- failed agent: {'type': 'server_error', 'message': 'this instance is draining; retry against the next instance'}

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load is not recorded. Total time of the suite: 134 s.
