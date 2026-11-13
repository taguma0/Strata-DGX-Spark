# Community benchmark on NVIDIA GB10

Measured on 2026-11-07 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GB10, VRAM not read
- CPU: aarch64 (20 cores, 20 threads); installed RAM: 121.6 GiB; storage: nvme0n1 ESL01TBTLCZ-27J2-TYN nvme
- OS: Linux 7.0.0-1019-nvidia; GPU driver: 595.91.07; CUDA (driver supports up to): 13.2
- Strata: project 0.1.58, commit a0aaa6df; engine 0.1.58
- GPU detail (after the run, near idle): power_limit_w=[N/A], power_draw_w=55.14, pcie=gen1 x1, pstate=P0, temp_c=79
- Background workloads: not recorded (GPU power limit/draw read above)

## Model and configuration

- Model and quantization: qwen3.8-flash-next-unsloth-ud-iq3_xxs
- Context: 262144 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
strata --pack unsloth-ud-iq3_xxs --native Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00004.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto:32768 --spec 4 --mtp rt --max-context 262144 --kv int8 --mmap-experts --mtp-q4 all --mtp-hnorm stream --ple-io ram --kv-resident 32768 --vision --vram-reserve-mib 700 --spec-min-p 0.5
```

Environment of the config: `STRATA_HC_Q8=1`, `STRATA_ROUTE_TAIL_SKIP=0`

## Method

Run by `python tools/strata_bench.py` against the HTTP API of a Strata server (started by the benchmark from the install configuration); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 58.3 (54.0-105.0) | 0.49 (0.45-0.53) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 1615.1 (1600.2-1634.9) | not measured | 2.52 (2.49-2.55) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 2076.4 (2044.4-2082.8) | not measured | 7.78 (7.75-7.89) |
| agent session, follow-up turns | 14026 (14013-14036) at the last turn | see below | up to 64 | 3 | see below | 58.2 (46.8-72.1) | 1.88 (1.24-2.43) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 70.2 % (70.2-70.3); earlier-turn tokens read again per session: 74 (3-75) (earlier turns that the cache did not hold; the 13213 (13213-13213) new tokens of the files are read once); first turn (nothing cached) first token: 1.38 s (1.36-1.42).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 479.7 | 1.364 | 37.4 |
| 1 | 4311 | 619 | 3692 | 1626.7 | 2.429 | 64.0 |
| 2 | 5844 | 4304 | 1540 | 1147.2 | 1.41 | 54.6 |
| 3 | 9088 | 5907 | 3181 | 1555.7 | 2.127 | 50.7 |
| 4 | 11291 | 9127 | 2164 | 1381.2 | 1.626 | 72.1 |
| 5 | 14013 | 11303 | 2710 | 1472.3 | 1.887 | 57.5 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load is not recorded. Total time of the suite: 104 s.
