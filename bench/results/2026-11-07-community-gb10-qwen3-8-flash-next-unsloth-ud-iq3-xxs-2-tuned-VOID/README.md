# Community benchmark on NVIDIA GB10

Measured on 2026-11-07 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GB10, VRAM not read
- CPU: aarch64 (20 cores, 20 threads); installed RAM: 121.6 GiB; storage: nvme0n1 ESL01TBTLCZ-27J2-TYN nvme
- OS: Linux 7.0.0-1019-nvidia; GPU driver: 595.91.07; CUDA (driver supports up to): 13.2
- Strata: project 0.1.58, commit a0aaa6df; engine 0.1.58
- GPU detail (after the run, near idle): power_limit_w=[N/A], power_draw_w=52.92, pcie=gen1 x1, pstate=P0, temp_c=80
- Background workloads: not recorded (GPU power limit/draw read above)

## Model and configuration

- Model and quantization: qwen3.8-flash-next-unsloth-ud-iq3_xxs
- Context: 262144 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
strata --pack unsloth-ud-iq3_xxs --native Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00004.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto:32768 --spec 4 --mtp rt --max-context 262144 --kv int8 --mmap-experts --mtp-q4 all --mtp-hnorm stream --ple-io direct --kv-resident 32768 --vision --vram-reserve-mib 700 --spec-min-p 0.5 --pool-affinity auto --expert-cache-per-layer --coupled-draft
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
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 55.5 (48.8-104.6) | 0.48 (0.44-0.52) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 1686.8 (1601.9-1687.0) | not measured | 2.42 (2.42-2.54) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 2103.9 (2056.7-2107.2) | not measured | 7.67 (7.66-7.85) |
| agent session, follow-up turns | 14039 (14030-14080) at the last turn | see below | up to 64 | 3 | see below | 54.0 (46.2-76.3) | 1.84 (1.24-2.60) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 70.4 % (70.3-70.4); earlier-turn tokens read again per session: 5 (3-5) (earlier turns that the cache did not hold; the 13213 (13212-13214) new tokens of the files are read once); first turn (nothing cached) first token: 1.38 s (1.33-1.40).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 469.5 | 1.331 | 57.0 |
| 1 | 4311 | 618 | 3693 | 1444.4 | 2.598 | 56.3 |
| 2 | 5844 | 4374 | 1470 | 1088.9 | 1.386 | 54.0 |
| 3 | 9088 | 5907 | 3181 | 1412.0 | 2.291 | 46.2 |
| 4 | 11314 | 9150 | 2164 | 1238.4 | 1.785 | 66.4 |
| 5 | 14080 | 11370 | 2710 | 1373.7 | 2.013 | 65.9 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load is not recorded. Total time of the suite: 104 s.
