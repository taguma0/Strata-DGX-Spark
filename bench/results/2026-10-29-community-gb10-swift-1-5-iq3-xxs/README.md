# Community benchmark on NVIDIA GB10

Measured on 2026-10-29 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GB10, VRAM not read
- CPU: aarch64 (20 cores, 20 threads); installed RAM: 121.6 GiB; storage: nvme0n1 ESL01TBTLCZ-27J2-TYN nvme
- OS: Linux 7.0.0-1019-nvidia; GPU driver: 595.91.07; CUDA (driver supports up to): 13.2
- Strata: project 0.1.52, commit 212de726; engine 0.1.53
- GPU detail (after the run, near idle): power_limit_w=[N/A], power_draw_w=52.27, pcie=gen1 x1, pstate=P0, temp_c=72
- Background workloads: not recorded (GPU power limit/draw read above)

## Model and configuration

- Model and quantization: swift-1.5-iq3_xxs
- Context: 262144 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
strata --pack swift-iq3_xxs --native Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf --ple-gguf Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill 8192 --spec 4 --mtp rt --mtp-q4 all --max-context 262144 --kv int8 --ple-io direct --kv-resident 32768 --expert-cache-per-layer --coupled-draft --mmap-experts --vision --spec-min-p 0.5 --pool-affinity auto
```

## Method

Run by `python tools/strata_bench.py` against the HTTP API of a Strata server (already running); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 65.8 (56.1-80.0) | 0.46 (0.42-0.50) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 1647.0 (1568.3-1647.5) | not measured | 2.48 (2.48-2.60) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 1900.9 (1882.7-1903.2) | not measured | 8.49 (8.48-8.57) |
| agent session, follow-up turns | 13785 (13772-13789) at the last turn | see below | up to 64 | 3 | see below | 63.0 (57.6-73.0) | 1.83 (1.20-2.63) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 70.5 % (70.4-70.5); earlier-turn tokens read again per session: 73 (73-73) (earlier turns that the cache did not hold; the 12914 (12914-12915) new tokens of the files are read once); first turn (nothing cached) first token: 1.28 s (1.24-1.33).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 506.5 | 1.236 | 46.8 |
| 1 | 4265 | 625 | 3640 | 1404.9 | 2.632 | 63.0 |
| 2 | 5798 | 4258 | 1540 | 1096.7 | 1.441 | 65.4 |
| 3 | 9042 | 5862 | 3180 | 1397.0 | 2.316 | 65.6 |
| 4 | 11063 | 9099 | 1964 | 1234.8 | 1.628 | 59.9 |
| 5 | 13785 | 11122 | 2663 | 1365.3 | 1.988 | 72.1 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load is not recorded. Total time of the suite: 105 s.
