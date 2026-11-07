# Community benchmark on NVIDIA GB10

Measured on 2026-11-04 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GB10, VRAM not read
- CPU: aarch64 (20 cores, 20 threads); installed RAM: 121.6 GiB; storage: nvme0n1 ESL01TBTLCZ-27J2-TYN nvme
- OS: Linux 7.0.0-1019-nvidia; GPU driver: 595.91.07; CUDA (driver supports up to): 13.2
- Strata: project 0.1.56, commit 9c951f94; engine 0.1.56
- GPU detail (after the run, near idle): power_limit_w=[N/A], power_draw_w=50.76, pcie=gen1 x1, pstate=P0, temp_c=81
- Background workloads: not recorded (GPU power limit/draw read above)

## Model and configuration

- Model and quantization: swift-1.5-iq3_xxs
- Context: 262144 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
strata --pack swift-iq3_xxs --native Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf --ple-gguf Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto:32768 --spec 4 --mtp rt --mtp-q4 all --mtp-hnorm stream --max-context 262144 --kv int8 --ple-io direct --kv-resident 32768 --expert-cache-per-layer --coupled-draft --mmap-experts --vision --spec-min-p 0.5 --pool-affinity auto
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
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 65.0 (60.1-84.0) | 0.41 (0.34-5.16) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 1835.2 (1719.9-1837.0) | not measured | 2.22 (2.22-2.37) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 2104.6 (1994.0-2156.0) | not measured | 12.32 (7.67-84.51) |
| agent session, follow-up turns | 14028 (14013-14034) at the last turn | see below | up to 64 | 3 | see below | 63.1 (57.5-82.2) | 1.92 (1.14-76.04) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 70.2 % (70.2-70.3); earlier-turn tokens read again per session: 71 (3-73) (earlier turns that the cache did not hold; the 13167 (13166-13169) new tokens of the files are read once); first turn (nothing cached) first token: 1.12 s (1.12-1.21).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 558.2 | 1.125 | 52.5 |
| 1 | 4271 | 625 | 3646 | 1565.8 | 2.383 | 68.7 |
| 2 | 5804 | 4335 | 1469 | 1197.3 | 1.265 | 57.5 |
| 3 | 9048 | 5797 | 3251 | 1445.9 | 2.293 | 59.8 |
| 4 | 11262 | 9098 | 2164 | 947.9 | 76.039 | 63.7 |
| 5 | 14028 | 11318 | 2710 | 1452.7 | 1.904 | 63.1 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load is not recorded. Total time of the suite: 260 s.
