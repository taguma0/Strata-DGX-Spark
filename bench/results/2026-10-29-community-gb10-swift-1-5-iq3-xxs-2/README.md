# Community benchmark on NVIDIA GB10

Measured on 2026-10-29 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GB10, VRAM not read
- CPU: aarch64 (20 cores, 20 threads); installed RAM: 121.6 GiB; storage: nvme0n1 ESL01TBTLCZ-27J2-TYN nvme
- OS: Linux 7.0.0-1019-nvidia; GPU driver: 595.91.07; CUDA (driver supports up to): 13.2
- Strata: project 0.1.52, commit 212de726; engine 0.1.52
- GPU detail (after the run, near idle): power_limit_w=[N/A], power_draw_w=48.88, pcie=gen1 x1, pstate=P0, temp_c=72
- Background workloads: not recorded (GPU power limit/draw read above)

## Model and configuration

- Model and quantization: swift-1.5-iq3_xxs
- Context: 262144 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
strata --pack swift-iq3_xxs --native Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf --ple-gguf Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill 8192 --spec 4 --mtp rt --mtp-q4 all --max-context 262144 --kv int8 --ple-io direct --kv-resident 32768 --expert-cache-per-layer --coupled-draft --mmap-experts --vision --spec-min-p 0.5 --pool-affinity auto
```

## Method

Run by `python tools/strata_bench.py` against the HTTP API of a Strata server (started by the benchmark from the install configuration); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 65.4 (54.8-83.4) | 0.44 (0.40-0.47) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 1427.3 (1372.0-1430.3) | not measured | 2.85 (2.84-2.96) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 1673.3 (1668.5-1676.8) | not measured | 9.63 (9.61-9.66) |
| agent session, follow-up turns | 13780 (13768-13791) at the last turn | see below | up to 64 | 3 | see below | 66.6 (56.9-79.5) | 2.30 (1.60-3.08) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 70.4 % (70.4-70.6); earlier-turn tokens read again per session: 73 (4-73) (earlier turns that the cache did not hold; the 12915 (12914-12916) new tokens of the files are read once); first turn (nothing cached) first token: 1.29 s (1.25-1.31).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 501.6 | 1.245 | 52.8 |
| 1 | 4263 | 620 | 3643 | 1195.1 | 3.083 | 59.5 |
| 2 | 5796 | 4256 | 1540 | 845.7 | 1.854 | 69.3 |
| 3 | 9040 | 5859 | 3181 | 1159.5 | 2.781 | 63.0 |
| 4 | 11060 | 9098 | 1962 | 975.1 | 2.048 | 68.0 |
| 5 | 13780 | 11119 | 2661 | 1101.8 | 2.452 | 79.5 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load is not recorded. Total time of the suite: 113 s.
