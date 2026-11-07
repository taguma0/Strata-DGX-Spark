# Strata standard benchmark: NVIDIA GB10

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261029T022405Z-bca52b7b`, started 2026-10-29T02:24:05Z. **Status: complete.**

Written by `strata bench run`. The machine-readable result is [results.json](results.json); every request, its raw usage and timings and the client event offsets are in [outputs.jsonl](outputs.jsonl).

## Hardware and software

- GPU: NVIDIA GB10
- CPU: aarch64; RAM: 121.6 GiB
- Storage of the model files: nvme ESL01TBTLCZ-27J2-TYN (nvme)
- OS: Ubuntu 24.04.5 LTS 7.0.0-1019-nvidia; drivers: gpu-driver 595.91.07; runtime: cuda 13.2
- Strata commit: `212de7269fe5a7577bad12b1444a2c04ef12348e`; engine 0.1.53; source build
- Background work: not measured

## Model and configuration

- Model: swift-1.5-iq3_xxs; quantization IQ3_XXS; revision not measured; source revision not measured
- Files: `Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf` `3bddaa667c750f63...`; `Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf` `b0b15f782af71eb4...`
- Track: **custom** (no notes given by the operator)
- Context 262144 tokens; server capacity 1 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0, reasoning_effort xhigh; thinking on, preserved
- Engine options: `config.json`; chat template: `provenance/chat_template.jinja`

## Method

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first nonempty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 2 min 32 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s | Capped |
| --- | --- | --- | --- | --- | --- | --- |
| short | n/a | 62.3 (59.3-64.4) n=3 | 0.59 (0.58-0.61) n=3 | 2.90 (1.62-4.81) n=3 | 7.3 (6.5-8.8) n=3 | 1/3 |
| prompt-4k | 1678.6 (1661.7-1684.5) n=3 | 64.5 (54.7-65.3) n=3 | 2.47 (2.46-2.50) n=3 | 5.11 (4.82-5.40) n=2 | 10.2 (10.1-11.8) n=3 | 1/3 |
| prompt-16k | 1911.1 (1903.6-1925.1) n=3 | 58.5 (53.3-62.4) n=3 | 8.62 (8.55-8.65) n=3 | 13.84 (13.44-14.24) n=2 | 17.3 (16.8-18.2) n=3 | 3/3 |

Cells are median (min-max) n=samples. Short is a prompt of about 80 tokens: its prompt rate is dominated by the fixed cost of a request, so it is not shown; read prompt speed from prompt-4k and prompt-16k. Capped = requests that reached the 512-token cap, reasoning and answer together. A capped request's total time is the time to the cap, not to an answer, and it has no first-answer time when it was still reasoning.

| Phase | Planned | Completed | Truncated | Failed | Skipped |
| --- | ---: | ---: | ---: | ---: | ---: |
| warmup | 1 | 0 | 1 | 0 | 0 |
| performance | 9 | 4 | 5 | 0 | 0 |
| sanity | 12 | 12 | 0 | 0 | 0 |
| concurrency | 0 | 0 | 0 | 0 | 0 |

Sanity checks: **12 of 12**.

| Case | Result | Reason |
| --- | --- | --- |
| I1 | pass | the expected JSON value |
| I2 | pass | the expected JSON value |
| I3 | pass | the expected JSON value |
| I4 | pass | the exact expected text |
| R1 | pass | the exact expected text |
| R2 | pass | the exact expected text |
| R3 | pass | the exact expected text |
| R4 | pass | the expected JSON value |
| T1 | pass | the expected tool call |
| T2 | pass | the expected tool call |
| T3 | pass | the expected tool call |
| T4 | pass | the expected tool call |

## Failures, exclusions and limitations

- No request failed.
- Excluded from speed summaries: performance-prompt-16k-r2: first_answer_s is not available for this request
- Excluded from speed summaries: performance-prompt-4k-r2: first_answer_s is not available for this request
- Prompt-prefix reuse is the shared template prefix only; the cache was not emptied between requests, and the expert cache stays warm and evolves. These are not cold-model measurements.
- Output caps include reasoning; a request that reaches its cap is `truncated` and fails its sanity check.
- The runner cannot see other programs on the machine, the repeat-stop and loop-recovery switches of the server, or the quantization repository's revision; see `missing` in results.json for every value it could not read.
- The server had answered 42 request(s) before this run.
