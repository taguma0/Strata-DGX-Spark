<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="app/host/public/brand/strata-lockup-dark.svg">
    <img src="app/host/public/brand/strata-lockup-light.svg" alt="Strata" width="340">
  </picture>
</p>

<p align="center"><b>English</b> · <a href="docs/lang/README.zh-CN.md">简体中文</a> · <a href="docs/lang/README.ja.md">日本語</a> · <a href="docs/lang/README.de.md">Deutsch</a> · <a href="docs/lang/README.fr.md">Français</a> · <a href="docs/lang/README.es.md">Español</a> · <a href="docs/lang/README.pt-BR.md">Português</a></p>

> **This is a fork of [Niko1221/Strata](https://github.com/Niko1221/Strata).** Upstream has supported the NVIDIA DGX Spark (GB10, Arm64) since 0.2.7.1 ([DGX Spark setup](docs/DGX_SPARK_LINUX.md)); this fork tracks every upstream release and adds what the official Spark path leaves open (below). Everything below the DGX Spark section is the original README and still mostly describes x86-64 PCs. All credit for the engine goes to the original project.

## Strata on the NVIDIA DGX Spark (GB10, Arm64)

Upstream runs on the Spark too: the official path is the Terminal with a ready-made linux-aarch64 engine ([DGX Spark setup](docs/DGX_SPARK_LINUX.md), IQ3_S or NVFP4). This fork is the delta on top: the UD-IQ3_XXS build tested **with image reading enabled** (upstream leaves the picture reader untested on the Spark and keeps `--vision no`), two tuned Arm SDOT kernels, IPv6 listening and the GB10 fixes carried through every upstream release. Tested here: an ASUS GX10, the same chip as the DGX Spark, **Qwen3.8-Flash-Next UD-IQ3_XXS (Unsloth Dynamic v3.1) with images on, a 262,144-token context**. Other sizes should work but were **not tested**.

### How fast is it?

Measured on one ASUS GX10 (a GB10 machine like the DGX Spark; GB10, 20 Arm cores, 121 GiB of memory shared by CPU
and GPU), engine 0.1.62, model Qwen3.8 Flash Next UD IQ3_XXS (the install's own configuration over a live serve,
`--prefill auto:32768`), context 262,144 tokens, 3 runs per test, medians with ranges:

| Test | Median | Range |
| ---: | ---: | ---: |
| Writing answers (chat) | **59.4 tokens/s** | 52.6-104.3 |
| Chat first token | **0.49 s** | 0.40-6.90 s |
| Reads a 4K prompt | **1,728 tokens/s** | 1,614-1,731 |
| First token after a 4K prompt | 5.92 s | 2.52-9.03 s |
| Reads a 16K prompt | **2,180 tokens/s** | 2,106-2,182 |
| First token after a 16K prompt | 12.4 s | 11.9-14.6 s |
| Agent session answers | **55.3 tokens/s** | 44.8-80.0 |
| Agent conversation-cache hit | **71.0%** | 5 tokens re-read per session |

- Chat and agent answers stay at **55-59 tokens/s**. The agent's conversation cache re-reads about 71% of each
  turn's earlier tokens, which keeps agent answers fast and cheap.
- A long *fresh* prompt is still the slow part: the first token after a 16K prompt takes 11.9-14.6 s. Steady chat
  and agent answers are not affected.
- Full record with setup and caveats:
  [docs/bench-results/2026-11-19-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs-4](docs/bench-results/2026-11-19-community-gb10-qwen3-8-flash-next-unsloth-ud-iq3-xxs-4/README.md).

<details>
<summary>Earlier figures (engine 0.1.52, model swift-1.5-iq3_xxs, measured 2026-10-28) - kept as history</summary>

Same machine and suite, the October install:

| Test | Median | Range |
| ---: | ---: | ---: |
| Writing answers (chat) | **64 tokens/s** | 58.6-84.5 |
| Chat first token | **0.46 s** | 0.38-149.95 s |
| Reads a 4K prompt | **1,456 tokens/s** | 1,312-1,470 |
| First token after a 4K prompt | **2.79 s** | 2.77-3.10 s |
| Reads a 16K prompt | **1,653 tokens/s** | 1,525-1,669 |
| First token after a 16K prompt | 149 s | 9.7-157 s |
| Agent session answers | **68 tokens/s** | 60.1-77.3 |
| Agent conversation-cache hit | **70.4%** | 73 tokens re-read per session |

- Chat and agent answers stayed at **64-68 tokens/s** on the older engine and model; the agent's cache re-read
  about 70% of each turn's earlier tokens.
- The first token after a 16K prompt ranged 9.7-157 s there, against 11.9-14.6 s in the 2026-11-19 run above.
- Full record with setup and caveats:
  [bench/results/2026-10-28-community-gb10-swift-1-5-iq3-xxs](bench/results/2026-10-28-community-gb10-swift-1-5-iq3-xxs/README.md).

</details>

- The CPU expert pool runs on the performance cores only (`--pool-affinity auto`): agent decode +6.2% against the mixed P+E pool with half the workers, prompt reading unchanged ([A/B record](bench/results/2026-10-28-gb10-pool-affinity-ab/README.md)).
- Replacing the IQ3_XXS gate/up dot with a tuned Arm SDOT kernel (1.26x at 1 token, 3.11x at 4+) changed
  end-to-end prefill/decode by **nothing measurable**: the expert rows are DRAM/GPU-bound, not dot-bound
  ([A/B record](bench/results/2026-10-29-iq3xxs-neon-ab/RESULTS.md)).

<details>
<summary>Earlier figures (engine 0.1.38, measured 2026-10-05) - kept as history</summary>

One request per row, 128-token answers, greedy decoding:

| Prompt length | Reads your prompt | Writes answers |
| ---: | ---: | ---: |
| 6,190 tokens | 1,196 tokens/s | 37 tokens/s |
| 12,334 tokens | 1,294 tokens/s | 40 tokens/s |
| 24,622 tokens | 1,332 tokens/s | 39 tokens/s |
| 49,199 tokens | 1,338 tokens/s | 38 tokens/s |
| 75,047 tokens | 1,329 tokens/s | 39 tokens/s |
| 1,582 tokens | **61 tokens/s** | 42 tokens/s |
| 431 tokens | **56 tokens/s** | 15 tokens/s (the first request after start-up) |

- Writing answers stayed at about **38-42 tokens/s** however long the prompt was (up to the 75K tokens measured).
- **Short prompts were read about 20x slower** than long ones. Why is not yet known.
- One run per row, on a machine that was also in normal use. Full table, setup and caveats:
  [bench/results/2026-10-05-dgx-spark-gb10](bench/results/2026-10-05-dgx-spark-gb10/README.md).

</details>

### Install

**Plain Strata on a Spark:** the upstream [DGX Spark setup](docs/DGX_SPARK_LINUX.md) covers everything - the Terminal with a ready-made linux-aarch64 engine (`install.sh` + `strata launch`, IQ3_S or NVFP4), a source checkout, and building the engine by hand. Take that path unless you want the delta above.

**This fork (UD-IQ3_XXS + image reading):** the same steps as [Install from a source checkout](docs/DGX_SPARK_LINUX.md#install-from-a-source-checkout), with the clone URL and two flags swapped. The feed's engine does not carry this fork's changes, so setup compiles this fork's engine and the image encoder on your machine the first time.

```
git clone https://github.com/taguma0/Strata-DGX-Spark.git
cd Strata-DGX-Spark
./setup.sh --yes --family unsloth --model UD-IQ3_XXS --vision gpu --no-start
```

The guide's requirements hold here (CUDA Toolkit 13 - 13.0 through 13.3 measured, `g++`, `git`; if a run stops, rerun the same command - it continues). Budget about **85 GB of free disk** for UD-IQ3_XXS (model, image encoder and draft layer).

### Start it

The same `./setup.sh` command, without `--no-start`: on a finished install it just starts the model.

It fills the GPU expert cache with about 47 GB of experts (the expert files stay memory-mapped, not loaded into RAM): **the machine can be slow for 1-3 minutes** and the first start is the longest. When the terminal prints `ready: http://127.0.0.1:8080/v1`, open **http://127.0.0.1:8080** in a browser for the chat page (pictures can be added there), or check it:

```
curl http://127.0.0.1:8080/health
```

The API is OpenAI- and Anthropic-compatible: `http://127.0.0.1:8080/v1` (more in [DETAILS.md](docs/DETAILS.md#using-it)). If port 8080 is taken by another program, start it by hand on another port:

```
terminal/.venv/bin/python terminal/serve/server.py --engine strata --config terminal/strata-<model>.json --port 8081
```

### Keep the model settings read-only (optional)

The chat page's Settings view can change the model settings (the run config `strata-<model>.json`) from the
browser, and on a server without an API key everyone who can open the page can save them. To refuse those
saves on this PC, start with `STRATA_RUN_CONFIG_RO=1`:

```bash
STRATA_RUN_CONFIG_RO=1 ./setup.sh --start
```

The settings card disappears from the page and every save is refused (403). Chat, the API and the model
itself are untouched, and the file itself is as it was.

### What is different from the x86-64 build

- Strata's own AVX-512 / AVX-2 CPU kernels are x86-only and are **not built** on Arm64. The i-quant experts
  run on ggml-cpu's NEON code, plus two tuned Arm SDOT kernels for the IQ4_NL and IQ3_XXS gate/up rows. Both are
  bit-identical to the ggml dot they replace and 1.26-3.11x faster in isolation, yet the engine's speed is
  unchanged - the expert windows are DRAM/GPU-bound, not dot-bound
  ([A/B record](bench/results/2026-10-29-iq3xxs-neon-ab/RESULTS.md)).
- The canonical **Q2_0 pack's CPU kernel is x86-only**: on Arm64 it stops at start with a clear message. Use an
  i-quant size.
- A GB10 reports no VRAM size (CPU and GPU share one memory). Setup counts **a quarter of the RAM, at most 32 GB,** as
  VRAM so the same bytes are not counted twice.
- The Arm64 / GB10 port first landed in [`324b0ef`](https://github.com/taguma0/Strata-DGX-Spark/commit/324b0ef) (`git show 324b0ef`); every upstream release since is merged on top of it.

Original project, releases and documentation: **[github.com/Niko1221/Strata](https://github.com/Niko1221/Strata)**.

---

<h3 align="center">Run powerful AI on your own computer.</h3>
<p align="center">Chat, code and work with images, locally.</p>

<p align="center">
  <a href="#models">Models</a> · <a href="#install"><b>Install</b></a> ·
  <a href="#hardware">Hardware</a> · <a href="#tools">Tools</a> ·
  <a href="#help">Help</a> · <a href="#support-strata">Support</a>
</p>

<p align="center">
  <a href="https://github.com/Niko1221/Strata/releases/latest"><img src="https://img.shields.io/github/v/release/Niko1221/Strata?label=download&color=0ab887" alt="Download Strata"></a>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/Niko1221/Strata" alt="MIT license"></a>
  <a href="https://github.com/sponsors/Niko1221"><img src="https://img.shields.io/badge/Sponsor-EA4AAA?logo=githubsponsors&logoColor=fff" alt="Sponsor Strata on GitHub Sponsors"></a>
  <a href="https://buymeacoffee.com/strataengine"><img src="https://img.shields.io/badge/Buy_me_a_coffee-FFDD00?logo=buymeacoffee&logoColor=000" alt="Support Strata on Buy Me a Coffee"></a>
</p>

<p align="center"><sub><b>English</b> · <a href="docs/lang/README.zh-CN.md">简体中文</a> · <a href="docs/lang/README.ja.md">日本語</a> · <a href="docs/lang/README.de.md">Deutsch</a> · <a href="docs/lang/README.fr.md">Français</a> · <a href="docs/lang/README.es.md">Español</a> · <a href="docs/lang/README.pt-BR.md">Português</a> · <a href="docs/lang/README.pl.md">Polski</a></sub></p>

**Free, open source, and yours to run.** Use Strata on your desktop or with your favorite coding tools. No account needed.

## Models

We recommend **ISTA-DASLab first**, then **Unsloth** and other providers, where a compatible model fits.

<details>
<summary><img src="docs/media/models/qwen.svg" width="22" alt="" align="middle"> &nbsp;<b>Qwen3.8-Flash-Next</b></summary>

| Provider | Choices |
| --- | --- |
| **ISTA-DASLab · recommended** | Q2\_0, IQ2\_XS, IQ3\_XXS, IQ3\_S. [Coder: IQ1\_M](docs/MODELS.md#coder) for coding. |
| **Unsloth** | UD-IQ3\_XXS (the DGX Spark build measured above), UD-IQ4\_XS, UD-Q4\_K\_XL, UD-Q5\_K\_XL, UD-Q6\_K\_XL, Q8\_0 |
| **Gyro · agentionai** | [Gyro-S, Gyro-M](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-Gyro-GGUF). Lower memory use but slower. |
| **Other options** | [Swift 1.5](docs/MODELS.md#swift-15), [NVFP4](docs/NVFP4.md), [OrcaRouter](docs/ORCA.md) |

[Compare sizes, memory needs and compatibility](docs/MODELS.md).

</details>

<details>
<summary><img src="docs/media/models/qwen.svg" width="22" alt="" align="middle"> &nbsp;<b>Qwen3.6-35B-A3B</b></summary>

**Unsloth:** UD-Q4_K_XL, UD-Q6_K_XL, Q8_0. These are the currently documented packs for this model.

[Model guide and compatibility](docs/QWEN36.md).

</details>

<a id="the-other-sizes"></a>

Not sure what fits? [Choose a model for your computer](docs/MODELS.md#pick-by-memory).

## Install

<a id="option-a-the-strata-app"></a>

### Desktop app

**[Download Strata for Windows, Linux or Mac →](https://github.com/Niko1221/Strata/releases/latest)**

Install the app, choose a model, and start chatting. There is one installer for each system. On first start the app
downloads the engine for your graphics card. [Help with the installer](docs/INSTALL.md#the-app).

<sub>Files: `Strata-0.2.16-windows-x64-setup.exe` · `Strata-0.2.16-linux-x86_64.AppImage` · `Strata-0.2.16-macos-arm64.dmg`</sub>

<p align="center">
  <img src="docs/media/app-chat.png" width="900" alt="Strata desktop app with a chat and live code preview side by side">
  <br><sub>Chat, files and a live preview in one window.</sub>
</p>

<a id="option-b-the-terminal-one-line"></a>

### Terminal app

Prefer the terminal? Install Strata, then run **`strata launch`**.

<details open>
<summary><b>Install commands</b></summary>

**Linux / Mac**

```sh
curl -fsSL --connect-timeout 15 --max-time 120 https://github.com/Niko1221/Strata/releases/latest/download/install.sh -o install.sh && sh install.sh
```

**Windows PowerShell**

```powershell
irm -TimeoutSec 120 https://github.com/Niko1221/Strata/releases/latest/download/install.ps1 | iex
```

On Windows, open a new terminal afterwards. Then:

```sh
strata launch
```

Follow the prompts to download and start a model. [Terminal guide](docs/TERMINAL_LAUNCH.md).

</details>

<p align="center">
  <img src="docs/media/terminal-monitor.png" width="900" alt="Strata terminal app showing model activity, speed, memory and recent requests">
  <br><sub>Watch your model and hardware while it runs.</sub>
</p>

<a id="build-from-source"></a>

[Build from source](docs/SOURCE_INSTALL.md) · [Docker](docs/DOCKER.md) · [Full install guide](docs/INSTALL.md)

<a id="what-you-need"></a>

## Hardware

| Supported hardware | Guide |
| --- | --- |
| **NVIDIA GeForce RTX 20 / 30 / 40 / 50 series** | [NVIDIA setup](docs/INSTALL.md#what-you-need) |
| **AMD Radeon RX 7700 / 7800 / 7900, RX 9060 / 9070, AI PRO R9700** | [AMD setup](docs/AMD_HIP.md) |
| **AMD Radeon RX 6800 / 6900 · experimental** | [RX 6000 setup](docs/AMD_HIP.md#rdna2-gfx1030) |
| **Intel Arc · Linux** | [Intel setup](docs/INTEL_ARC.md) |
| **Apple Silicon · M1 or newer** | [Mac setup](docs/MAC.md) |
| **AMD Ryzen AI Max · Strix Halo** | [Strix Halo setup](docs/STRIX_HALO.md) |
| **NVIDIA DGX Spark · Linux** | [DGX Spark setup](docs/DGX_SPARK_LINUX.md) |

<a id="using-it"></a>

## Tools

**Claude Code · Codex CLI · OpenCode · oh-my-pi · pi.dev · Hermes Agent · OpenClaw**, and more through the OpenAI- and Anthropic-compatible APIs.

Run **`strata connect`** to see setup instructions for your tool. [Connection guide](docs/TERMINAL_LAUNCH.md#connect-your-apps-strata-connect).

<a id="guides"></a>
<a id="how-fast-is-it"></a>
<a id="how-does-it-work"></a>
<a id="something-went-wrong"></a>

## Help

[All guides](docs) · [Troubleshooting](docs/TROUBLESHOOTING.md) · [Report a bug](https://github.com/Niko1221/Strata/issues/new/choose) ·
[Benchmarks](docs/COMMUNITY_BENCHMARKS.md) · [How it works](docs/HOW_IT_WORKS.md) · [Contribute](.github/CONTRIBUTING.md)

## Support Strata

**Help keep local AI open to everyone.** A GitHub sponsorship or a coffee helps fund development and testing on more hardware.

<a href="https://github.com/sponsors/Niko1221"><img src="https://img.shields.io/badge/Sponsor_on_GitHub-EA4AAA?style=for-the-badge&logo=githubsponsors&logoColor=white" alt="GitHub Sponsors: sponsor Strata" height="48"></a>
<a href="https://buymeacoffee.com/strataengine"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me a Coffee: support Strata" height="48"></a>

### Loyal supporters

Thank you to everyone who supports Strata.

<!-- supporters:start (one line per public supporter, newest last; copy a line and change the GitHub name in all four places) -->
<a href="https://github.com/ryan4yin"><img src="https://github.com/ryan4yin.png?size=96" width="48" height="48" alt="ryan4yin" title="ryan4yin"></a>
<a href="https://github.com/themactep"><img src="https://github.com/themactep.png?size=96" width="48" height="48" alt="themactep" title="themactep"></a>
<a href="https://github.com/macintog"><img src="https://github.com/macintog.png?size=96" width="48" height="48" alt="macintog" title="macintog"></a>
<a href="https://github.com/bisonbet"><img src="https://github.com/bisonbet.png?size=96" width="48" height="48" alt="bisonbet" title="bisonbet"></a>
<a href="https://github.com/vizv"><img src="https://github.com/vizv.png?size=96" width="48" height="48" alt="vizv" title="vizv"></a>
<a href="https://github.com/kod88vn"><img src="https://github.com/kod88vn.png?size=96" width="48" height="48" alt="kod88vn" title="kod88vn"></a>
<a href="https://github.com/seoulworker"><img src="https://github.com/seoulworker.png?size=96" width="48" height="48" alt="seoulworker" title="seoulworker"></a>
<a href="https://github.com/jackyini"><img src="https://github.com/jackyini.png?size=96" width="48" height="48" alt="jackyini" title="jackyini"></a>
<a href="https://github.com/router0mail"><img src="https://github.com/router0mail.png?size=96" width="48" height="48" alt="router0mail" title="router0mail"></a>
<!-- supporters:end -->

All supporters and how to support Strata: [SUPPORTERS.md](SUPPORTERS.md)

[MIT license](LICENSE) · [Credits and model licenses](docs/HOW_IT_WORKS.md#credits) · [Privacy](docs/SAFETY.md)

<a id="option-c-let-your-ai-set-it-up"></a>

<sub>For AI tools and automated readers: <a href="AGENTS.md">technical reference</a> · <a href="docs/AI_SETUP.md">setup instructions</a>.</sub>

**Sponsorships, hardware partnerships or collaborations:** [niko.veit@outlook.com](mailto:niko.veit@outlook.com).

