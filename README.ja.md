<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="apps/host/public/brand/strata-lockup-dark.svg">
    <img src="apps/host/public/brand/strata-lockup-light.svg" alt="Strata" width="340">
  </picture>
</p>

<p align="center"><a href="README.md">English</a> · <a href="README.zh-CN.md">简体中文</a> · <b>日本語</b> · <a href="README.de.md">Deutsch</a> · <a href="README.fr.md">Français</a> · <a href="README.es.md">Español</a> · <a href="README.pt-BR.md">Português</a></p>

> **これは [Niko1221/Strata](https://github.com/Niko1221/Strata) のフォークで、NVIDIA DGX Spark（GB10、Arm64）でも動きます。**
> 元のプロジェクトは x86-64 の PC 専用です。DGX Spark のセクションより下は元の README のままで、x86-64 の PC について
> 書かれています。エンジンの功績はすべて元のプロジェクトにあります。

## NVIDIA DGX Spark（GB10、Arm64）で Strata を動かす

このフォークは Arm64 対応を加えて、GB10 搭載機で Strata を動かせるようにします。確認したのは ASUS GX10（DGX Spark と同じチップ）です。
PC と同じように、チャット、コード作成、**画像の読み取り**ができます。ここで確認したのは、
**Qwen3.8-Flash-Next の IQ3_XXS、画像あり、262,144 トークンのコンテキスト**です。ほかのサイズも動くはずですが、**確認していません**。

### どのくらい速い？

ASUS GX10 1 台（DGX Spark と同じ GB10 搭載機。GB10、Arm コア 20、CPU と GPU で共有する 121 GiB のメモリ）、エンジン 0.1.52、モデル swift-1.5-iq3_xxs、コンテキスト 262,144 トークンで測りました。1 テスト 3 回の中央値（範囲も併記）です。

| 測定 | 中央値 | 範囲 |
| ---: | ---: | ---: |
| 答えを書く速さ（チャット） | **64 トークン/秒** | 58.6〜84.5 |
| チャットの最初のトークン | **0.46 秒** | 0.38〜149.95 秒 |
| 4K プロンプトを読む | **1,456 トークン/秒** | 1,312〜1,470 |
| 4K の後の最初のトークン | **2.79 秒** | 2.77〜3.10 秒 |
| 16K プロンプトを読む | **1,653 トークン/秒** | 1,525〜1,669 |
| 16K の後の最初のトークン | 149 秒 | 9.7〜157 秒 |
| エージェントセッションの答え | **68 トークン/秒** | 60.1〜77.3 |
| 会話キャッシュのヒット率 | **70.4%** | セッションあたり 73 トークン再読み取り |

- 答えを書く速さはチャットで **64〜68 トークン/秒**。エージェントでは会話キャッシュが各ターンの過去のトークンの約 70% を再利用し、応答を速く軽く保ちます。
- 長い**新規**プロンプトがいちばん弱い点です。16K の新規プロンプトの後の最初のトークンまでは 10〜157 秒の幅がありました（チャットとエージェントの安定速度には影響しません）。
- 条件と注意点をこめた全体の記録:
  [bench/results/2026-10-28-community-gb10-swift-1-5-iq3-xxs](bench/results/2026-10-28-community-gb10-swift-1-5-iq3-xxs/README.md)。

<details>
<summary>過去の数値（エンジン 0.1.38、2026-10-05 計測）- 歴史記録として維持</summary>

各行 1 回のリクエスト、答えは 128 トークン、貪欲デコードです。

| プロンプトの長さ | プロンプトを読む速さ | 答えを書く速さ |
| ---: | ---: | ---: |
| 6,190 トークン | 1,196 トークン/秒 | 37 トークン/秒 |
| 12,334 トークン | 1,294 トークン/秒 | 40 トークン/秒 |
| 24,622 トークン | 1,332 トークン/秒 | 39 トークン/秒 |
| 49,199 トークン | 1,338 トークン/秒 | 38 トークン/秒 |
| 75,047 トークン | 1,329 トークン/秒 | 39 トークン/秒 |
| 1,582 トークン | **61 トークン/秒** | 42 トークン/秒 |
| 431 トークン | **56 トークン/秒** | 15 トークン/秒（起動後の最初のリクエスト） |

- 答えを書く速さは、プロンプトがどれだけ長くても約 **38〜42 トークン/秒**で変わりませんでした。
- **短いプロンプトは、長いものより約 20 倍遅く**読まれました。理由はまだ分かっていません。
- 各行 1 回の測定で、マシンが通常利用されている状態で測りました。全体の表、条件、注意点は
  [bench/results/2026-10-05-dgx-spark-gb10](bench/results/2026-10-05-dgx-spark-gb10/README.md) にあります。

</details>

### インストール

必要なもの：CUDA Toolkit 13（`/usr/local/cuda`、13.0〜13.3 で確認）が入った DGX Spark、`g++`、`git`、IQ3_XXS 用に約 **85 GB の空きディスク**
（モデル、画像エンコーダ、ドラフト層）。Arm64 用のビルド済みエンジンはないので、初回は setup が**エンジンと画像エンコーダを
このマシンでコンパイル**します（CUDA カーネルが多いので時間がかかります）。そのあとモデルを Hugging Face からダウンロードします。
途中で止まっても、同じコマンドをもう一度実行すれば続きから再開します。

```
git clone https://github.com/shi3z/Strata-DGX-Spark.git
cd Strata-DGX-Spark
./setup.sh --yes --family qwen --model IQ3_XXS --vision gpu --no-start
```

### 起動

```
./run-iq3_xxs.sh
```

約 40 GB の expert をメモリに読み込むので、**1〜3 分のあいだマシンが重くなることがあります**（初回がいちばん長いです）。
ターミナルに `ready: http://127.0.0.1:8080/v1` と出たら、ブラウザで **http://127.0.0.1:8080** を開くとチャット画面です
（画像もここで追加できます）。動作確認は次のコマンドでもできます。

```
curl http://127.0.0.1:8080/health
```

API は OpenAI / Anthropic 互換です：`http://127.0.0.1:8080/v1`（詳細は [DETAILS.md](docs/DETAILS.md#using-it)）。
8080 番ポートがほかのプログラムに使われているときは、別のポートで手動起動します。

```
.venv/bin/python serve/server.py --engine strata --config strata-iq3_xxs.json --port 8081
```

### モデル設定を変更不能にする (任意)

チャット画面の Settings は、モデル設定（run config の `strata-<model>.json`）をブラウザから変更できます。API key なしで空いているサーバーでは、画面を開ける人は誰でも保存できます。これを拒否したい場合は `STRATA_RUN_CONFIG_RO=1` で起動します:

```bash
STRATA_RUN_CONFIG_RO=1 ./setup.sh --start
```

すべての保存が拒否 (403) され、設定カードは画面から消えます。チャット、API、モデルそれ自身は無邊で、ファイルは変わりません。

### ほかの端末からチャット画面を開く（Tailscale）

DGX Spark の tailnet アドレスだけで待ち受け（`0.0.0.0` ではありません）、ほかの端末が使う名前を許可します。

```
TS_IP=$(tailscale ip -4)
TS_NAME=$(tailscale status --json | python3 -c "import json,sys; print(json.load(sys.stdin)['Self']['DNSName'].rstrip('.'))")
STRATA_ALLOWED_HOSTS="$TS_NAME,$TS_IP" .venv/bin/python serve/server.py --engine strata \
  --config strata-iq3_xxs.json --host "$TS_IP" --port 8081
```

そのあと、tailnet 内の端末から `http://<TS_NAME>:8081/` を開きます。到達できるのは tailnet 内の端末だけですが、
**tailnet 内の誰でもモデルを使えます**。キーを必須にするには、`strata-iq3_xxs.json` に `"api_key": "<長い秘密の文字列>"`
を追加してください（チャット画面がキーを聞いてきます）。キーなしで `127.0.0.1` や tailnet アドレスの外に公開しないでください。

### x86-64 版との違いと、未対応のこと

- Strata 独自の AVX-512 / AVX-2 の CPU カーネルは x86 専用で、Arm64 では**ビルドされません**。i-quant の expert
  （IQ2_XS、IQ3_XXS、IQ3_S など）は代わりに ggml-cpu の NEON コードで動くので、**Arm 向けに調整したカーネルはまだありません**。
- 本来の **Q2_0 pack の CPU カーネルは x86 専用**です。Arm64 では起動時にメッセージを出して止まります。i-quant のサイズを使ってください。
- GB10 は VRAM の大きさを報告しません（CPU と GPU が 1 つのメモリを共有）。setup は同じバイトを二重に数えないよう、
  **RAM の 4 分の 1（最大 32 GB）**を VRAM として扱います。
- このマシンで失敗する setup のテスト（`test_setup_amd`、`test_setup_golden`、`test_setup_choices` の 2 件、
  `test_setup_unsloth` の 1 件）は、元のコードでもここで同じように失敗します。
- 変更は 1 つのコミットにまとまっています：`git show 324b0ef`

元のプロジェクト、リリース、ドキュメント：**[github.com/Niko1221/Strata](https://github.com/Niko1221/Strata)**

---

<h3 align="center">あなた自身のパソコンで、高性能な AI を。</h3>
<p align="center">チャット、コーディング、画像の処理を、すべてローカルで。</p>

<p align="center">
  <a href="#モデル">モデル</a> · <a href="#インストール"><b>インストール</b></a> ·
  <a href="#ハードウェア">ハードウェア</a> · <a href="#ツール">ツール</a> ·
  <a href="#ヘルプ">ヘルプ</a> · <a href="#strata-を支援する">支援</a>
</p>

<p align="center">
  <a href="https://github.com/Niko1221/Strata/releases/latest"><img src="https://img.shields.io/github/v/release/Niko1221/Strata?label=download&color=0ab887" alt="Strata をダウンロード"></a>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/Niko1221/Strata" alt="MIT ライセンス"></a>
  <a href="https://buymeacoffee.com/strataengine"><img src="https://img.shields.io/badge/Buy_me_a_coffee-FFDD00?logo=buymeacoffee&logoColor=000" alt="Buy Me a Coffee で Strata を支援する"></a>
</p>

<p align="center"><sub><a href="README.md">English</a> · <a href="README.zh-CN.md">简体中文</a> · <b>日本語</b> · <a href="README.de.md">Deutsch</a> · <a href="README.fr.md">Français</a> · <a href="README.es.md">Español</a> · <a href="README.pt-BR.md">Português</a> · <a href="README.pl.md">Polski</a></sub></p>

**無料でオープンソース。自分の環境で動かせます。** デスクトップで、またはお好みのコーディングツールと一緒に Strata を使えます。アカウントは不要です。

## モデル

まず **ISTA-DASLab** をおすすめします。次に **Unsloth** などの提供元です。メモリに収まる互換モデルがある場合に選べます。

<details>
<summary><img src="docs/media/models/qwen.svg" width="22" alt="" align="middle"> &nbsp;<b>Qwen3.8-Flash-Next</b></summary>

| 提供元 | 選択肢 |
| --- | --- |
| **ISTA-DASLab · おすすめ** | Q2\_0, IQ2\_XS, IQ3\_XXS, IQ3\_S。[Coder: IQ1\_M](docs/MODELS.md#coder)（コーディング向け）。 |
| **Unsloth** | UD-IQ4\_XS, UD-Q4\_K\_XL, UD-Q5\_K\_XL, UD-Q6\_K\_XL, Q8\_0 |
| **Gyro · agentionai** | [Gyro-S, Gyro-M](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-Gyro-GGUF)。メモリ使用量は少ないですが、動作は遅くなります。 |
| **その他の選択肢** | [Swift 1.5](docs/MODELS.md#swift-15), [NVFP4](docs/NVFP4.md), [OrcaRouter](docs/ORCA.md) |

[サイズ、必要なメモリ、互換性を比べる](docs/MODELS.md)。

</details>

<details>
<summary><img src="docs/media/models/qwen.svg" width="22" alt="" align="middle"> &nbsp;<b>Qwen3.6-35B-A3B</b></summary>

**Unsloth:** UD-Q4_K_XL, UD-Q6_K_XL, Q8_0。現在、このモデルで文書化されているパックです。

[モデルガイドと互換性](docs/QWEN36.md)。

</details>

<details>
<summary><img src="docs/media/models/deepseek.svg" width="22" alt="" align="middle"> &nbsp;<b>DeepSeekV4.1-Flash-1017</b></summary>

**デスクトップアプリ**と**ターミナル**で使えます。メモリに収まらない分は SSD から読み込むため、空きディスク容量がたくさん必要です。

| セット（IST-DAS） | ダウンロード | 必要な空き容量 |
| --- | ---: | ---: |
| 1.75 bpw | 239 GB | 約245 GB |
| 2.00 bpw | 257 GB | 約265 GB |
| 2.25 bpw | 274 GB | 約280 GB |

NVIDIA RTX 30・40・50 シリーズ（Windows と Linux）と、AMD Radeon AI PRO R9700（Linux）で動きます。セットアップがファイルと DeepSeek エンジンをダウンロードし、モデルを起動します。

```sh
strata launch --family deepseek41 --model ist-das-2.00bpw --plan
```

Strata 0.2.5 で 2.00 bpw のセットを使って測定しました。「出力」は回答の速さ、「プロンプト」は Strata が入力を読み込む速さです。

| ハードウェア | 出力 | プロンプト |
| --- | ---: | ---: |
| RTX 5070 12 GB, 64 GB RAM | 4〜6 トークン/秒 | 約 110 トークン/秒 (4K) |
| RTX 3060 12 GB, 46 GB RAM | 9〜11 トークン/秒 | 約 155 トークン/秒 (4K) |
| 1x Radeon AI PRO R9700, 251 GB RAM | 21〜25 トークン/秒 | 320〜360 トークン/秒 (8K) |
| 4x Radeon AI PRO R9700 | 74〜77 トークン/秒 | 400〜450 トークン/秒 (8K) |

[メモリ、ディスク、詳細](docs/DEEPSEEK41.md) · [リリースノート](https://github.com/Niko1221/Strata/releases)

</details>

<a id="the-other-sizes"></a>

どれが動くかわからない場合は、[お使いのパソコンに合うモデルを選ぶ](docs/MODELS.md#pick-by-memory)。

## インストール

<a id="option-a-the-strata-app"></a>

### デスクトップアプリ

**[Windows、Linux、Mac 用の Strata をダウンロード →](https://github.com/Niko1221/Strata/releases/latest)**

アプリをインストールし、モデルを選んで、チャットを始めましょう。インストーラーはシステムごとに1つです。初回起動時に、アプリがお使いのグラフィックスカード用のエンジンをダウンロードします。[インストーラーのヘルプ](docs/INSTALL.md#the-app)。

<sub>ファイル: `Strata-0.2.7-windows-x64-setup.exe` · `Strata-0.2.7-linux-x86_64.AppImage` · `Strata-0.2.7-macos-arm64.dmg`</sub>

<p align="center">
  <img src="docs/media/app-chat.png" width="900" alt="チャットとライブのコードプレビューを並べて表示した Strata デスクトップアプリ">
  <br><sub>チャット、ファイル、ライブプレビューを1つのウィンドウで。</sub>
</p>

<a id="option-b-the-terminal-one-line"></a>

### ターミナルアプリ

ターミナルのほうが好みですか？ Strata をインストールして、**`strata launch`** を実行します。

<details open>
<summary><b>インストールコマンド</b></summary>

**Linux / Mac**

```sh
curl -fsSL https://github.com/Niko1221/Strata/releases/latest/download/install.sh -o install.sh && sh install.sh
```

**Windows PowerShell**

```powershell
irm https://github.com/Niko1221/Strata/releases/latest/download/install.ps1 | iex
```

Windows では、インストール後に新しいターミナルを開きます。そのあと:

```sh
strata launch
```

画面の案内に従って、モデルをダウンロードして起動します。[ターミナルガイド](docs/TERMINAL_LAUNCH.md)。

</details>

<p align="center">
  <img src="docs/media/terminal-monitor.png" width="900" alt="モデルの動作、速度、メモリ、最近のリクエストを表示する Strata ターミナルアプリ">
  <br><sub>動作中のモデルとハードウェアの状態を確認できます。</sub>
</p>

<a id="build-from-source"></a>

[ソースからビルド](docs/SOURCE_INSTALL.md) · [Docker](docs/DOCKER.md) · [インストールガイド（全文）](docs/INSTALL.md)

<a id="what-you-need"></a>

## ハードウェア

| 対応ハードウェア | ガイド |
| --- | --- |
| **NVIDIA GeForce RTX 20 / 30 / 40 / 50 シリーズ** | [NVIDIA のセットアップ](docs/INSTALL.md#what-you-need) |
| **AMD Radeon RX 7700 / 7800 / 7900、RX 9060 / 9070、AI PRO R9700** | [AMD のセットアップ](docs/AMD_HIP.md) |
| **AMD Radeon RX 6800 / 6900 · 試験的** | [RX 6000 のセットアップ](docs/AMD_HIP.md#rdna2-gfx1030) |
| **Intel Arc · Linux** | [Intel のセットアップ](docs/INTEL_ARC.md) |
| **Apple Silicon · M1 以降** | [Mac のセットアップ](docs/MAC.md) |
| **AMD Ryzen AI Max · Strix Halo** | [Strix Halo のセットアップ](docs/STRIX_HALO.md) |

<a id="using-it"></a>

## ツール

**Claude Code · Codex CLI · OpenCode · oh-my-pi · pi.dev · Hermes Agent · OpenClaw**。OpenAI 互換と Anthropic 互換の API を通じて、ほかのツールも使えます。

**`strata connect`** を実行すると、お使いのツールの設定方法が表示されます。[接続ガイド](docs/TERMINAL_LAUNCH.md#connect-your-apps-strata-connect)。

<a id="guides"></a>
<a id="how-fast-is-it"></a>
<a id="how-does-it-work"></a>
<a id="something-went-wrong"></a>

## ヘルプ

[すべてのガイド](docs) · [トラブルシューティング](docs/TROUBLESHOOTING.md) · [不具合を報告](https://github.com/Niko1221/Strata/issues/new/choose) ·
[ベンチマーク](docs/COMMUNITY_BENCHMARKS.md) · [仕組み](docs/HOW_IT_WORKS.md) · [貢献する](CONTRIBUTING.md)

## Strata を支援する

**ローカル AI をみんなに開かれたものに。** コーヒー1杯が、開発と、より多くのハードウェアでのテストの支えになります。

<a href="https://buymeacoffee.com/strataengine"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me a Coffee: Strata を支援する" height="48"></a>

[MIT ライセンス](LICENSE) · [クレジットとモデルのライセンス](docs/HOW_IT_WORKS.md#credits) · [プライバシー](docs/SAFETY.md)

<a id="option-c-let-your-ai-set-it-up"></a>

<sub>AI ツールと自動リーダー向け: <a href="AGENTS.md">技術リファレンス</a> · <a href="docs/AI_SETUP.md">セットアップ手順</a></sub>

**スポンサー、ハードウェア提携、共同開発のご相談:** [niko.veit@outlook.com](mailto:niko.veit@outlook.com).
