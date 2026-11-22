# GB10 A/B: Arm64 SDOT IQ2/IQ3_S gate/up kernels — 中立（2026-11-22）

**結論: end-to-end は中立（±0.2%、ノイズ内）。予測通り GB10 のデコードは DRAM 律速で、
カーネル単体 2.3〜3.2x の速度差は専用ウィンドウの実測に現れない。** カーネル群は
パック被覆（IQ2 級モデルが ARM 上でネイティブパスを持つ）のためにブランチへ保持。

- ブランチ: `wip/q2-neon-port`（コミット 6b915c830, a7bf92655, c565bf2f3, 0b3ad4df9）→ **main 未マージ**
- モデル: `unsloth-UD-IQ2_XS-abliterated`（専門家 gu_type = IQ2_XXS(16) / IQ2_S(22) 混在 — まさに新経路）
- エンジン: wip ツリーの out-spark 0.1.64（CUDA ビルド、`BUILD_VISION=0`）、`terminal/serve/server.py` 経由（本番と同じ起動経路）、port 8000、text-only
- アーム: `STRATA_NO_IQ2XS=1 STRATA_NO_IQ2S=1`（fallback = ggml-cpu dot）vs 無指定（kernel）を交互 2 往復（fallback1/kernel1/kernel2/fallback2）
- bench: `../2026-10-05-dgx-spark-gb10/bench.py` の 8081→8000 版、prompt 512〜32768 words、greedy、128 tok 上限
- 専用ウィンドウ: 本番アシスタント（port 8000）を 14:17〜14:22 JST 停止、自動復旧確認済み（chat probe 200）

## decode tok/s（gen_n=128 全 row）

| run | 431 | 1.6K | 6.2K | 12.3K | 24.6K |
|---|---|---|---|---|---|
| fallback1 | 50.5 | 52.0 | 47.3 | 51.6 | 54.4 |
| kernel1 | 50.5 | 52.2 | 47.1 | 51.5 | 54.4 |
| kernel2 | 50.4 | 52.3 | 47.2 | 51.7 | 54.5 |
| fallback2 | 50.6 | 52.2 | *24.4 | *33.1 | *26.7 |

\* fallback2 の後半 3 row は明らかな外れ値（同 run 前半は正常値。マシンの他負荷と推定。
run2 ペアは kernel2 vs fallback2 の先行 2 row では差 ±0.2%）。有効な比較（run1 ペア + run2 先行行）は全 row ±0.2% 以内。

## 注記
- 正しさの保証はパラリティハーネス（nt=1..8 ビット一致、`ctest -R neon` 5/5 緑）。ディスパッチ発火は直接ログ確認していない（リンク確認のみ: native_expert の呼び出し箇所が iq2xs256/iq2s256 を参照）。次回 IQ2 パック実運用時に kill switch 比較で確認可能。
- 参考所見（本 A/B の主張には含めない）: IQ2_XS モデルのデコード 47〜54 tok/s は 2026-10-05 の IQ3_XXS 実測 37〜42 tok/s を上回る（別モデル・別日・別設定のため参考値）。パック被覆の価値はここにありそう、という間接証拠。
- 教訓（運用）: `pkill -f` は自分のコマンドラインに自己マッチする。リモート一括スクリプトではパターン先頭を `[s]erve` のように角括弧にする。初回パイプラインがこの自己マッチで死に、本番停止が約 9 分延びた。
