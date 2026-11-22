# GB10 P2/P3 計測 — STRATA_DF2_PROFILE（専有ウィンドウ、2026-11-22 15:19–15:23 JST）

モデル=unsloth-UD-IQ3_XXS（本番構成、vision なし）、port 8000 単独エンジン、`STRATA_DF2_PROFILE=1`、
リクエスト 4×256 tok（デコード 49.0→54.4 tok/s）。ダンプはサーバのグレースフル停止で出る
（エンジンへ直接 TERM/INT しても出ない — server.py 経由で止めるとデストラクタが print する）。

    strata dflash2 profile (509 rounds, ms per round, total 3.987):
    native_mmvq 1.190 | gemm:gp 0.534 | gemm:kp 0.416 | gemm:fc 0.268 | gemm:mlp.down_proj 0.255
    gemm:qp 0.220 | df2_attention 0.198 | df2_rms_quant 0.165 | gemm:o_proj 0.146
    df2_kv_ring 0.149 | df2_conv_quant 0.041 | gemm:akp 0.044 | gemm:mkp 0.041
    df2_rope_table 0.070 | df2_quant_rows 0.073 | df2_conv_rms_quant 0.082
    df2_topk 0.037 | df2_norm_rope_qk 0.021 | df2_swiglu_quant 0.021 | df2_norm... | embed_ 0.007

## 読み取れること
1. **ドラフター側の頭 GEMV（`native_mmvq`、106k 語彙 Q4_0 212.9MB ドラフト head）が 1.190 ms/round =
   ドラフター合計の約 30%。** 実効 ~180 GB/s（212.9MB/1.19ms）— head GEMV は Byte で張る、時間対 Byte 比はここで確定。
2. **専門家キャッシュは全常駐（24576/24576 slots、"every expert is in the GPU caches"）で、
   定常デコードの専門家 DRAM 読取は 0 MB/token**（4 リクエスト目: OS read 18.9MB、expert reads 0.0MB）。
   → **前提の修正**: 「DRAM 律速=専門家」はキャッシュ常駐構成では成り立たない。P1 が中立だった本命の理由
   （専門家経路がそもそも DRAM を読んでいない）。パック被覆の速度価値は「専門家キャッシュが常駐しきれない
   大きいモデル/長いコンテキストでのみ」出る。
3. **P2（メイン head Q6_K 521MB/行 の二段完全検証）**: このダンプは DRAFTER のみ。メイン head（verify 側）の
   占有率は未計測だが、draft head の 180 GB/s 実効からメイン head GEMV ≈ 2.9 ms/verify 推定、round wall ≈ 20 ms/token。
   **頭系（draft+main）で round の ~25〜40% の可能性 — P2 の上限は 10〜20% 級で「あり」。ただし着手前に
   verify 側のステージ計測が必要**（df2 profile は drafter 限定。次の一手は native_head/verify に同型の
   イベント計測を足す小改修+窓 1 回、1〜2 時間）。
4. **P3（launch/sync 予算）**: プロファイルモードは graph を無効化して回る（"the round runs without its
   graph"）が、そのモードのデコード 49.0–54.4 tok/s は graph 有効の本番相当（A/B iq2_xs 47–54、iq3_xxs 常時 ~50）と
   ノイズ帯で同一。20 ステージのダンプに sync/launch の空欄もなし。→ **launch/sync 差は ≤~2%、規則（<1% 不採用）
   の精神で P3 不採用確定**。

## 手順の再現
`terminal/strata-profile.json`（本番設定 - vision + STRATA_DF2_PROFILE=1、port 8000）+ `/tmp/p23d.sh`
（pkill は `[s]erve` 型で自己マッチ回避、停止は **server.py 経由**）。エンジンへ直接 TERM/INT しても
デストラクタまで届かずダンプが出ない（2 回無駄窓を回した教訓）。

注: 事故（14:33 の 2 機目 OOM）を受け、この計測は専有ウィンドウ（承認済み）で実施。ウィンドウ計 3 回
（p23b: TERM でダンプ出ず / p23c: INT で出ず / p23d: server 停止経由で成功）。本番は毎回自動復旧、最終確認 200。

---

# 追計測（同窓 5 回目、15:34–15:37）— verify 側ステージ実測で P2 を裁定

`STRATA_VERIFY_PROFILE=1 STRATA_DECODE_TIMING=1`（両方必要 — DECODE_TIMING が print ゲート）。
本番バイナリそのまま、iq3_xxs、4×512 tok（デコード 49.3–52.0 tok/s）、250–263 windows。

    strata decode GPU stages (ms/window):
    GDN: q8+qkv/q-idx gemv 6.08 | VRAM hits 8.90 | head 2.61 | out-proj 2.36 | hc0/hc1 (norm+down+up)×2 ≈ 4.6
         rec 0.78 | shared+quant 1.27 | router 0.61 | conv 0.30 | ...
    QSA: VRAM hits 2.68 | q8+kv-idx 1.52 | hc×2 ≈ 1.3 | out-proj 0.79 | ...
    total ≈ 36.5 ms/window

## 裁定
- **P2（二段完全検証 head）: 保留**。head = **2.61 ms/window = 36.5 の 7.2%**（521MB/2.61ms ≈ 200GB/s、
  draft head の 180GB/s と同率 — 整合）。二段化で head 読取を 3 割まで削れても **net ≈ 5%**.
  effort 大 + top-K 近似の正しさリスクに見合わず。**実測で降格**（drafter 内 30% を round 比と取り違えた
  前掲 §3 の 25〜40% 見積は誤り、失礼しましたという話）。
- **前提の再修正（重要）**: 「専門家 DRAM 読取 0MB/token」は FILE 読取の話。**VRAM は GB10 では DRAM そのもの**
  で、`VRAM hits 8.90ms/window + QSA 2.68` = 専門家キャッシュ読取が依然 DRAM トラフィックの最大級
  （qkv gemv 6.08、head 2.61 を引き離す）。→ **P1 のパック被覆価値の動機は生きていた**（SDOT が中立だった
  のは計算が速くなっても bytes が減らず、律速が bytes だから — 計画書当初の見通しどおり）。
- ウィンドウ別 read-byte 順位（ms 代理）: 専門家 11.6 > qkv 6.1 > hc 約 6 > out-proj 約 3 > **head 2.6** > router 0.6。
  read-byte 級の次の候補があるとするなら専門家側（=パック解像度）で、head は 4 番目。

## 手順の完全形（窓の再現）
検証ステージ計測 = `STRATA_VERIFY_PROFILE=1` **+ `STRATA_DECODE_TIMING=1`**（どちらかだけでは出ない）。
df2 計測 = `STRATA_DF2_PROFILE=1` + **server.py 経由のグレースフル停止**。
p23e.sh/p23f.sh は /tmp に（消失時は本 PROFILE.md の手順で再構成）。
