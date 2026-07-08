# JBShield Gemma Phase2 Hidden Representation Report

作成日: 2026-07-08

## 要約

このレポートは `analysis/jbshield-gemma-phase2-hidden/` に格納された既存成果物だけを対象にした。Phase2 artifact が完全に確認できているのは Gemma2 系の2モデルのみである。`gemma-3-4b-it`、`gemma-3-12b-it`、`gemma-3-27b-it` は summary 上 failed であり、今回の内部表現比較には含めない。

主な結論は次の通り。

- `gemma-2-9b-it` は全体F1 `0.904` で、今回のGemma2対象では明確に最良である。
- `gemma-2-27b-it` は全体F1 `0.752` に留まり、サイズ増加による改善は見られない。
- `gemma-2-27b-it` の主な弱点は `saa` と `puzzler` で、特に `saa` はF1 `0.072`、`puzzler` はF1 `0.000` である。
- `gemma-2-9b-it` は `pair`、`drattack`、`puzzler` が相対的に弱いが、それ以外のfamilyではF1 `0.88` 以上を維持する。
- 両モデルとも jailbreak concept vector はfamily間で正にまとまり、toxic conceptとは負方向に分離している。したがって、concept vector構造自体はGemma2でも観察できている。
- 一方で `gemma-2-27b-it` は hidden divergence が `gemma-2-9b-it` よりかなり小さく、score/threshold上の分離が弱い。4bit量子化やモデル固有の表現スケールの影響を切り分ける必要がある。

## 分析対象

`runs.csv` によると、Gemma2の2モデルはどちらも Phase2 artifact が完全である。

| model | quantization | layers | hidden size | prompts | score rows | hidden shape |
|---|---:|---:|---:|---:|---:|---|
| gemma-2-9b-it | 8bit | 43 | 3584 | 7400 | 12500 | `[7400, 43, 3584]` |
| gemma-2-27b-it | 4bit | 47 | 4608 | 7400 | 12500 | `[7400, 47, 4608]` |

`result_summaries.csv` では以下の5モデルが並ぶが、成功しているのは Gemma2 の2モデルのみである。

- success: `gemma-2-9b-it`, `gemma-2-27b-it`
- failed: `gemma-3-4b-it`, `gemma-3-12b-it`, `gemma-3-27b-it`

したがって、本レポートではGemma2内の比較に限定する。Gemma2 vs Gemma3 の世代比較は、現状の成果物だけでは実施できない。

## 検出性能

`score_metrics.csv` を攻撃family横断でmicro集計した全体性能は以下の通り。

| model | accuracy | precision | recall | F1 | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| gemma-2-9b-it | 0.895 | 0.835 | 0.985 | 0.904 | 1217 | 94 |
| gemma-2-27b-it | 0.739 | 0.718 | 0.789 | 0.752 | 1936 | 1321 |

`gemma-2-9b-it` は高recall型で、FNは94件に抑えられている。FPは1217件あり、過検知は残るが、検知漏れを避ける設定としては安定している。

`gemma-2-27b-it` はFPとFNの両方が多い。特にFNが1321件と大きく、jailbreak側を十分に拾えていない。27Bは4bit、9Bは8bitであるため、この差はサイズ効果だけではなく量子化条件も含めて読む必要がある。

## 攻撃Family別の傾向

| attack family | gemma-2-9b-it F1 | gemma-2-27b-it F1 |
|---|---:|---:|
| autodan | 0.915 | 0.878 |
| base64 | 0.924 | 0.899 |
| drattack | 0.847 | 0.798 |
| gcg | 0.932 | 0.864 |
| ijp | 0.884 | 0.711 |
| pair | 0.827 | 0.838 |
| puzzler | 0.619 | 0.000 |
| saa | 0.949 | 0.072 |
| zulu | 0.952 | 0.738 |

`gemma-2-9b-it` は `puzzler` が最も低く、次に `pair`、`drattack` が低い。ただし `puzzler` は40サンプルのみで、他familyよりサンプル数が少ないため解釈には注意が必要である。

`gemma-2-27b-it` では `saa` と `puzzler` が大きく崩れている。`saa` はTP 42 / FN 778で、多くを見逃している。`puzzler` はTP 0 / FN 20で、今回の閾値では検出できていない。

一方で `base64` は両モデルとも比較的高い。Qwen側ではbase64が不安定だったため、Gemma2ではbase64表現または閾値設定が相対的に機能している。

## Hidden Representation の観察

今回保存されている主な内部表現は、全layerの `last_token` hidden state と、`user_prompt` span pooled hidden state である。Attention Trackerのような全token pair attentionではなく、JBShieldのconcept方向・threshold評価に必要なhidden summaryを保存している。

### 層深度

`critical_layer_summary.csv` では、両モデルとも toxic/jailbreak の重要層は中盤から後段に出ている。

| model | toxic depth mean | jailbreak depth mean | jailbreak depth range |
|---|---:|---:|---|
| gemma-2-9b-it | 0.738 | 0.780 | 0.405-0.929 |
| gemma-2-27b-it | 0.652 | 0.812 | 0.652-0.957 |

Qwen2.5 Instructでは重要層がほぼ最終層に集中していたが、Gemma2ではより広い後段範囲に分布する。特に `gemma-2-9b-it` は jailbreak depth の最小値が0.405であり、familyによって中盤層も使われている。

### 表現差分

`hidden_source_summary.csv` の平均divergenceは以下の通り。

| comparison | model | last token | user prompt |
|---|---|---:|---:|
| attack vs harmless | gemma-2-9b-it | 0.125 | 0.171 |
| attack vs harmless | gemma-2-27b-it | 0.022 | 0.035 |
| label 1 vs label 0 | gemma-2-9b-it | 0.059 | 0.119 |
| label 1 vs label 0 | gemma-2-27b-it | 0.009 | 0.023 |

`gemma-2-9b-it` は `user_prompt` 側の差分が比較的大きく、攻撃/非攻撃の分離がspan表現に出ている。これは高いrecallと整合する。

`gemma-2-27b-it` は全体にdivergenceが小さい。concept vector cosineでは構造が見えているものの、サンプル単位のスコア分離が弱いため、threshold判定で崩れている可能性が高い。

## Concept Vector の構造

`concept_vector_cosine.csv` を見ると、両モデルとも jailbreak family同士は正方向にまとまり、toxic conceptとは負方向に分離している。

| model | jailbreak-family cosine mean | toxic-vs-jailbreak cosine mean |
|---|---:|---:|
| gemma-2-9b-it | 0.416 | -0.499 |
| gemma-2-27b-it | 0.442 | -0.560 |

この点だけを見ると、`gemma-2-27b-it` のconcept vector構造は壊れていない。むしろ jailbreak-family cosine mean と toxic-vs-jailbreak cosine mean は `gemma-2-9b-it` と同等以上に見える。

したがって、`gemma-2-27b-it` の性能低下は「concept vectorが全く作れていない」というより、個別サンプルのscore margin、threshold、量子化後の表現スケール、または特定familyのcalibration適合性に起因している可能性が高い。

## モデル間比較

Gemma2内では、9B ITが27B ITを大きく上回る。これは「モデルサイズが大きいほどJBShield検出性能が高い」という仮説を支持しない。

ただし、実行条件は完全には揃っていない。`gemma-2-9b-it` は8bit、`gemma-2-27b-it` は4bitであり、量子化条件が異なる。27Bの性能低下をモデルサイズの問題として断定するには、27Bの8bitまたは別条件での再実行、または4bit条件での9B再実行が必要である。

Qwen側の結果と比べると、`gemma-2-9b-it` は `qwen2.5-7b-instruct` とほぼ同等の全体F1であり、JBShieldの対象として十分に有望である。一方、`gemma-2-27b-it` は `qwen2.5-14b-instruct` よりやや低い水準で、特定familyの崩れが全体性能を押し下げている。

## 図表

主な図は以下に出力されている。

- `figures/f1_by_model_attack.png`: モデル別・攻撃family別F1
- `figures/confusion_breakdown_by_model.png`: FP/FN/TP/TNの構成
- `figures/critical_layer_depth_by_model.png`: toxic/jailbreak conceptの重要層深度
- `figures/jailbreak_concept_vector_cosine.png`: jailbreak concept vector cosine
- `figures/hidden_layer_centroid_divergence.png`: hidden layer centroid divergence
- `figures/score_means_by_attack_family.png`: attack family別スコア平均

## 制約と注意点

- Gemma3系は成果物がないため、Gemma2 vs Gemma3 の比較はできない。
- `puzzler` は40サンプルのみであり、他familyと同じ重みで一般化するのは危険である。
- 27Bは4bit、9Bは8bitであり、モデルサイズ効果と量子化効果は分離できていない。
- 本分析はhidden summaryに基づく。全token pair attentionやattention massは含まない。
- summary上は behavior label alignment が未提供であり、生成応答の詳細な挙動ラベルとの対応分析は限定的である。

## 現時点の判断

Gemma2については、`gemma-2-9b-it` のJBShield Phase2結果は十分に分析・比較可能な品質である。検出性能、後段寄りのcritical layer、jailbreak conceptのまとまり、toxic conceptとの分離が一貫して観察できる。

`gemma-2-27b-it` はartifact自体は完全だが、`saa` と `puzzler` の崩れが大きく、防御性能の代表値としては慎重に扱うべきである。内部表現比較の対象としては有用だが、27Bが9Bより弱いという結論は、量子化条件を揃えるまでは暫定扱いにする。
