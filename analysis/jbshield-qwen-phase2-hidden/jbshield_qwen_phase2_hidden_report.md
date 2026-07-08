# JBShield Qwen Phase2 Hidden Representation Report

作成日: 2026-07-08

## 要約

この分析は `analysis/jbshield-qwen-phase2-hidden/` に格納された既存成果物だけを対象にした。Phase2 artifact が完全に確認できているモデルは7モデルであり、`qwen3-14b` と `qwen3-32b` は今回の分析対象外である。

主な結論は次の通り。

- Qwen2.5 Base群は、JBShieldの現在のcalibration/thresholdでは検出器としてほぼ機能していない。F1は `0.005` から `0.053` 程度で、誤りの大半はFNである。
- Qwen2.5 Instruct群と Qwen3-8B は高い検出性能を示す。最良は `qwen2.5-7b-instruct` で F1 `0.905`、次点が `qwen3-8b` で F1 `0.891`。
- サイズ効果は単調ではない。Qwen2.5 Instructでは `7B > 32B > 14B` の順でF1が高く、単純に大きいほどよいとは言えない。
- Instruct/Qwen3では toxic/jailbreak concept の重要層が後段層に集中する。Base群では層0または初期層に寄っており、concept方向の抽出が不安定である。
- jailbreak family同士の concept vector cosine は Instruct/Qwen3で正にまとまる一方、toxic と jailbreak は負方向に分離する。Base群ではこの構造が弱い。
- 攻撃family別では `base64` が最も不安定で、特に `qwen2.5-14b-instruct` と `qwen2.5-32b-instruct` ではbase64のF1が低い。一方、`saa` と `zulu` は高性能モデルでは検出しやすい。

> qwen2.5-14 / 32 (base / instruct) は追加の実行時にエラーが発生している可能性がある（成果物が一部上書きされている可能性があるため再実行中...現在の暫定情報を参照）。

## 分析対象

Phase2 artifact の完全性は `runs.csv` を主に確認した。各モデルで `prompts.parquet` は7400行、`jbshield_scores.parquet` は12500行、hidden tensor は `hidden_last` と `hidden_spans/user_prompt` の両方が存在する。

| model | quantization | layers | hidden size | prompts | score rows | hidden shape |
|---|---:|---:|---:|---:|---:|---|
| qwen2.5-7b | 8bit | 29 | 3584 | 7400 | 12500 | `[7400, 29, 3584]` |
| qwen2.5-7b-instruct | 8bit | 29 | 3584 | 7400 | 12500 | `[7400, 29, 3584]` |
| qwen2.5-14b | 8bit | 49 | 5120 | 7400 | 12500 | `[7400, 49, 5120]` |
| qwen2.5-14b-instruct | 8bit | 49 | 5120 | 7400 | 12500 | `[7400, 49, 5120]` |
| qwen2.5-32b | 4bit | 65 | 5120 | 7400 | 12500 | `[7400, 65, 5120]` |
| qwen2.5-32b-instruct | 4bit | 65 | 5120 | 7400 | 12500 | `[7400, 65, 5120]` |
| qwen3-8b | 8bit | 37 | 4096 | 7400 | 12500 | `[7400, 37, 4096]` |

補足として、`result_summaries.csv` では `qwen2.5-14b-instruct` と `qwen3-8b` が failed 扱いになっているが、これは後続の失敗runで summary が上書きされた可能性が高い。Phase2 artifact 自体は完全であり、本レポートでは `outputs/phase2` 由来の分析テーブルを優先する。

## 検出性能

`score_metrics.csv` を攻撃family横断でmicro集計した全体性能は以下の通り。

| model | accuracy | precision | recall | F1 | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| qwen2.5-7b-instruct | 0.897 | 0.839 | 0.983 | 0.905 | 1181 | 108 |
| qwen3-8b | 0.880 | 0.817 | 0.980 | 0.891 | 1374 | 128 |
| qwen2.5-32b-instruct | 0.844 | 0.836 | 0.856 | 0.846 | 1052 | 897 |
| qwen2.5-14b-instruct | 0.784 | 0.750 | 0.851 | 0.797 | 1777 | 929 |
| qwen2.5-14b | 0.357 | 0.101 | 0.036 | 0.053 | 2012 | 6025 |
| qwen2.5-7b | 0.344 | 0.013 | 0.004 | 0.006 | 1972 | 6225 |
| qwen2.5-32b | 0.383 | 0.014 | 0.003 | 0.005 | 1479 | 6229 |

Base群はrecallがほぼゼロで、jailbreak側の多くを見逃している。これはJBShieldのconcept vectorやthresholdが、raw/baseモデルの表現空間では意図した方向に安定していないことを示す。

Instruct/Qwen3では recall が高い。`qwen2.5-7b-instruct` と `qwen3-8b` はFNが少なく、検知漏れを抑える方向に強い。一方でFPは1000件以上あり、特に `qwen3-8b` はrecall優先の挙動になっている。

## 攻撃Family別の傾向

Instruct/Qwen3の4モデルに限定した平均F1では、`base64` が最も低い。

| attack family | mean F1, Instruct/Qwen3 |
|---|---:|
| base64 | 0.489 |
| pair | 0.811 |
| drattack | 0.827 |
| ijp | 0.841 |
| gcg | 0.884 |
| autodan | 0.914 |
| puzzler | 0.926 |
| saa | 0.963 |
| zulu | 0.972 |

`base64` はモデル差が大きく、`qwen2.5-7b-instruct` はF1 `0.991`、`qwen3-8b` は `0.887` と高いが、`qwen2.5-14b-instruct` は `0.000`、`qwen2.5-32b-instruct` は `0.079` に落ちる。単純なサイズ差では説明できず、chat template、calibration set、量子化、モデル固有の表現差の影響を切り分ける必要がある。

`pair` と `drattack` は全体的にやや難しいが、Instruct/Qwen3ではF1 `0.79` から `0.86` 程度を維持している。`saa` と `zulu` は高性能モデルでは安定して検出される。

## Hidden Representation の観察

今回保存されている主な内部表現は、全layerの `last_token` hidden state と、`user_prompt` span pooled hidden state である。全token pair attentionではなく、JBShieldの手法に必要なhidden summaryを保存している点に注意する。

### 層深度

`critical_layer_summary.csv` では、Base群とInstruct/Qwen3で重要層の位置が明確に分かれる。

| model | toxic depth mean | jailbreak depth mean |
|---|---:|---:|
| qwen2.5-7b | 0.000 | 0.000 |
| qwen2.5-14b | 0.000 | 0.111 |
| qwen2.5-32b | 0.000 | 0.111 |
| qwen2.5-7b-instruct | 1.000 | 1.000 |
| qwen2.5-14b-instruct | 1.000 | 0.942 |
| qwen2.5-32b-instruct | 1.000 | 0.993 |
| qwen3-8b | 0.778 | 0.877 |

Instructモデルでは、toxic/jailbreak concept の分離がほぼ最終層付近に出る。Qwen3-8Bは最終層ぴったりではなく、0.78から1.0付近に広がる。Base群は層0または初期層に寄り、検出性能の低さと整合している。

### 表現差分

`hidden_source_summary.csv` の `attack_vs_harmless` 平均divergenceを見ると、Base群は `last_token` の差分が大きい一方で `user_prompt` の差分が小さい。

| model | last token | user prompt |
|---|---:|---:|
| qwen2.5-7b | 0.407 | 0.092 |
| qwen2.5-14b | 0.334 | 0.096 |
| qwen2.5-32b | 0.412 | 0.087 |
| qwen2.5-7b-instruct | 0.132 | 0.182 |
| qwen2.5-14b-instruct | 0.130 | 0.153 |
| qwen2.5-32b-instruct | 0.129 | 0.182 |
| qwen3-8b | 0.097 | 0.096 |

この差分値は単独では性能指標ではない。Base群は `last_token` の差分が大きくてもF1は低く、差分の大きさがそのまま検出可能なconcept方向を意味しない。むしろ、Instructモデルでは `user_prompt` span側にも安定した差分が出ており、calibrationされたconcept vectorと検出性能が結びついている。

## Concept Vector の構造

`concept_vector_cosine.csv` から見ると、jailbreak family同士のcosine平均はInstruct/Qwen3で高く、Base群では低い。

| model | jailbreak-family cosine mean | toxic-vs-jailbreak cosine mean |
|---|---:|---:|
| qwen2.5-7b | -0.010 | 0.069 |
| qwen2.5-14b | 0.081 | 0.242 |
| qwen2.5-32b | -0.029 | 0.104 |
| qwen2.5-7b-instruct | 0.542 | -0.635 |
| qwen2.5-14b-instruct | 0.372 | -0.545 |
| qwen2.5-32b-instruct | 0.478 | -0.615 |
| qwen3-8b | 0.412 | -0.438 |

Instruct/Qwen3では jailbreak family が同じ方向にまとまり、toxic conceptとは負方向に分離している。JBShieldが意図する「有害性」と「jailbreak性」の分離が、少なくともhidden representation上では観察できている。

Base群では jailbreak family同士のcosineが低く、toxic-vs-jailbreakも正または不安定である。これはBase群の低F1と対応しており、現在のJBShield特徴量がBaseモデルにそのまま適用しにくいことを示す。

## モデル間比較

Qwen2.5 Base vs Instructでは、Instruct化によって検出性能と内部表現の構造が大きく改善する。Base群は大半の攻撃をFNにしており、サイズを7Bから32Bに上げても改善しない。

Qwen2.5 Instruct内のサイズ比較では、7B Instructが最良で、32B Instruct、14B Instructの順になる。32Bは4bit量子化、7B/14Bは8bit量子化であり、量子化差も含めた実験結果として読む必要がある。現時点では「サイズを上げればJBShield検出が改善する」とは言えない。

Qwen3-8Bは、Qwen2.5-7B-Instructに近い高recall型の挙動を示す。F1は `0.891` で、Qwen2.5-32B-Instructより高い。critical layerは後段寄りだがQwen2.5 Instructほど最終層固定ではなく、内部表現の形成位置にやや違いがある。

## 図表

主な図は以下に出力されている。

- `figures/f1_by_model_attack.png`: モデル別・攻撃family別F1
- `figures/confusion_breakdown_by_model.png`: FP/FN/TP/TNの構成
- `figures/critical_layer_depth_by_model.png`: toxic/jailbreak conceptの重要層深度
- `figures/jailbreak_concept_vector_cosine.png`: jailbreak concept vector cosine
- `figures/hidden_layer_centroid_divergence.png`: hidden layer centroid divergence
- `figures/score_means_by_attack_family.png`: attack family別スコア平均

## 制約と注意点

- 本分析はJBShieldのhidden representation summaryを対象としており、Attention Trackerのようなattention massやtoken pair attentionは含まない。
- `behavior_score_alignment.csv` は生成されているが、summary上は behavior label alignment が未提供である。したがって、生成応答の詳細な挙動ラベルとの対応分析は限定的である。
- `result_summaries.csv` のstatusは、後続失敗runで上書きされている可能性がある。artifact完全性は `runs.csv` と `outputs/phase2` の存在を優先して判断する。
- `qwen3-14b` はOOM、`qwen3-32b` はValueErrorでPhase2 artifactがなく、今回のモデル比較には含めない。
- 32B系は4bit、7B/14B/Qwen3-8Bは8bitであり、サイズ効果と量子化効果は完全には分離できていない。

## 現時点の判断

現状確認できている7モデルについては、JBShieldで収集したhidden representationとscore artifactは分析可能な状態にある。特に Instruct/Qwen3では、検出性能、後段層へのcritical layer集中、jailbreak concept vectorのまとまり、toxic conceptとの分離が一貫して観察される。

一方、Base群は現設定では検出器としての性能が低く、内部表現分析の比較対象としては有用だが、防御性能の主張には使いにくい。次の段階では、`qwen3-14b` と `qwen3-32b` の完走、base64失敗要因の切り分け、behavior labelとの対応付けを優先するとよい。
