# JBShield Remote Analysis Summary

- Phase2 runs: 7
- Result summaries: 9
- Artifact size scanned: 43120.449 MB
- Hidden tensor layer analysis: enabled
- CSV export: enabled
- Figures: 5
- Warnings: 0

## Figures

- `figures/f1_by_model_attack.png`
- `figures/critical_layer_depth_by_model.png`
- `figures/jailbreak_concept_vector_cosine.png`
- `figures/score_means_by_attack_family.png`
- `figures/hidden_layer_centroid_divergence.png`

## Runs

```text
                                   run_id           model_name  phase2_complete  num_prompts  num_score_rows  hidden_last_mb  hidden_spans_mb
         jbshield-qwen-phase2-qwen2.5-14b          qwen2.5-14b             True         7400           12500        3541.016         3541.016
jbshield-qwen-phase2-qwen2.5-14b-instruct qwen2.5-14b-instruct             True         7400           12500        3541.016         3541.016
         jbshield-qwen-phase2-qwen2.5-32b          qwen2.5-32b             True         7400           12500        4697.266         4697.266
jbshield-qwen-phase2-qwen2.5-32b-instruct qwen2.5-32b-instruct             True         7400           12500        4697.266         4697.266
          jbshield-qwen-phase2-qwen2.5-7b           qwen2.5-7b             True         7400           12500        1466.992         1466.992
 jbshield-qwen-phase2-qwen2.5-7b-instruct  qwen2.5-7b-instruct             True         7400           12500        1466.992         1466.992
            jbshield-qwen-phase2-qwen3-8b             qwen3-8b             True         7400           12500        2139.063         2139.063
```

## Result Metrics

```text
                                   run_id           model_name  status  accuracy       f1  num_samples
         jbshield-qwen-phase2-qwen2.5-14b          qwen2.5-14b success   0.35704 0.053022      12500.0
jbshield-qwen-phase2-qwen2.5-14b-instruct qwen2.5-14b-instruct success   0.78352 0.797273      12500.0
         jbshield-qwen-phase2-qwen2.5-32b          qwen2.5-32b success   0.38336 0.005419      12500.0
jbshield-qwen-phase2-qwen2.5-32b-instruct qwen2.5-32b-instruct success   0.84408 0.845990      12500.0
          jbshield-qwen-phase2-qwen2.5-7b           qwen2.5-7b success   0.34424 0.006063      12500.0
 jbshield-qwen-phase2-qwen2.5-7b-instruct  qwen2.5-7b-instruct success   0.89688 0.905032      12500.0
           jbshield-qwen-phase2-qwen3-14b            qwen3-14b  failed       NaN      NaN          NaN
           jbshield-qwen-phase2-qwen3-32b            qwen3-32b  failed       NaN      NaN          NaN
            jbshield-qwen-phase2-qwen3-8b             qwen3-8b success   0.87984 0.890732      12500.0
```

