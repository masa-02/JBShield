# JBShield Remote Analysis Summary

- Phase2 runs: 2
- Result summaries: 5
- Artifact size scanned: 10470.187 MB
- Hidden tensor layer analysis: enabled
- Behavior label alignment: not provided
- CSV export: enabled
- Figures: 6
- Warnings: 0

## Figures

- `figures/f1_by_model_attack.png`
- `figures/critical_layer_depth_by_model.png`
- `figures/jailbreak_concept_vector_cosine.png`
- `figures/score_means_by_attack_family.png`
- `figures/confusion_breakdown_by_model.png`
- `figures/hidden_layer_centroid_divergence.png`

## Runs

```text
                              run_id     model_name  phase2_complete  num_prompts  num_score_rows  hidden_last_mb  hidden_spans_mb
jbshield-gemma-phase2-gemma-2-27b-it gemma-2-27b-it             True         7400           12500        3056.836         3056.836
 jbshield-gemma-phase2-gemma-2-9b-it  gemma-2-9b-it             True         7400           12500        2175.195         2175.195
```

## Result Metrics

```text
                              run_id     model_name  status  accuracy       f1  num_samples
jbshield-gemma-phase2-gemma-2-27b-it gemma-2-27b-it success   0.73944 0.751658      12500.0
 jbshield-gemma-phase2-gemma-2-9b-it  gemma-2-9b-it success   0.89512 0.903766      12500.0
jbshield-gemma-phase2-gemma-3-12b-it gemma-3-12b-it  failed       NaN      NaN          NaN
jbshield-gemma-phase2-gemma-3-27b-it gemma-3-27b-it  failed       NaN      NaN          NaN
 jbshield-gemma-phase2-gemma-3-4b-it  gemma-3-4b-it  failed       NaN      NaN          NaN
```

