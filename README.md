# :shield:JBShield: Defending Large Language Models from Jailbreak Attacks through Activated Concept Analysis and Manipulation

This repository includes the data and python implementation for the paper "JBShield: Defending Large Language Models from Jailbreak Attacks through Activated Concept Analysis and Manipulation." (Accepted by the 34rd USENIX Security Symposium 2025).

[![arXiv](https://img.shields.io/badge/usenix-paper-8f1d05.svg)](https://www.usenix.org/conference/usenixsecurity25/presentation/zhang-shenyi) [![arXiv](https://img.shields.io/badge/arXiv-paper-b31b1b.svg)](https://arxiv.org/abs/2502.07557) [![license](https://img.shields.io/github/license/NISPlab/JBSHield)](https://opensource.org/licenses/MIT)

![available](./figs/usenixbadges-available.png)![functional](./figs/usenixbadges-functional.png)![reproduced](./figs/usenixbadges-reproduced.png)

This repo includes the following main components:

- Test Datasets: The datasets used in our evaluation, including Jailbreak prompts from nine distinct attack types across five target LLMs, benign prompts from the [Alpaca dataset](https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json), and harmful prompts from [AdvBench](https://github.com/llm-attacks/llm-attacks/tree/main/data/advbench) and [Hex-PHI](https://huggingface.co/datasets/LLM-Tuning-Safety/HEx-PHI) datasets. These datasets enable a comprehensive evaluation of JBShield’s detection and mitigation capabilities.

- Concept Extraction Code: Based on our proposed concept analysis methodology, we provides the implementation for extracting interpretable tokens associated with toxic and jailbreak concepts. This component aligns with our promise to share tools that facilitate understanding and manipulation of high-level concepts in LLM hidden representations.

- `JBShield-D` and `JBShield-M` Implementation: We provides complete testing scripts for evaluating the detection (`JBShield-D`) and mitigation (`JBShield-M`) components of `JBShield`. These scripts are tailored to assess performance across five target LLMs, ensuring reproducibility of the results reported in the paper.

![pipeline](./figs/pipeline.png)

:star: Note: The experimental results presented in this artifact may exhibit some variations due to differences in testing environments, the randomness in calibration set selection, and dataset size discrepancies across different jailbreak methods (DrAttack and Puzzler contain fewer samples compared to others). Despite these variations, the overall trends and effectiveness of `JBShield` remain stable, as demonstrated by the extensive evaluations conducted in our study.

Please feel free to contact us at [email](mailto:shenyizhang@whu.edu.cn) if you have any questions about this repo.

## Requirements

The minimum hardware requirement is two GPUs with at least 24GB VRAM each (e.g., RTX 3090 or RTX 4090). For optimal performance, we recommend a setup with 4 RTX 4090 GPUs (24GB VRAM each) or 1 A100 GPUs (80GB VRAM each). 

The code for `JBShield` runs with Python 3 and requires Pytorch. We recommend using `Anaconda` or `miniconda` for python. Our code has been tested with `python=3.12.8` and `torch=2.5.1` on linux. First, create a conda environment activate it:

```shell
conda create -n jbshield python=3.12
conda activate jbshield
```

Please follow [this](https://pytorch.org/get-started/locally/) to install Pytorch and `pip install` the following packages:

```shell
python -m pip install transformers fschat pandas accelerate nltk sentencepiece scikit-learn protobuf
```

## Our Dataset

Our dataset is located in `./data`. The jailbreak prompts are located in `./data/jailbreak`, while the harmful and harmless prompts can be found in `./data/harmful{}.csv` and `./data/harmless{}.csv`, respectively. 

We randomly divided the data into calibration and test sets. The calibration data is located in `./data/jailbreak/{jailbreak_name}/{model_name}_calibration.json`, while the test data can be found in `./data/jailbreak/{jailbreak_name}/{model_name}_test.json`.

## Model Preparation

We selected five target LLMs and one judge LLM, as detailed in the table below:

| Model name                                     | Link                                                         |
| ---------------------------------------------- | ------------------------------------------------------------ |
| Mistral-7B-Instruct-v0.2                       | [:hugs:[Huggingface link]](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2) |
| Llama-2-7b-chat-hf                             | [:hugs:[Huggingface link]](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) |
| Meta-Llama-3-8B-Instruct                       | [:hugs:[Huggingface link]](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) |
| vicuna-7b-v1.5                                 | [:hugs:[Huggingface link]](https://huggingface.co/lmsys/vicuna-7b-v1.5) |
| vicuna-13b-v1.5                                | [:hugs:[Huggingface link]](https://huggingface.co/lmsys/vicuna-13b-v1.5) |
| ft-mistral-7b-instruct-v0.2-sorry-bench-202406 | [:hugs:[Huggingface link]](https://huggingface.co/sorry-bench/ft-mistral-7b-instruct-v0.2-sorry-bench-202406) |

Mistral, LLaMA-2, LLaMA-3, Vicuna-7B, and Vicuna-13B are the target models in our evaluation. We need `git-lfs` to download models. Install `git-lfs` by:

```shell
# For ubuntu
sudo apt install git-lfs
```

Run the following shell commands to download models:

```shell
cd models
chmod +x ./download_models.sh
./download_models.sh
```

Place the models in the `./model`, structured as follows:

```
./models
├── Llama-2-7b-chat-hf
├── Meta-Llama-3-8B-Instruct
├── Mistral-7B-Instruct-v0.2
├── ft-mistral-7b-instruct-v0.2-sorry-bench-202406
├── vicuna-13b-v1.5
└── vicuna-7b-v1.5
```

## Interpretation

We provide the code and results for concept analysis on five target LLMs. Run the following shell commands to extract concept-related tokens:

```shell
chmod +x ./interpret.sh
./interpret.sh
```

The results are saved in `./interpret_results` (see Table 2 and Appendix A in our paper).

## JBShield-D

Run the following commands to evaluate the jailbreak detection performance of `JBShield-D` on the five LLMs:

```shell
chmod +x ./evaluate_detection.sh
./evaluate_detection.sh
```

The results are saved in `/logs/JBShield-D_{model_name}.log`. We have also provided the logs from our runs in the same directory.

### YAML Runtime Configs

For the Sec4AI experiment workflow, JBShield-D can also be run from YAML runtime configs.
The original `python detection.py --model mistral` entrypoint is still supported.

```shell
uv sync
uv run python detection.py --config configs/runtime/mistral.yml --audit-log --run-id mistral-phase1
```

To run the official reproduction set, use the manifest runner. If a model is missing,
gated, or fails because of GPU memory exhaustion, the runner records `Skip ...` and
continues to the next config.

```shell
./scripts/run_detection_config_list.sh configs/runtime/manifests/official.txt phase1-official
```

The fourth argument can restrict the jailbreak families. Phase1 in
`memo/official_implementation_experiment_procedure.md` starts with GCG only:

```shell
./scripts/run_detection_config_list.sh configs/runtime/manifests/phase1-official.txt phase1-official-gcg logs/JBShield-D_phase1_gcg.log gcg
```

On Windows or any environment without bash, use the Python runner:

```shell
uv run python scripts/run_detection_config_list.py configs/runtime/manifests/phase1-official.txt phase1-official-gcg logs/JBShield-D_phase1_gcg.log gcg
```

Phase2 writes the common parquet/safetensors artifacts used by the internal
representation experiments:

```shell
uv run python detection.py --config configs/runtime/mistral.yml --audit-log --phase2 --run-id mistral-phase2-smoke
./scripts/phase2_detection_config_list.sh configs/runtime/manifests/official.txt phase2-official
```

Experiment manifests are grouped by the Sec4AI experiment plan:

| Manifest | Purpose |
| --- | --- |
| `phase1-official.txt` | Official reproduction smoke set: Mistral-7B and Llama-3-8B. |
| `gate1-official.txt` | Compatibility alias for the Phase1 smoke set. |
| `official.txt` | Full official artifact set: Mistral, Llama-2, Llama-3, Vicuna-7B, Vicuna-13B. |
| `migration-pilot.txt` | Migration pilot: Qwen2.5/Qwen3/Llama3.1/Gemma3 small to mid models. |
| `architecture-core.txt` | Core architecture comparison: Dense GQA, MoE, MLA, hybrid mixer, local/global. |
| `qwen.txt` | Qwen scale, generation, dense/MoE, and hybrid-version comparisons. |
| `gemma.txt` | Gemma generation and dense/MoE comparisons. |
| `domain.txt` | General/code domain-adaptation comparisons. |
| `all-experiment.txt` | Full experiment set; large/gated models may be skipped. |

Examples:

```shell
./scripts/run_detection_config_list.sh configs/runtime/manifests/migration-pilot.txt jbshield-migration-pilot logs/JBShield-D_migration_pilot.log gcg
./scripts/run_detection_config_list.sh configs/runtime/manifests/architecture-core.txt jbshield-architecture logs/JBShield-D_architecture.log gcg
./scripts/run_detection_config_list.sh configs/runtime/manifests/all-experiment.txt jbshield-full logs/JBShield-D_full.log
```

Equivalent cross-platform commands:

```shell
uv run python scripts/run_detection_config_list.py configs/runtime/manifests/migration-pilot.txt jbshield-migration-pilot logs/JBShield-D_migration_pilot.log gcg
uv run python scripts/run_detection_config_list.py configs/runtime/manifests/architecture-core.txt jbshield-architecture logs/JBShield-D_architecture.log gcg
uv run python scripts/run_detection_config_list.py configs/runtime/manifests/all-experiment.txt jbshield-full logs/JBShield-D_full.log
```

For non-official target models, YAML configs use `data.jailbreak_model_name: llama-3`
so the same official Llama-3 jailbreak prompts are reused across models. This keeps the
input set fixed for cross-model comparison while changing only the target model.

Structured outputs are written to:

```text
result/{model}/runs/{run_id}/
```

Each run contains:

- `summary.json`
- `metrics.json`
- `samples.jsonl`
- `config_snapshot.yaml`

`samples.jsonl` includes per-sample toxic and jailbreak cosine scores, selected layers,
thresholds, labels, and predictions. `summary.json` includes critical layers,
thresholds, aggregate metrics, and hidden-state audit metadata.

When `--phase2` is enabled, additional artifacts are written under:

```text
outputs/phase2/{run_id}/
```

The Phase2 directory contains `prompts.parquet`, `model_metadata.parquet`,
`token_spans.parquet`, `generation_outputs.parquet`, `behavior_labels.parquet`,
`hidden_last.safetensors`, `hidden_spans.safetensors`, `jbshield_scores.parquet`,
and calibration artifacts for thresholds, concept vectors, and concept stats.

Performance (accuracy/F1-score) of `JBShield-D` across different LLMs (see Table 4 in our paper):

![detection](./figs/detection.png)

## JBShield-M

Run the following commands to evaluate the jailbreak mitigation performance of `JBShield-M` on the five LLMs:

```shell
chmod +x ./evaluate_mitigation.sh
./evaluate_mitigation.sh
```

The results are saved in `/logs/JBShield-M.log`. We have also provided the logs from our runs in the same directory.

Performance (attack success rate) of `JBShield-M` across different LLMs (see Table 7 in our paper):

![mitigation](./figs/mitigation.png)

## Citation

If you find this helpful, please cite our work:

```bibtex
@inproceedings{zhang2025jbshield,
      title={JBShield: Defending Large Language Models from Jailbreak Attacks through Activated Concept Analysis and Manipulation}, 
      author={Shenyi Zhang and Yuchen Zhai and Keyan Guo and Hongxin Hu and Shengnan Guo and Zheng Fang and Lingchen Zhao and Chao Shen and Cong Wang and Qian Wang},
      year={2025},
      booktitle={Proc. of {USENIX} Security Symposium},
      pages={8215--8234},
      publisher={{USENIX} Association},
      url={https://www.usenix.org/conference/usenixsecurity25/presentation/zhang-shenyi},
}
```

```bibtex
@misc{zhang2025jbshield,
      title={JBShield: Defending Large Language Models from Jailbreak Attacks through Activated Concept Analysis and Manipulation}, 
      author={Shenyi Zhang and Yuchen Zhai and Keyan Guo and Hongxin Hu and Shengnan Guo and Zheng Fang and Lingchen Zhao and Chao Shen and Cong Wang and Qian Wang},
      year={2025},
      eprint={2502.07557},
      archivePrefix={arXiv},
      primaryClass={cs.CR}
}
```
