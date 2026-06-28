import argparse
import time
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_curve

from audit import write_json, write_jsonl, write_yaml
from phase2_core import write_phase2_outputs
from repr_cache import EXTRACTION_VERSION, write_hidden_cache
from config import model_paths
from config import path_harmful_test, path_harmless_test, path_harmful_calibration, path_harmless_calibration
from runtime_config import load_runtime_config
from utils import load_model, load_ori_prompts, get_jailbreak_prompts
from utils import get_sentence_embeddings
from utils import interpret_difference_matrix
from utils import cosine_similarity


DEFAULT_JAILBREAKS = ["ijp", "gcg", "saa", "autodan", "pair", "drattack", "puzzler", "zulu", "base64"]
LEGACY_VECTOR_ORDER = ["gcg", "puzzler", "saa", "autodan", "drattack", "pair", "ijp", "base64", "zulu"]


def _score_detection(model, tokenizer, embeddings, calibration_embedding, calibration_vector, threshold):
    labels = []
    scores = []
    for embed in embeddings:
        vec, _ = interpret_difference_matrix(
            model,
            tokenizer,
            embed,
            calibration_embedding,
            return_tokens=False,
        )
        score = cosine_similarity(vec, calibration_vector).item()
        scores.append(score)
        labels.append(1.0 if score >= threshold else 0.0)
    return labels, scores


def _metrics(tp, fp, fn, tn):
    total = tp + fp + fn + tn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": (tp + tn) / total if total else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "num_samples": int(total),
    }


def _run_dir(output_dir, model_name, run_id):
    return Path(output_dir) / model_name / "runs" / run_id


def _parse_jailbreaks(value):
    if value is None:
        return None
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return list(value)


def _validate_jailbreaks(jailbreaks):
    unknown = [name for name in jailbreaks if name not in DEFAULT_JAILBREAKS]
    if unknown:
        raise ValueError(f"Unknown jailbreak family: {unknown}. Supported: {DEFAULT_JAILBREAKS}")


def find_critical_layer(embeddings1, embeddings2):
    '''
    Find the layer with the minimum average cosine similarity between the two sets of embeddings
    
    Args:
    - embeddings1: first set of embeddings
    - embeddings2: second set of embeddings
    - visualize: whether to visualize the results

    Returns:
    - cosine_similarities: list of average cosine similarities for each layer
    - seleced_layer_index: index of the selected layer
    '''
    num_layers = len(embeddings1)
    cosine_similarities = []
    seleced_layer_index = 0
    min_cosine = 1

    # if the number of embeddings in two sets are not equal, truncate the longer one.
    if len(embeddings1[0]) != len(embeddings2[0]):
        min_len = min(len(embeddings1[0]), len(embeddings2[0]))
        embeddings1 = [emb[:min_len] for emb in embeddings1]
        embeddings2 = [emb[:min_len] for emb in embeddings2]

    for layer_index in range(num_layers):
        layer_embeddings1 = torch.stack(embeddings1[layer_index])
        layer_embeddings2 = torch.stack(embeddings2[layer_index])
        
        layer_cosine = []

        # Calculate the cosine similarity between each pair of embeddings
        for emb1 in layer_embeddings1:
            for emb2 in layer_embeddings2:
                cos_sim = cosine_similarity(emb1, emb2)
                layer_cosine.append(cos_sim.item())
        
        # Calculate the average cosine similarity for the layer
        avg_cosine = sum(layer_cosine) / len(layer_cosine)
        cosine_similarities.append(avg_cosine)
        if avg_cosine < min_cosine:
            min_cosine = avg_cosine
            seleced_layer_index = layer_index

    return cosine_similarities, seleced_layer_index


def get_thershold(scores1, scores2):
    '''
    Get the optimal threshold for the given scores

    Args:
    - scores1: first set of scores (eg [0.2, 0.3, 0.4, 0.2, 0.5])
    - scores2: second set of scores (eg [0.8, 0.9, 0.4, 0.6, 1.0])

    Returns:
    - optimal_threshold: optimal threshold to distinguish the two sets of scores (eg 0.6)
    '''
    scores1 = np.array(scores1)
    scores2 = np.array(scores2)

    scores = np.concatenate((scores1, scores2))
    labels = np.array([0] * len(scores1) + [1] * len(scores2))
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(labels, scores)

    # Find the optimal threshold
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    # If the optimal threshold is not finite, set it to the average of the min and max scores
    if not np.isfinite(optimal_threshold):
        optimal_threshold = (np.min(scores1) + np.max(scores2)) / 2

    return optimal_threshold


def find_optimal_threshold(model, tokenizer, calibration_embeddings1, calibration_embeddings2, base_calibration_embedding, calibration_vector):
    '''
    Find the optimal threshold to distinguish the two sets of embeddings

    Args:
    - model: model to get embeddings from
    - tokenizer: tokenizer to use for encoding prompts
    - calibration_embeddings1: first set of embeddings
    - calibration_embeddings2: second set of embeddings
    - calibration_embedding: base embedding to compare against
    - calibration_vector: base vector to compare against
    - layer_index: index of the layer to find the optimal threshold for

    Returns:
    - thershold: optimal threshold to distinguish the two sets of embeddings
    '''
    v_cs1 = []
    v_cs2 = []

    for embed1, embed2 in zip(
        calibration_embeddings1,
        calibration_embeddings2,
    ):
        v1, _ = interpret_difference_matrix(
            model,
            tokenizer,
            embed1,
            base_calibration_embedding,
            return_tokens=False,
        )
        v2 = torch.zeros_like(v1)
        # v2, _ = interpret_difference_matrix(
        #     model,
        #     tokenizer,
        #     embed2,
        #     base_calibration_embedding,
        #     return_tokens=False,
        # )
        v_cs1.append(cosine_similarity(v1, calibration_vector).item())
        v_cs2.append(cosine_similarity(v2, calibration_vector).item())

    thershold = get_thershold(v_cs1, v_cs2)

    return thershold


def detection_judge(model, tokenizer, embeddings1, calibration_embedding, calibration_vector, threshold, return_scores=False):
    results, scores = _score_detection(
        model,
        tokenizer,
        embeddings1,
        calibration_embedding,
        calibration_vector,
        threshold,
    )
    if return_scores:
        return results, scores
    return results


def _classify_group(name):
    """Map an embedding group name to (attack_family, label=attack_present)."""
    if name in ("calibration_harmless", "test_harmless"):
        return "none", 0
    if name in ("calibration_harmful", "test_harmful"):
        return "direct_harmful", 0
    family = name.replace("calibration_", "").replace("test_", "")
    return family, 1


def detection(
    model_name,
    update_vectors=False,
    jailbreaks=None,
    output_dir="result",
    audit_log=False,
    run_id=None,
    runtime_config=None,
    config_path=None,
    model_path=None,
    data_config=None,
    chat_template=None,
    cache_hidden=False,
    cache_dir="representations",
    cache_last_k=5,
    phase2=False,
    phase2_output_dir="outputs/phase2",
    strict_spans=False,
):
    start_time = time.time()
    jailbreaks = jailbreaks or DEFAULT_JAILBREAKS
    _validate_jailbreaks(jailbreaks)
    run_id = run_id or f"{model_name}-phase1"
    run_dir = _run_dir(output_dir, model_name, run_id)
    runtime_config = runtime_config or {}
    data_config = data_config or {}
    cache_model_id = model_path or model_name
    if audit_log:
        run_dir.mkdir(parents=True, exist_ok=True)
        snapshot = runtime_config or {
            "model": {"name": model_name, "path": model_path},
            "runtime": {"output_dir": output_dir},
            "audit": {"enabled": audit_log, "run_id": run_id},
            "detection": {"jailbreaks": jailbreaks, "update_vectors": update_vectors},
        }
        if config_path:
            snapshot["config_path"] = str(config_path)
        write_yaml(run_dir / "config_snapshot.yaml", snapshot)

    # Load model
    effective_model_paths = dict(model_paths)
    if model_path:
        effective_model_paths[model_name] = model_path
    model, tokenizer = load_model(model_name, effective_model_paths)

    # Load data
    # harmful_prompts, harmless_prompts = load_ori_prompts(path_harmful, path_harmless)
    harmful_test_path = data_config.get("harmful_test", path_harmful_test)
    harmless_test_path = data_config.get("harmless_test", path_harmless_test)
    harmful_calibration_path = data_config.get("harmful_calibration", path_harmful_calibration)
    harmless_calibration_path = data_config.get("harmless_calibration", path_harmless_calibration)
    _, harmless_prompts_test = load_ori_prompts(harmful_test_path, harmless_test_path)
    harmful_prompts_calibration, harmless_prompts_calibration = load_ori_prompts(harmful_calibration_path, harmless_calibration_path)
    jailbreak_model_name = data_config.get("jailbreak_model_name", model_name)
    jailbreak_prompts_calibration = get_jailbreak_prompts(jailbreak_model_name, jailbreaks, split="calibration")
    jailbreak_prompts_test = get_jailbreak_prompts(jailbreak_model_name, jailbreaks, split="test")

    # Remove for potential data leakage
    # # Get embdddings for prompts
    # print("Get embeddings for harmful and harmless prompts...")
    # harmful_embeddings = get_sentence_embeddings(harmful_prompts, model, model_name, tokenizer)
    # harmless_embeddings = get_sentence_embeddings(harmless_prompts, model, model_name, tokenizer)
    # # Mean embeddings for harmful and harmless prompts
    # mean_harmful_embedding = []
    # mean_harmless_embedding = []
    # for i in range(len(harmful_embeddings)):
    #     mean_harmful_embedding.append(torch.mean(torch.stack(harmful_embeddings[i]), dim=0))
    #     mean_harmless_embedding.append(torch.mean(torch.stack(harmless_embeddings[i]), dim=0))

    # Embeddings for calibration prompts
    print("Get embeddings for calibration prompts...")
    embedding_audit = {}
    cache_splits = []
    phase2_groups = []

    def _write_cache(name, prompts, embeddings, tail_embeddings, tail_lens):
        family, label = _classify_group(name)
        metadata_rows = [
            {
                "sample_id": f"{model_name}:{name}:{idx}",
                "base_request_id": f"{name}:{idx}",
                "runtime_model_name": model_name,
                "prompt_index": idx,
                "attack_family": family,
                "label": label,
                "tail_len": int(tail_lens[idx]),
                "prompt_chars": len(prompt),
            }
            for idx, prompt in enumerate(prompts)
        ]
        split_dir = write_hidden_cache(
            cache_dir,
            cache_model_id,
            name,
            embeddings,
            metadata_rows,
            tail_embeddings=tail_embeddings,
            extraction_version=EXTRACTION_VERSION,
        )
        cache_splits.append({"split": name, "path": str(split_dir), "num_samples": len(prompts)})

    def collect_embeddings(name, prompts):
        if not (cache_hidden or audit_log or phase2):
            return get_sentence_embeddings(
                prompts, model, model_name, tokenizer, chat_template=chat_template
            )

        result = get_sentence_embeddings(
            prompts,
            model,
            model_name,
            tokenizer,
            return_audit=audit_log,
            chat_template=chat_template,
            last_k=cache_last_k,
            return_tail=cache_hidden,
            return_spans=phase2,
            strict_spans=strict_spans,
        )
        cursor = 0
        embeddings = result[cursor]
        cursor += 1
        tail_embeddings = None
        tail_lens = None
        if cache_hidden:
            tail_embeddings = result[cursor]
            tail_lens = result[cursor + 1]
            cursor += 2
        if audit_log:
            embedding_audit[name] = result[cursor]
            cursor += 1
        if phase2:
            span_embeddings = result[cursor]
            span_records = result[cursor + 1]
            phase2_groups.append(
                {
                    "group_name": name,
                    "prompts": prompts,
                    "embeddings": embeddings,
                    "span_embeddings": span_embeddings,
                    "span_records": span_records,
                }
            )
        if cache_hidden:
            _write_cache(name, prompts, embeddings, tail_embeddings, tail_lens)
        return embeddings

    calibration_harmless_embeddings = collect_embeddings("calibration_harmless", harmless_prompts_calibration)
    calibration_harmful_embeddings = collect_embeddings("calibration_harmful", harmful_prompts_calibration)
    # Mean embeddings for harmful and harmless prompts
    mean_harmful_embedding = []
    mean_harmless_embedding = []
    for i in range(len(calibration_harmless_embeddings)):
        mean_harmful_embedding.append(torch.mean(torch.stack(calibration_harmful_embeddings[i]), dim=0))
        mean_harmless_embedding.append(torch.mean(torch.stack(calibration_harmless_embeddings[i]), dim=0))
    if update_vectors:
        # Save mean embeddings for harmful and harmless prompts when the first time to run this script
        (Path("vectors") / model_name).mkdir(parents=True, exist_ok=True)
        torch.save(mean_harmful_embedding, './vectors/{}/mean_harmful_embedding.pt'.format(model_name))
        torch.save(mean_harmless_embedding, './vectors/{}/mean_harmless_embedding.pt'.format(model_name))

    calibration_attack_embeddings = {
        jailbreak: collect_embeddings(
            f"calibration_{jailbreak}", jailbreak_prompts_calibration[jailbreak]
        )
        for jailbreak in jailbreaks
    }

    # Embeddings for test prompts
    print("Get embeddings for test prompts...")
    test_harmless_embeddings = collect_embeddings("test_harmless", harmless_prompts_test)
    # test_harmful_embeddings = get_sentence_embeddings(harmful_prompts_test, model, model_name, tokenizer)

    test_attack_embeddings = {
        jailbreak: collect_embeddings(f"test_{jailbreak}", jailbreak_prompts_test[jailbreak])
        for jailbreak in jailbreaks
    }


    # Find the critical layers
    _, seleced_safety_layer_index = find_critical_layer(calibration_harmful_embeddings, calibration_harmless_embeddings)
    print("Selected layer index for toxic concept detection: {}".format(seleced_safety_layer_index))
    seleced_jailbreak_layer_indexs = {}
    for jailbreak in jailbreaks:
        _, layer_index = find_critical_layer(
            calibration_attack_embeddings[jailbreak], calibration_harmful_embeddings
        )
        seleced_jailbreak_layer_indexs[jailbreak] = layer_index
        print("Selected layer index for {} jailbreak concept detection: {}".format(jailbreak, layer_index))
    

    # Get calibration vectors and thersholds
    print("Get calibration vectors and thersholds...")
    calibration_safety_vector, delta_safety = interpret_difference_matrix(
        model,
        tokenizer,
        calibration_harmful_embeddings[seleced_safety_layer_index],
        calibration_harmless_embeddings[seleced_safety_layer_index],
        return_tokens=False,
    )

    calibration_jailbreak_vectors = {}
    delta_jailbreaks = {}
    for jailbreak in jailbreaks:
        layer_index = seleced_jailbreak_layer_indexs[jailbreak]
        calibration_vector, delta_jailbreak = interpret_difference_matrix(
            model,
            tokenizer,
            calibration_attack_embeddings[jailbreak][layer_index],
            calibration_harmful_embeddings[layer_index],
            return_tokens=False,
        )
        calibration_jailbreak_vectors[jailbreak] = calibration_vector
        delta_jailbreaks[jailbreak] = delta_jailbreak * -1

    # Evaluate the jailbreak detection
    print("Evaluate the jailbreak detection...")
    ## For transfer attack
    # for idx_calibration in tqdm(range(len(jailbreaks))):
    #     for idx_test in range(len(jailbreaks)):
    metrics_by_attack = {}
    sample_records = []
    totals = {"tp": 0.0, "fp": 0.0, "fn": 0.0, "tn": 0.0}
    thresholds = {}
    for jailbreak in tqdm(jailbreaks):
        print("Calibration data from : ", jailbreak, "- Test data from : ", jailbreak)
        calibration_embedding = calibration_attack_embeddings[jailbreak]
        test_embedding = test_attack_embeddings[jailbreak]
        labels_jb = []
        labels_harmless = []
        # Find the optimal threshold for the jailbreak detection
        thershold_safety = find_optimal_threshold(
            model,
            tokenizer,
            calibration_embedding[seleced_safety_layer_index],
            calibration_harmless_embeddings[seleced_safety_layer_index],
            mean_harmless_embedding[seleced_safety_layer_index],
            calibration_safety_vector,
        )
        thershold_jailbreak = find_optimal_threshold(
            model,
            tokenizer,
            calibration_embedding[seleced_jailbreak_layer_indexs[jailbreak]],
            calibration_harmful_embeddings[seleced_jailbreak_layer_indexs[jailbreak]],
            mean_harmful_embedding[seleced_jailbreak_layer_indexs[jailbreak]],
            calibration_jailbreak_vectors[jailbreak],
        )
        thresholds[jailbreak] = {
            "toxic": thershold_safety,
            "jailbreak": thershold_jailbreak,
        }
        if update_vectors:
            # Save thersholds for mitigation when the first time to run this script
            torch.save(thershold_safety, './vectors/{}/thershold_safety_{}.pt'.format(model_name, jailbreak))
            torch.save(thershold_jailbreak, './vectors/{}/thershold_jailbreak_{}.pt'.format(model_name, jailbreak))
        # Detect the jailbreak prompts
        print("Num of test jailbreak prompts: ", len(test_embedding[seleced_safety_layer_index]))
        results_safety, scores_safety = detection_judge(
            model,
            tokenizer,
            test_embedding[seleced_safety_layer_index],
            mean_harmless_embedding[seleced_safety_layer_index],
            calibration_safety_vector,
            thershold_safety,
            return_scores=True,
        )
        results_jailbreak, scores_jailbreak = detection_judge(
            model,
            tokenizer,
            test_embedding[seleced_jailbreak_layer_indexs[jailbreak]],
            mean_harmful_embedding[seleced_jailbreak_layer_indexs[jailbreak]],
            calibration_jailbreak_vectors[jailbreak],
            thershold_jailbreak,
            return_scores=True,
        )
        # Detect the harmless prompts
        print("Num of test harmless prompts: ", len(test_harmless_embeddings[seleced_safety_layer_index][:len(test_embedding[seleced_safety_layer_index])]))
        results_harmless_safety, scores_harmless_safety = detection_judge(
            model,
            tokenizer,
            test_harmless_embeddings[seleced_safety_layer_index][:len(test_embedding[seleced_safety_layer_index])],
            mean_harmless_embedding[seleced_safety_layer_index],
            calibration_safety_vector,
            thershold_safety,
            return_scores=True,
        )
        results_harmless_jailbreak, scores_harmless_jailbreak = detection_judge(
            model,
            tokenizer,
            test_harmless_embeddings[seleced_jailbreak_layer_indexs[jailbreak]][:len(test_embedding[seleced_jailbreak_layer_indexs[jailbreak]])],
            mean_harmful_embedding[seleced_jailbreak_layer_indexs[jailbreak]],
            calibration_jailbreak_vectors[jailbreak],
            thershold_jailbreak,
            return_scores=True,
        )
        # If result_safety and result_jailbreak are all 1.0, this prompt is judged as jailbreak
        for idx, (result_safety, result_jailbreak, score_safety, score_jailbreak) in enumerate(zip(results_safety, results_jailbreak, scores_safety, scores_jailbreak)):
            prediction = 1.0 if result_safety == 1.0 and result_jailbreak == 1.0 else 0.0
            labels_jb.append(prediction)
            if audit_log or phase2:
                sample_records.append({
                    "id": f"{model_name}:{jailbreak}:test:jailbreak:{idx}",
                    "prompt_id": f"{model_name}:test_{jailbreak}:{idx}",
                    "split": "test",
                    "attack_family": jailbreak,
                    "label": 1,
                    "toxic_score": score_safety,
                    "jailbreak_score": score_jailbreak,
                    "prediction": int(prediction),
                    "thresholds": thresholds[jailbreak],
                    "layers": {
                        "toxic": seleced_safety_layer_index,
                        "jailbreak": seleced_jailbreak_layer_indexs[jailbreak],
                    },
                })
        for idx, (result_safety, result_jailbreak, score_safety, score_jailbreak) in enumerate(zip(results_harmless_safety, results_harmless_jailbreak, scores_harmless_safety, scores_harmless_jailbreak)):
            prediction = 1.0 if result_safety == 1.0 and result_jailbreak == 1.0 else 0.0
            labels_harmless.append(prediction)
            if audit_log or phase2:
                sample_records.append({
                    "id": f"{model_name}:{jailbreak}:test:harmless:{idx}",
                    "prompt_id": f"{model_name}:test_harmless:{idx}",
                    "split": "test",
                    "attack_family": jailbreak,
                    "label": 0,
                    "toxic_score": score_safety,
                    "jailbreak_score": score_jailbreak,
                    "prediction": int(prediction),
                    "thresholds": thresholds[jailbreak],
                    "layers": {
                        "toxic": seleced_safety_layer_index,
                        "jailbreak": seleced_jailbreak_layer_indexs[jailbreak],
                    },
                })
        tp = sum(labels_jb)
        fp = sum(labels_harmless)
        fn = len(labels_jb) - tp
        tn = len(labels_harmless) - fp
        totals["tp"] += tp
        totals["fp"] += fp
        totals["fn"] += fn
        totals["tn"] += tn
        family_metrics = _metrics(tp, fp, fn, tn)
        family_metrics["thresholds"] = thresholds[jailbreak]
        family_metrics["layers"] = {
            "toxic": seleced_safety_layer_index,
            "jailbreak": seleced_jailbreak_layer_indexs[jailbreak],
        }
        metrics_by_attack[jailbreak] = family_metrics
        accuracy = family_metrics["accuracy"]
        f1 = family_metrics["f1"]
        print("Accuracy: {}".format(accuracy), " | F1 score: {}".format(f1))

    if update_vectors:
        # Save vectors for mitigation when the first time to run this script.
        vector_dir = Path("vectors") / model_name
        vector_dir.mkdir(parents=True, exist_ok=True)
        layer_indexs = [seleced_safety_layer_index] + [
            seleced_jailbreak_layer_indexs[jailbreak]
            for jailbreak in LEGACY_VECTOR_ORDER
            if jailbreak in seleced_jailbreak_layer_indexs
        ]
        torch.save(layer_indexs, vector_dir / "layer_indexs.pt")
        torch.save(delta_safety, vector_dir / "delta_safety.pt")
        torch.save(calibration_harmless_embeddings[seleced_safety_layer_index], vector_dir / "calibration_harmless_embedding.pt")
        torch.save(calibration_safety_vector, vector_dir / "calibration_safety_vector.pt")
        for jailbreak in jailbreaks:
            layer_index = seleced_jailbreak_layer_indexs[jailbreak]
            torch.save(delta_jailbreaks[jailbreak], vector_dir / f"delta_jailbreak_{jailbreak}.pt")
            torch.save(calibration_harmful_embeddings[layer_index], vector_dir / f"calibration_harmful_embedding_{jailbreak}.pt")
            torch.save(calibration_jailbreak_vectors[jailbreak], vector_dir / f"calibration_jailbreak_vector_{jailbreak}.pt")

    overall = _metrics(totals["tp"], totals["fp"], totals["fn"], totals["tn"])
    overall["macro_f1"] = float(np.mean([m["f1"] for m in metrics_by_attack.values()])) if metrics_by_attack else 0.0
    metrics = {
        "model": model_name,
        "run_id": run_id,
        "overall": overall,
        "by_attack_family": metrics_by_attack,
    }
    summary = {
        "status": "success",
        "model": model_name,
        "model_path": effective_model_paths.get(model_name),
        "chat_template": chat_template,
        "jailbreak_model_name": jailbreak_model_name,
        "run_id": run_id,
        "elapsed_seconds": time.time() - start_time,
        "jailbreaks": jailbreaks,
        "critical_layers": {
            "toxic": seleced_safety_layer_index,
            "jailbreak": seleced_jailbreak_layer_indexs,
        },
        "thresholds": thresholds,
        "metrics": metrics,
        "hidden_state_audit": embedding_audit,
        "hidden_cache": {
            "enabled": cache_hidden,
            "model_id": cache_model_id if cache_hidden else None,
            "cache_dir": cache_dir if cache_hidden else None,
            "last_k": cache_last_k if cache_hidden else None,
            "extraction_version": EXTRACTION_VERSION if cache_hidden else None,
            "splits": cache_splits,
        },
        "phase2": {
            "enabled": phase2,
            "output_dir": phase2_output_dir if phase2 else None,
            "strict_spans": strict_spans if phase2 else None,
        },
    }
    if phase2:
        model_metadata = {
            "model_id": cache_model_id,
            "model_name": model_name,
            "model_path": effective_model_paths.get(model_name),
            "chat_template": chat_template,
            "tokenizer_class": type(tokenizer).__name__,
            "model_class": type(model).__name__,
            "num_layers": int(getattr(model.config, "num_hidden_layers", 0) + 1),
            "hidden_size": getattr(model.config, "hidden_size", None),
            "vocab_size": getattr(model.config, "vocab_size", None),
            "dtype": str(getattr(model, "dtype", "")),
            "device": str(getattr(model, "device", "")),
            "extraction_version": "jbshield-phase2-v1",
        }
        phase2_dir = write_phase2_outputs(
            phase2_output_dir,
            run_id,
            model_metadata,
            phase2_groups,
            sample_records,
            thresholds,
            summary["critical_layers"],
            {
                "toxic": calibration_safety_vector,
                "jailbreak": calibration_jailbreak_vectors,
            },
            {
                "toxic": delta_safety,
                "jailbreak": delta_jailbreaks,
            },
        )
        summary["phase2"]["path"] = str(phase2_dir)
    if audit_log:
        write_json(run_dir / "metrics.json", metrics)
        write_json(run_dir / "summary.json", summary)
        write_jsonl(run_dir / "samples.jsonl", sample_records)
    return metrics



if __name__ == '__main__':
    # Get parameters
    parser = argparse.ArgumentParser(description='JBShield-D')
    parser.add_argument('--model', type=str, default=None, help='Target model')
    parser.add_argument('--config', type=str, default=None, help='Runtime YAML config')
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--audit-log', action='store_true')
    parser.add_argument('--run-id', type=str, default=None)
    parser.add_argument('--update-vectors', action='store_true')
    parser.add_argument('--jailbreaks', type=str, default=None, help='Comma-separated jailbreak families')
    parser.add_argument('--cache-hidden', action='store_true', help='Persist per-prompt hidden states')
    parser.add_argument('--cache-dir', type=str, default=None, help='Root dir for hidden-state cache')
    parser.add_argument('--cache-last-k', type=int, default=None, help='Trailing tokens to cache')
    parser.add_argument('--phase2', action='store_true', help='Write Phase2 parquet/safetensors artifacts')
    parser.add_argument('--phase2-output-dir', type=str, default=None, help='Root dir for Phase2 artifacts')
    parser.add_argument('--strict-spans', action='store_true', help='Fail when Phase2 token span mapping is ambiguous or missing')

    args = parser.parse_args()
    runtime_config = {}
    config_path = None
    if args.config:
        runtime_config, config_path = load_runtime_config(args.config)

    model_config = runtime_config.get("model", {})
    data_config = runtime_config.get("data", {})
    runtime = runtime_config.get("runtime", {})
    audit = runtime_config.get("audit", {})
    detection_config = runtime_config.get("detection", {})
    cache_config = runtime_config.get("cache", {})
    phase2_config = runtime_config.get("phase2", {})

    model_name = args.model or model_config.get("name")
    if not model_name:
        parser.error("Set --model or model.name in --config")

    jailbreaks = _parse_jailbreaks(args.jailbreaks)
    if jailbreaks is None:
        jailbreaks = _parse_jailbreaks(detection_config.get("jailbreaks")) or DEFAULT_JAILBREAKS
    output_dir = args.output_dir or runtime.get("output_dir", "result")
    audit_log = bool(args.audit_log or audit.get("enabled", False))
    run_id = args.run_id or audit.get("run_id") or f"{model_name}-phase1"
    update_vectors = bool(args.update_vectors or detection_config.get("update_vectors", False))
    cache_hidden = bool(args.cache_hidden or cache_config.get("hidden_states", False))
    cache_dir = args.cache_dir or cache_config.get("dir", "representations")
    cache_last_k = args.cache_last_k if args.cache_last_k is not None else int(cache_config.get("last_k", 5))
    phase2 = bool(args.phase2 or phase2_config.get("enabled", False))
    phase2_output_dir = args.phase2_output_dir or phase2_config.get("output_dir", "outputs/phase2")
    strict_spans = bool(args.strict_spans or phase2_config.get("strict_spans", False))
    if cache_last_k < 1:
        parser.error("--cache-last-k / cache.last_k must be >= 1")

    # Run this script to evaluate the detection performance of JBShield-D
    try:
        detection(
            model_name,
            update_vectors=update_vectors,
            jailbreaks=jailbreaks,
            output_dir=output_dir,
            audit_log=audit_log,
            run_id=run_id,
            runtime_config=runtime_config,
            config_path=config_path,
            model_path=model_config.get("path"),
            data_config=data_config,
            chat_template=model_config.get("chat_template"),
            cache_hidden=cache_hidden,
            cache_dir=cache_dir,
            cache_last_k=cache_last_k,
            phase2=phase2,
            phase2_output_dir=phase2_output_dir,
            strict_spans=strict_spans,
        )
    except Exception as exc:
        if audit_log:
            failure_dir = _run_dir(output_dir, model_name, run_id)
            write_json(failure_dir / "summary.json", {
                "status": "failed",
                "model": model_name,
                "run_id": run_id,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "config_path": str(config_path) if config_path else None,
            })
        raise

# An example for run this script to evaluate JBShield-D on the Mistral model
# python detection.py --model mistral
