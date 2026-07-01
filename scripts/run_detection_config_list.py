import argparse
import os
import subprocess
import sys
from pathlib import Path


def read_manifest(path):
    configs = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        item = line.split("#", 1)[0].strip()
        if item:
            configs.append(item)
    return configs


def should_cleanup(policy, reason):
    normalized = (policy or "never").lower()
    if normalized in {"always", "true", "1", "yes"}:
        return True
    if normalized == "on_success":
        return reason == "success"
    if normalized in {"on_failure", "interrupted"}:
        return reason in {"failure", "interrupted"}
    if normalized in {"never", "false", "0", "no", ""}:
        return False
    return False


def cleanup_hf_cache(config, reason, log):
    policy = os.environ.get("JBSHIELD_CLEAN_HF_CACHE", "always")
    if not should_cleanup(policy, reason):
        return

    cmd = [
        sys.executable,
        "-u",
        "scripts/cleanup_hf_cache.py",
        "--config",
        config,
    ]
    if os.environ.get("JBSHIELD_CLEAN_HF_CACHE_DRY_RUN") == "1":
        cmd.append("--dry-run")

    log.write(f"===== HF cache cleanup ({reason}): {config} =====\n")
    log.flush()
    completed = subprocess.run(cmd, stdout=log, stderr=log, text=True)
    if completed.returncode != 0:
        log.write(f"Warning: HF cache cleanup failed for {config}\n")
        log.flush()


def main():
    parser = argparse.ArgumentParser(description="Run JBShield-D configs from a manifest")
    parser.add_argument("manifest", nargs="?", default="configs/runtime/manifests/official.txt")
    parser.add_argument("run_prefix", nargs="?", default="phase1-official")
    parser.add_argument("log_file", nargs="?", default="logs/JBShield-D_runtime.log")
    parser.add_argument("jailbreaks", nargs="?", default="")
    parser.add_argument("--phase2", action="store_true", help="Write Phase2 artifacts")
    parser.add_argument("--phase2-output-dir", default=None)
    parser.add_argument("--strict-spans", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("JBSHIELD_CLEAN_HF_CACHE", "always")
    log_path = Path(args.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    configs = read_manifest(args.manifest)

    current_config = None
    with log_path.open("w", encoding="utf-8") as log:
        try:
            for config in configs:
                current_config = config
                name = Path(config).stem
                run_id = f"{args.run_prefix}-{name}"
                log.write(f"===== {config} =====\n")
                log.flush()

                cmd = [
                    sys.executable,
                    "-u",
                    "detection.py",
                    "--config",
                    config,
                    "--audit-log",
                    "--run-id",
                    run_id,
                ]
                if args.jailbreaks:
                    cmd.extend(["--jailbreaks", args.jailbreaks])
                if args.phase2:
                    cmd.append("--phase2")
                if args.phase2_output_dir:
                    cmd.extend(["--phase2-output-dir", args.phase2_output_dir])
                if args.strict_spans:
                    cmd.append("--strict-spans")

                completed = subprocess.run(
                    cmd,
                    stdout=log,
                    stderr=log,
                    text=True,
                )
                if completed.returncode != 0:
                    log.write(
                        f"Skip {config}: detection failed. Check GPU memory, model files, "
                        "gated model access, HF download/cache state, span mapping, or model-specific runtime support.\n"
                    )
                    cleanup_hf_cache(config, "failure", log)
                    current_config = None
                    log.flush()
                    continue

                cleanup_hf_cache(config, "success", log)
                current_config = None
        finally:
            if current_config:
                cleanup_hf_cache(current_config, "interrupted", log)


if __name__ == "__main__":
    main()
