import argparse
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


def main():
    parser = argparse.ArgumentParser(description="Run JBShield-D configs from a manifest")
    parser.add_argument("manifest", nargs="?", default="configs/runtime/manifests/official.txt")
    parser.add_argument("run_prefix", nargs="?", default="gate1-official")
    parser.add_argument("log_file", nargs="?", default="logs/JBShield-D_runtime.log")
    parser.add_argument("jailbreaks", nargs="?", default="")
    args = parser.parse_args()

    log_path = Path(args.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    configs = read_manifest(args.manifest)

    with log_path.open("w", encoding="utf-8") as log:
        for config in configs:
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

            completed = subprocess.run(
                cmd,
                stdout=log,
                stderr=log,
                text=True,
            )
            if completed.returncode != 0:
                log.write(
                    f"Skip {config}: detection failed. Check GPU memory, model files, "
                    "gated model access, or model-specific runtime support.\n"
                )
                log.flush()
                continue


if __name__ == "__main__":
    main()
