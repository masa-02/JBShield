from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def resolve_config_path(config):
    path = Path(config)
    candidates = [path] if path.is_absolute() else [Path.cwd() / path, PROJECT_ROOT / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Config not found: {config}")


def load_runtime_config(config):
    import yaml

    config_path = resolve_config_path(config)
    with config_path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Runtime config must be a mapping: {config_path}")
    return payload, config_path
