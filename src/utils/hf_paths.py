from pathlib import Path


def resolve_hf_model_path(path_or_name: str) -> str:
    """
    Resolve a Hugging Face cache directory to an actual snapshot directory.

    Accepts either:
    - model id like "BAAI/bge-small-en"
    - full local path
    - cache root path like ".../models--org--name"
    """
    candidate = Path(path_or_name)
    if not candidate.exists():
        return path_or_name

    refs_main = candidate / "refs" / "main"
    snapshots = candidate / "snapshots"

    if refs_main.exists() and snapshots.exists():
        commit = refs_main.read_text(encoding="utf-8").strip()
        snapshot_dir = snapshots / commit
        if snapshot_dir.exists():
            return str(snapshot_dir)

    if (candidate / "config.json").exists():
        return str(candidate)

    return path_or_name


def has_hf_weights(path_or_name: str) -> bool:
    candidate = Path(path_or_name)
    if not candidate.exists():
        return True

    weight_files = (
        "model.safetensors",
        "pytorch_model.bin",
        "tf_model.h5",
        "flax_model.msgpack",
    )
    return any((candidate / weight_file).exists() for weight_file in weight_files)
