import os

from datasets import load_dataset


def repo_root() -> str:
    # This file is in EvalTree/intra-node-test/src/
    return os.path.dirname(os.path.dirname(os.path.dirname(__file__)))


def datasets_root() -> str:
    root = repo_root()
    path = os.path.join(root, "Datasets")
    os.makedirs(path, exist_ok=True)
    return path


def download_mmlu_pro() -> None:
    root = datasets_root()
    print("Downloading MMLU-Pro dataset...")
    ds = load_dataset("TIGER-Lab/MMLU-Pro")
    out_dir = os.path.join(root, "MMLU-Pro")
    ds.save_to_disk(out_dir)
    print(f"Saved MMLU-Pro to {out_dir}")


def download_gpqa() -> None:
    root = datasets_root()
    print("Downloading GPQA dataset...")
    ds = load_dataset("Idavidrein/gpqa", "gpqa_main")
    out_dir = os.path.join(root, "GPQA")
    ds.save_to_disk(out_dir)
    print(f"Saved GPQA to {out_dir}")


def download_math_hard() -> None:
    root = datasets_root()
    print("Downloading MATH Hard dataset...")
    ds = load_dataset("lighteval/MATH-Hard")
    out_dir = os.path.join(root, "MATH-Hard")
    ds.save_to_disk(out_dir)
    print(f"Saved MATH Hard to {out_dir}")


if __name__ == "__main__":
    download_mmlu_pro()
    download_gpqa()
    download_math_hard()