"""
Configuration and path helpers for EvalTree intra-node ranking stability experiment.
"""
from __future__ import annotations

import os

# Default EvalTree dataset root relative to repo root (DILL)
DEFAULT_EVALTREE_ROOT = os.path.join(os.path.dirname(__file__), "EvalTree", "Datasets")

# Tree path components (match EvalTree README)
DEFAULT_SPLIT = "full"
DEFAULT_ANNOTATION = "gpt-4o-mini"
DEFAULT_EMBEDDING = "text-embedding-3-small"
DEFAULT_MAX_CHILDREN = 10
DEFAULT_STAGE4_MODEL = "gpt-4o-mini"

TREE_FOLDER = "stage3-RecursiveClustering"
TREE_SUFFIX = f"[split={{split}}]_[annotation={{annotation}}]_[embedding={{embedding}}]_[max-children={{max_children}}]"
TREE_SUFFIX_STAGE4 = TREE_SUFFIX + "_[stage4-CapabilityDescription-model=" + DEFAULT_STAGE4_MODEL + "]"

TREE_DIR_NAME = f"TREE=[stage3-RecursiveClustering]_{TREE_SUFFIX}"


def tree_bin_path(
    benchmark: str,
    split: str = DEFAULT_SPLIT,
    evaltree_root: str | None = None,
) -> str:
    root = evaltree_root or DEFAULT_EVALTREE_ROOT
    name = TREE_SUFFIX.format(
        split=split,
        annotation=DEFAULT_ANNOTATION,
        embedding=DEFAULT_EMBEDDING,
        max_children=DEFAULT_MAX_CHILDREN,
    )
    return os.path.join(root, benchmark, "EvalTree", TREE_FOLDER, f"{name}.bin")


def tree_stage4_json_path(
    benchmark: str,
    split: str = DEFAULT_SPLIT,
    evaltree_root: str | None = None,
) -> str:
    root = evaltree_root or DEFAULT_EVALTREE_ROOT
    name = TREE_SUFFIX_STAGE4.format(
        split=split,
        annotation=DEFAULT_ANNOTATION,
        embedding=DEFAULT_EMBEDDING,
        max_children=DEFAULT_MAX_CHILDREN,
    )
    return os.path.join(root, benchmark, "EvalTree", TREE_FOLDER, f"{name}.json")


def confidence_interval_path(
    benchmark: str,
    model: str,
    split: str = DEFAULT_SPLIT,
    evaltree_root: str | None = None,
) -> str:
    root = evaltree_root or DEFAULT_EVALTREE_ROOT
    tree_part = TREE_DIR_NAME.format(
        split=split,
        annotation=DEFAULT_ANNOTATION,
        embedding=DEFAULT_EMBEDDING,
        max_children=DEFAULT_MAX_CHILDREN,
    )
    return os.path.join(
        root,
        benchmark,
        "eval_results",
        "real",
        model,
        "EvalTree",
        tree_part,
        "confidence_interval.json",
    )


def results_json_path(
    benchmark: str,
    model: str,
    evaltree_root: str | None = None,
) -> str:
    root = evaltree_root or DEFAULT_EVALTREE_ROOT
    return os.path.join(root, benchmark, "eval_results", "real", model, "results.json")

