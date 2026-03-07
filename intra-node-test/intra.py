"""
EvalTree intra-node ranking stability experiment (bootstrap Kendall's Tau).

For each capability node Ck with N instances and M models:
- Build reference ranking from mean accuracy across instances.
- Bootstrap B times: resample instances, compute ranking, Kendall's Tau vs reference.
- Compute mean τ and 95% CI; mark nodes as unreliable if τ is low or CI is wide.
- Plot histogram of mean Kendall's Tau across nodes.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any

import numpy as np
from scipy import stats

from intra_config import (
    DEFAULT_EVALTREE_ROOT,
    DEFAULT_SPLIT,
    confidence_interval_path,
    results_json_path,
)


def _collect_all_nodes(
    ci_node: dict,
    path: list,
    out: list[tuple[list, list[int], bool, bool]],
) -> list[int]:
    """Traverse confidence_interval tree; append (path, instance_list, is_root, is_list_subtree) for every internal node (subtrees is dict or list). Returns instance indices under this node."""
    st = ci_node.get("subtrees")
    if st is None:
        return []
    if isinstance(st, int):
        return [st]
    if isinstance(st, list):
        instances = []
        for sub in st:
            instances.extend(_collect_all_nodes(sub, path, out))
        out.append((path, instances, False, True))  # leaf capability node
        return instances
    if isinstance(st, dict):
        all_instances = []
        for k, sub in st.items():
            all_instances.extend(_collect_all_nodes(sub, path + [str(k)], out))
        is_root = len(path) == 0
        out.append((path, all_instances, is_root, False))  # internal node (dict)
        return all_instances
    return []


def collect_eligible_nodes_with_stats(
    confidence_interval: dict,
    min_instances: int,
) -> tuple[list[tuple[list, list[int]]], dict[str, int]]:
    """
    Return (eligible_nodes, filter_stats).
    eligible_nodes: [(path, instance_list)] for nodes used in analysis (non-root, dict subtrees only, >= min_instances).
    filter_stats: counts for printing (total_nodes, filtered_root, filtered_leaf_capability, filtered_below_min_instances, eligible_count).
    """
    all_nodes: list[tuple[list, list[int], bool, bool]] = []
    _collect_all_nodes(confidence_interval, [], all_nodes)

    total_nodes = len(all_nodes)
    filtered_root = sum(1 for (_, _, is_root, _) in all_nodes if is_root)
    filtered_leaf_capability = sum(1 for (_, _, _, is_list) in all_nodes if is_list)
    # Internal non-root = dict subtrees, not root
    internal_non_root = [
        (path, inst) for (path, inst, is_root, is_list) in all_nodes
        if not is_root and not is_list
    ]
    filtered_below_min_instances = sum(1 for (_, inst) in internal_non_root if len(inst) < min_instances)
    eligible = [(p, inst) for p, inst in internal_non_root if len(inst) >= min_instances]

    filter_stats = {
        "total_nodes": total_nodes,
        "filtered_root": filtered_root,
        "filtered_leaf_capability": filtered_leaf_capability,
        "filtered_below_min_instances": filtered_below_min_instances,
        "eligible_count": len(eligible),
    }
    return eligible, filter_stats


def accuracy_to_ranking(accs: np.ndarray) -> np.ndarray:
    """Convert vector of accuracies to 1-based ranks (higher accuracy = better rank). Ties broken by original index."""
    order = np.argsort(-accs, kind="stable")  # descending
    ranks = np.empty_like(accs, dtype=float)
    ranks[order] = np.arange(1, len(accs) + 1, dtype=float)
    return ranks


def kendall_tau_from_accuracies(ref_accs: np.ndarray, boot_accs: np.ndarray) -> float:
    """Kendall's Tau between rankings induced by ref_accs and boot_accs (same order of models)."""
    ref_ranks = accuracy_to_ranking(ref_accs)
    boot_ranks = accuracy_to_ranking(boot_accs)
    tau, _ = stats.kendalltau(ref_ranks, boot_ranks)
    return float(tau) if not np.isnan(tau) else 0.0


def bootstrap_kendall_taus(
    A: np.ndarray,
    B: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float, float, float]:
    """
    A: (N, M) per-instance correctness (0/1).
    Returns (taus array of length B, mean_tau, ci_low, ci_high) using 95% CI.
    """
    N, M = A.shape
    ref_means = A.mean(axis=0)
    taus = np.zeros(B)
    for b in range(B):
        idx = rng.integers(0, N, size=N)
        Ab = A[idx, :]
        boot_means = Ab.mean(axis=0)
        taus[b] = kendall_tau_from_accuracies(ref_means, boot_means)
    mean_tau = float(np.mean(taus))
    ci_low = float(np.percentile(taus, 2.5))
    ci_high = float(np.percentile(taus, 97.5))
    return taus, mean_tau, ci_low, ci_high


def load_results_list(path: str) -> list[int]:
    """Load results.json as list of 0/1 (index = instance id)."""
    with open(path, "r") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    # if dict keyed by str indices
    n = max(int(k) for k in data) + 1
    out = [0] * n
    for k, v in data.items():
        out[int(k)] = v
    return out


def run_intra_node_analysis(
    benchmark: str,
    models: list[str],
    split: str = DEFAULT_SPLIT,
    B: int = 1000,
    min_instances: int = 5,
    min_tau_reliable: float = 0.8,
    max_ci_width_unreliable: float = 0.2,
    reliability_criterion: str = "tau",
    evaltree_root: str | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Run full intra-node analysis. Returns dict with:
    - node_results: list of {path, n_instances, mean_tau, ci_low, ci_high, reliable}
    - mean_taus: array of mean tau per node (for histogram)
    - reliable_mask: which nodes are reliable
    - fraction_reliable

    reliability_criterion: "tau" = reliable when mean_tau >= min_tau_reliable and tau CI width <= max_ci_width_unreliable;
    "ci_overlap" = reliable when the two models' binomial (accuracy) 95%% CIs do NOT overlap (only for 2 models).
    """
    root = evaltree_root or DEFAULT_EVALTREE_ROOT
    rng = np.random.default_rng(seed)

    # Load confidence_interval from first model (same tree structure)
    ci_path = confidence_interval_path(benchmark, models[0], split=split, evaltree_root=root)
    if not os.path.isfile(ci_path):
        raise FileNotFoundError(f"Confidence interval not found: {ci_path}")
    with open(ci_path, "r") as f:
        confidence_interval = json.load(f)

    # For CI-overlap criterion, load each model's CI tree (same structure, different per-node CIs)
    ci_trees_for_overlap: list[dict] = []
    if reliability_criterion == "ci_overlap":
        if len(models) != 2:
            raise ValueError("reliability_criterion=ci_overlap requires exactly 2 models")
        for m in models:
            p = confidence_interval_path(benchmark, m, split=split, evaltree_root=root)
            if not os.path.isfile(p):
                raise FileNotFoundError(f"Confidence interval not found for {m}: {p}")
            with open(p, "r") as f:
                ci_trees_for_overlap.append(json.load(f))

    nodes, filter_stats = collect_eligible_nodes_with_stats(confidence_interval, min_instances)
    if not nodes:
        return {
            "node_results": [],
            "mean_taus": np.array([]),
            "reliable_mask": np.array([], dtype=bool),
            "fraction_reliable": 0.0,
            "models": models,
            "n_nodes": 0,
            "filter_stats": filter_stats,
            "reliability_criterion": reliability_criterion,
        }

    # Load per-instance results for each model
    model_results: list[list[int]] = []
    for m in models:
        rpath = results_json_path(benchmark, m, evaltree_root=root)
        if not os.path.isfile(rpath):
            raise FileNotFoundError(f"Results not found: {rpath}")
        model_results.append(load_results_list(rpath))

    M = len(models)
    node_results = []
    mean_taus_list = []
    reliable_list = []

    for path, instances in nodes:
        N = len(instances)
        A = np.zeros((N, M), dtype=np.float64)
        for j, res in enumerate(model_results):
            for i, idx in enumerate(instances):
                if idx < len(res):
                    A[i, j] = res[idx]
                else:
                    A[i, j] = 0.0

        taus, mean_tau, ci_low, ci_high = bootstrap_kendall_taus(A, B, rng)
        ci_width = ci_high - ci_low
        if reliability_criterion == "ci_overlap":
            reliable = _binomial_ci_overlap_reliable(ci_trees_for_overlap, path)
        else:
            reliable = mean_tau >= min_tau_reliable and ci_width <= max_ci_width_unreliable

        node_results.append({
            "path": path,
            "n_instances": N,
            "mean_tau": mean_tau,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "ci_width": ci_width,
            "reliable": reliable,
        })
        mean_taus_list.append(mean_tau)
        reliable_list.append(reliable)

    mean_taus = np.array(mean_taus_list)
    reliable_mask = np.array(reliable_list)
    n_reliable = reliable_mask.sum()
    fraction_reliable = n_reliable / len(reliable_mask) if reliable_mask.size else 0.0

    return {
        "node_results": node_results,
        "mean_taus": mean_taus,
        "reliable_mask": reliable_mask,
        "fraction_reliable": fraction_reliable,
        "models": models,
        "n_nodes": len(nodes),
        "n_reliable": int(n_reliable),
        "filter_stats": filter_stats,
        "reliability_criterion": reliability_criterion,
    }


def _shorten_model_name(name: str) -> str:
    """Produce a compact label for a model name."""
    replacements = [
        ("Llama-3.1-8B-Instruct", "Llama3.1-8B"),
        ("Llama-3.1-Tulu-3-8B", "Tulu3-8B"),
        ("dart-math-llama3-8b-uniform", "DartMath-8B"),
        ("gpt-4o-mini-2024-07-18", "GPT4o-mini"),
        ("gpt-4o-2024-08-06", "GPT4o"),
        ("gpt-3.5-turbo-0613", "GPT3.5"),
        ("deepseek-coder-6.7b-base", "DSCoder-6.7B"),
    ]
    for full, short in replacements:
        if name == full:
            return short
    return name


def plot_histogram(
    mean_taus: np.ndarray,
    out_path: str | None = None,
    benchmark: str = "",
    models: list[str] | None = None,
) -> None:
    """Plot histogram of mean Kendall's Tau across nodes."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.hist(mean_taus, bins=min(30, max(1, len(mean_taus))), edgecolor="black", alpha=0.7)
    ax.set_xlabel("Mean Kendall's τ (ranking stability)")
    ax.set_ylabel("Number of nodes")
    title = "Intra-node ranking stability (bootstrap)"
    if benchmark:
        title += f" — {benchmark}"
    if models:
        short = [_shorten_model_name(m) for m in models]
        title += f"\n{short[0]} vs {short[1]}" if len(short) == 2 else f"\n{', '.join(short)}"
    ax.set_title(title)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="EvalTree intra-node ranking stability (bootstrap Kendall's Tau)")
    parser.add_argument("--benchmark", type=str, default="MATH")
    parser.add_argument("--models", type=str, nargs="+", default=["Llama-3.1-8B-Instruct", "gpt-4o-mini-2024-07-18"])
    parser.add_argument("--split", type=str, default=DEFAULT_SPLIT)
    parser.add_argument("--B", type=int, default=1000, help="Bootstrap iterations")
    parser.add_argument("--min_instances", type=int, default=5, help="Min instances per node to include")
    parser.add_argument("--min_nodes", type=int, default=None, help="Min instances/entries per node (overrides --min_instances if set)")
    parser.add_argument("--min_tau_reliable", type=float, default=0.8, help="Min mean τ to consider node reliable (when --reliability_criterion tau)")
    parser.add_argument("--max_ci_width_unreliable", type=float, default=0.4, help="Max 95%% CI width for τ to consider node reliable (when --reliability_criterion tau)")
    parser.add_argument("--reliability_criterion", type=str, choices=("tau", "ci_overlap"), default="tau",
        help="tau: reliable by bootstrap τ and its CI width; ci_overlap: reliable when the two models' binomial accuracy 95%% CIs do not overlap (2 models only)")
    parser.add_argument("--evaltree_root", type=str, default=None, help="EvalTree Datasets root (default: EvalTree/Datasets)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_json", type=str, default=None, help="Save results to this JSON path")
    parser.add_argument("--output_histogram", type=str, default=None, help="Save histogram to this path")
    parser.add_argument("--no_plot", action="store_true", help="Do not show histogram (only save if --output_histogram)")
    args = parser.parse_args()

    evaltree_root = args.evaltree_root or DEFAULT_EVALTREE_ROOT
    min_instances = args.min_nodes if args.min_nodes is not None else args.min_instances

    result = run_intra_node_analysis(
        benchmark=args.benchmark,
        models=args.models,
        split=args.split,
        B=args.B,
        min_instances=min_instances,
        min_tau_reliable=args.min_tau_reliable,
        max_ci_width_unreliable=args.max_ci_width_unreliable,
        reliability_criterion=args.reliability_criterion,
        evaltree_root=evaltree_root,
        seed=args.seed,
    )

    # Convert numpy for JSON
    out_serializable = {
        "node_results": result["node_results"],
        "mean_taus": result["mean_taus"].tolist(),
        "reliable_mask": result["reliable_mask"].tolist(),
        "fraction_reliable": result["fraction_reliable"],
        "models": result["models"],
        "n_nodes": result["n_nodes"],
        "n_reliable": result["n_reliable"],
        "filter_stats": result.get("filter_stats"),
        "reliability_criterion": result.get("reliability_criterion"),
    }

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(out_serializable, f, indent=2)

    fs = result.get("filter_stats") or {}
    print("Node counts (before and after filtering):")
    print(f"  Total nodes in tree (before any filtering): {fs.get('total_nodes', 'N/A')}")
    print(f"  Filtered out (root node):                  {fs.get('filtered_root', 'N/A')}")
    print(f"  Filtered out (leaf capability nodes):      {fs.get('filtered_leaf_capability', 'N/A')}")
    print(f"  Filtered out (fewer than min_instances):   {fs.get('filtered_below_min_instances', 'N/A')}")
    print(f"  Eligible nodes (used in analysis):         {fs.get('eligible_count', result['n_nodes'])}")
    print()
    print(f"Reliability criterion: {result.get('reliability_criterion', 'tau')}")
    if result.get("reliability_criterion") == "tau":
        print(f"  (thresholds: min_tau_reliable={args.min_tau_reliable}, max_ci_width_unreliable={args.max_ci_width_unreliable})")
    else:
        print("  (reliable = two models' binomial 95% accuracy CIs do not overlap)")
    total_in_tree = fs.get("total_nodes", result.get("n_nodes"))
    print(f"Nodes: {result['n_nodes']} (total in tree: {total_in_tree}), Reliable: {result['n_reliable']}, Fraction reliable: {result['fraction_reliable']:.2%}")
    if result["n_nodes"]:
        print(f"Mean τ: min={result['mean_taus'].min():.3f}, max={result['mean_taus'].max():.3f}, mean={result['mean_taus'].mean():.3f}")
        if result["fraction_reliable"] < 0.5:
            print("Most nodes are unreliable after intra-node analysis; consider skipping inter-node analysis.")

    if result["mean_taus"].size:
        if not args.output_histogram and not args.no_plot:
            os.makedirs("tau_histograms", exist_ok=True)
            short_models = "_vs_".join(_shorten_model_name(m) for m in args.models)
            args.output_histogram = os.path.join(
                "tau_histograms",
                f"{args.benchmark}"
                f"_{short_models}"
                f"_B{args.B}"
                f"_min{min_instances}"
                f"_tau{args.min_tau_reliable}"
                f"_ci{args.max_ci_width_unreliable}"
                f".png",
            )
        if args.output_histogram:
            plot_histogram(result["mean_taus"], out_path=args.output_histogram,
                           benchmark=args.benchmark, models=args.models)
            print(f"Histogram saved to {args.output_histogram}")
        elif not args.no_plot:
            plot_histogram(result["mean_taus"], out_path=None,
                           benchmark=args.benchmark, models=args.models)


if __name__ == "__main__":
    main()
