#!/usr/bin/env bash
#
# Run intra-node analysis for all benchmarks using all models ranked together,
# with two parameter sets.
#
set -euo pipefail
cd "$(dirname "$0")"

PARAM_SETS=(
    "--B 1000 --min_instances 5 --min_tau_reliable 0.8 --max_ci_width_unreliable 0.4"
    "--B 1000 --min_instances 2 --min_tau_reliable 0.3 --max_ci_width_unreliable 0.5"
)

total=$(( 3 * ${#PARAM_SETS[@]} ))
done_count=0

echo "=== Running $total jobs (all models ranked together) ==="
echo ""

run_benchmark() {
    local bench="$1"
    local split="$2"
    shift 2
    local models=("$@")

    for params in "${PARAM_SETS[@]}"; do
        done_count=$(( done_count + 1 ))
        echo "[$done_count/$total] $bench: ${models[*]}  ($params)"
        python intra.py \
            --benchmark "$bench" \
            --split "$split" \
            --models "${models[@]}" \
            $params
        echo ""
    done
}

run_benchmark "MATH" "full" \
    "Llama-3.1-8B-Instruct" "dart-math-llama3-8b-uniform" "gpt-4o-mini-2024-07-18"

run_benchmark "MMLU" "10042-4000" \
    "Llama-3.1-8B-Instruct" "Llama-3.1-Tulu-3-8B" "gpt-4o-mini-2024-07-18"

run_benchmark "DS-1000" "600-400" \
    "deepseek-coder-6.7b-base" "gpt-3.5-turbo-0613" "gpt-4o-2024-08-06"

echo "=== All $total runs complete ==="
