#!/usr/bin/env bash
#
# Run intra-node analysis for all benchmark/model-pair combinations
# with two parameter sets.
#
set -euo pipefail
cd "$(dirname "$0")"

BENCHMARKS_AND_SPLITS=(
    "MATH:full"
    "MMLU:10042-4000"
    "DS-1000:600-400"
)

MODELS_MATH=("Llama-3.1-8B-Instruct" "dart-math-llama3-8b-uniform" "gpt-4o-mini-2024-07-18")
MODELS_MMLU=("Llama-3.1-8B-Instruct" "Llama-3.1-Tulu-3-8B" "gpt-4o-mini-2024-07-18")
MODELS_DS1000=("deepseek-coder-6.7b-base" "gpt-3.5-turbo-0613" "gpt-4o-2024-08-06")

PARAM_SETS=(
    "--B 1000 --min_instances 5 --min_tau_reliable 0.8 --max_ci_width_unreliable 0.4"
    "--B 1000 --min_instances 2 --min_tau_reliable 0.3 --max_ci_width_unreliable 0.5"
)

get_models() {
    local bench="$1"
    case "$bench" in
        MATH)    echo "${MODELS_MATH[@]}" ;;
        MMLU)    echo "${MODELS_MMLU[@]}" ;;
        DS-1000) echo "${MODELS_DS1000[@]}" ;;
    esac
}

total=0
done_count=0

for entry in "${BENCHMARKS_AND_SPLITS[@]}"; do
    bench="${entry%%:*}"
    split="${entry##*:}"
    read -ra models <<< "$(get_models "$bench")"
    n=${#models[@]}
    for (( i=0; i<n; i++ )); do
        for (( j=i+1; j<n; j++ )); do
            total=$(( total + ${#PARAM_SETS[@]} ))
        done
    done
done

echo "=== Running $total jobs ==="
echo ""

for entry in "${BENCHMARKS_AND_SPLITS[@]}"; do
    bench="${entry%%:*}"
    split="${entry##*:}"
    read -ra models <<< "$(get_models "$bench")"
    n=${#models[@]}

    for (( i=0; i<n; i++ )); do
        for (( j=i+1; j<n; j++ )); do
            for params in "${PARAM_SETS[@]}"; do
                done_count=$(( done_count + 1 ))
                echo "[$done_count/$total] $bench: ${models[$i]} vs ${models[$j]}  ($params)"
                python intra.py \
                    --benchmark "$bench" \
                    --split "$split" \
                    --models "${models[$i]}" "${models[$j]}" \
                    $params
                echo ""
            done
        done
    done
done

echo "=== All $total runs complete ==="
