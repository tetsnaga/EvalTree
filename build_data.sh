for dataset in MATH-Hard MMLU-Pro BBH GPQA; do
    python -m build_data --dataset ${dataset}
done
