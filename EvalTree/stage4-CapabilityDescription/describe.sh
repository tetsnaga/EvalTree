# python -m EvalTree.stage4-CapabilityDescription.describe \
#     --dataset MATH \
#     --tree_path stage3-RecursiveClustering/[split=full]_[annotation=gpt-4o-mini]_[embedding=text-embedding-3-small]_[max-children=10]
# python -m EvalTree.stage4-CapabilityDescription.describe \
#     --dataset MATH \
#     --tree_path stage3-RecursiveClustering/[split=4k-1k]_[annotation=gpt-4o-mini]_[embedding=text-embedding-3-small]_[max-children=10]

# python -m EvalTree.stage4-CapabilityDescription.describe \
#     --dataset WildChat10K \
#     --tree_path stage3-RecursiveClustering/[split=full]_[annotation=gpt-4o-mini]_[embedding=text-embedding-3-small]_[max-children=10]
# python -m EvalTree.stage4-CapabilityDescription.describe \
#     --dataset WildChat10K \
#     --tree_path stage3-RecursiveClustering/[split=8k-2k]_[annotation=gpt-4o-mini]_[embedding=text-embedding-3-small]_[max-children=10]

# python -m EvalTree.stage4-CapabilityDescription.describe \
#     --dataset DS-1000 \
#     --tree_path stage3-RecursiveClustering/[split=600-400]_[annotation=gpt-4o-mini]_[embedding=text-embedding-3-small]_[max-children=10]

python -m EvalTree.stage4-CapabilityDescription.describe \
    --dataset MATH-Hard \
    --tree_path stage3-RecursiveClustering/[split=full]_[annotation=gpt-4o-mini]_[embedding=text-embedding-3-small]_[max-children=10]
python -m EvalTree.stage4-CapabilityDescription.describe \
    --dataset MMLU-Pro \
    --tree_path stage3-RecursiveClustering/[split=full]_[annotation=gpt-4o-mini]_[embedding=text-embedding-3-small]_[max-children=10]
python -m EvalTree.stage4-CapabilityDescription.describe \
    --dataset BBH \
    --tree_path stage3-RecursiveClustering/[split=full]_[annotation=gpt-4o-mini]_[embedding=text-embedding-3-small]_[max-children=10]
python -m EvalTree.stage4-CapabilityDescription.describe \
    --dataset GPQA \
    --tree_path stage3-RecursiveClustering/[split=full]_[annotation=gpt-4o-mini]_[embedding=text-embedding-3-small]_[max-children=10]
