# python -m EvalTree.stage2-CapabilityEmbedding.embedding --dataset MATH
# python -m EvalTree.stage2-CapabilityEmbedding.embedding --dataset WildChat10K
# python -m EvalTree.stage2-CapabilityEmbedding.embedding --dataset DS-1000

# python -m EvalTree.stage2-CapabilityEmbedding.embedding --dataset Chatbot-Arena
# python -m EvalTree.stage2-CapabilityEmbedding.embedding --dataset ShareGPT10K
# python -m EvalTree.stage2-CapabilityEmbedding.embedding --dataset MMLU
# python -m EvalTree.stage2-CapabilityEmbedding.embedding --dataset CollegeMath

python -m EvalTree.stage2-CapabilityEmbedding.embedding --dataset MATH-Hard --num_procs 64
python -m EvalTree.stage2-CapabilityEmbedding.embedding --dataset MMLU-Pro --num_procs 64
python -m EvalTree.stage2-CapabilityEmbedding.embedding --dataset BBH --num_procs 64
python -m EvalTree.stage2-CapabilityEmbedding.embedding --dataset GPQA --num_procs 64
