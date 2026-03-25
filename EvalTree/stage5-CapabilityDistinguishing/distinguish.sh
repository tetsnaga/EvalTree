# python -m EvalTree.stage5-CapabilityDistinguishing.distinguish \
#     --dataset MATH
# python -m EvalTree.stage5-CapabilityDistinguishing.distinguish \
#     --dataset WildChat10K
# python -m EvalTree.stage5-CapabilityDistinguishing.distinguish \
#     --dataset Chatbot-Arena
# python -m EvalTree.stage5-CapabilityDistinguishing.distinguish \
#     --dataset MMLU
# python -m EvalTree.stage5-CapabilityDistinguishing.distinguish \
#     --dataset DS-1000

python -m EvalTree.stage5-CapabilityDistinguishing.distinguish \
    --dataset MATH-Hard \
    --num_procs 64
python -m EvalTree.stage5-CapabilityDistinguishing.distinguish \
    --dataset MMLU-Pro \
    --num_procs 64
python -m EvalTree.stage5-CapabilityDistinguishing.distinguish \
    --dataset BBH \
    --num_procs 64
python -m EvalTree.stage5-CapabilityDistinguishing.distinguish \
    --dataset GPQA \
    --num_procs 64
