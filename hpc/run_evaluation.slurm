#!/bin/bash
#SBATCH --job-name=mem-eval
#SBATCH --partition=gpu
#SBATCH --nodelist=node2
#SBATCH --error=/home/sudhakar/Vedang_U23AI111_8817182016/memory_agent/logs/eval_%j.err
#SBATCH --output=/home/sudhakar/Vedang_U23AI111_8817182016/memory_agent/logs/eval_%j.out
#SBATCH --gres=shard:10
#SBATCH --cpus-per-task=6

cd $SLURM_SUBMIT_DIR

module load anaconda3-2024.2
module load cuda-12.8

source ~/envs/selector_env/bin/activate

# ── Verify GPU ──
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
python -c "import torch; print('CUDA version:', torch.version.cuda)"

export TF_ENABLE_ONEDNN_OPTS=0

# ── HuggingFace token (set once, reused from cache) ──
# export HF_TOKEN="your_token_here"
export HF_HOME="/home/sudhakar/Vedang_U23AI111_8817182016/hf_cache"
export TRANSFORMERS_CACHE="/home/sudhakar/Vedang_U23AI111_8817182016/hf_cache"

# ── Create required directories ──
mkdir -p /home/sudhakar/Vedang_U23AI111_8817182016/memory_agent/logs
mkdir -p /home/sudhakar/Vedang_U23AI111_8817182016/memory_agent/results
mkdir -p /home/sudhakar/Vedang_U23AI111_8817182016/hf_cache

cd /home/sudhakar/Vedang_U23AI111_8817182016/memory_agent

# ── Run evaluation ──
# Change --user_id to distinguish runs (memory, baseline, rag)
python src/eval.py \
    --model_id "google/gemma-2-2b-it" \
    --n_sessions 5 \
    --user_id "eval_memory_run1" \
    --output_dir "results/" \
    --problems_path "problems/calibrated_20.json" \
    --memory_layers "4,8,16,24" \
    --hf_token "$HF_TOKEN"
