#!/bin/bash
#SBATCH --job-name=mem-calib
#SBATCH --partition=gpu
#SBATCH --nodelist=node2
#SBATCH --error=/home/sudhakar/Vedang_U23AI111_8817182016/memory_agent/logs/calib_%j.err
#SBATCH --output=/home/sudhakar/Vedang_U23AI111_8817182016/memory_agent/logs/calib_%j.out
#SBATCH --gres=shard:10
#SBATCH --cpus-per-task=6

cd $SLURM_SUBMIT_DIR

module load anaconda3-2024.2
module load cuda-12.8

source ~/envs/selector_env/bin/activate

python -c "import torch; print('CUDA:', torch.cuda.is_available())"

export TF_ENABLE_ONEDNN_OPTS=0
# export HF_TOKEN="your_token_here"
export HF_HOME="/home/sudhakar/Vedang_U23AI111_8817182016/hf_cache"
export TRANSFORMERS_CACHE="/home/sudhakar/Vedang_U23AI111_8817182016/hf_cache"

mkdir -p /home/sudhakar/Vedang_U23AI111_8817182016/memory_agent/logs
mkdir -p /home/sudhakar/Vedang_U23AI111_8817182016/memory_agent/problems

cd /home/sudhakar/Vedang_U23AI111_8817182016/memory_agent

# Run calibration scan (scans first 80 HumanEval problems,
# keeps the 20 in the 30–60% pass-rate sweet spot)
python src/eval.py \
    --model_id "google/gemma-2-2b-it" \
    --n_sessions 1 \
    --user_id "calib_scan" \
    --output_dir "results/" \
    --problems_path "problems/calibrated_20.json" \
    --memory_layers "4,8,16,24" \
    --calibrate \
    --hf_token "$HF_TOKEN"
