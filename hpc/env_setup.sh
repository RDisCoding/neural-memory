#!/bin/bash
# env_setup.sh
# Run ONCE on the HPC to install all dependencies into selector_env.
# Your selector_env already has most packages from your previous work.
# This adds only what's new for this project.
#
# Usage:
#   bash hpc/env_setup.sh

module load anaconda3-2024.2
module load cuda-12.8

source ~/envs/selector_env/bin/activate

echo "Python: $(python --version)"
echo "Installing memory_agent dependencies..."

# Core (likely already installed from your RLVR project)
pip install --upgrade --quiet \
    torch \
    transformers \
    accelerate \
    datasets \
    huggingface_hub

# For the RAG baseline comparison
pip install --quiet faiss-cpu

echo "Done. Verifying..."
python -c "import torch, transformers, datasets, faiss; print('All imports OK')"
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
