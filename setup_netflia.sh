#!/bin/bash
echo "Activando entorno..."
conda activate sd_env

echo "Desinstalando versiones conflictivas..."
pip uninstall torch torchvision torchaudio xformers -y

echo "Instalando PyTorch 2.4.1 + CUDA 12.4..."
pip install torch==2.4.1+cu124 torchvision==0.19.1+cu124 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124 -q

echo "Instalando xformers 0.0.28.post1 (rápido + bajo VRAM)..."
pip install xformers==0.0.28.post1 --index-url https://download.pytorch.org/whl/cu124 -q

echo "Actualizando diffusers y dependencias..."
pip install --upgrade diffusers[torch] transformers accelerate safetensors opencv-python -q

echo "Verificando instalación..."
python -c "import torch, torchvision, xformers; print(f'torch: {torch.__version__} | xformers: {xformers.__version__} | CUDA: {torch.version.cuda}')"

echo "¡Listo! Ejecutá: python app.py"