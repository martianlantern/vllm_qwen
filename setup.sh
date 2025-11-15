#!/bin/bash
set -e  # Exit on error

echo "=========================================="
echo "vLLM Development Environment Setup"
echo "=========================================="

# Set CUDA device to only use cuda:0
export CUDA_VISIBLE_DEVICES=0
echo "✓ CUDA_VISIBLE_DEVICES set to 0 (using only cuda:0)"

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    echo "✓ uv installed successfully"
else
    echo "✓ uv is already installed"
fi

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "Working directory: $SCRIPT_DIR"

# Create virtual environment with uv if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment with uv..."
    uv venv .venv --python 3.11
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate the virtual environment
source .venv/bin/activate
echo "✓ Virtual environment activated"

# Upgrade pip and install build tools
echo "Installing build dependencies..."
uv pip install --upgrade pip setuptools wheel

# Install build requirements
echo "Installing build requirements from requirements/build.txt..."
if [ -f "requirements/build.txt" ]; then
    uv pip install -r requirements/build.txt
else
    # Install build requirements from pyproject.toml manually if build.txt doesn't exist
    uv pip install cmake>=3.26 ninja packaging "setuptools>=61" "setuptools-scm>=8.0" "torch==2.6.0" wheel jinja2
fi
echo "✓ Build dependencies installed"

# Install CUDA-specific dependencies
echo "Installing CUDA dependencies..."
uv pip install -r requirements/cuda.txt
echo "✓ CUDA dependencies installed"

# Install common dependencies
echo "Installing common dependencies..."
uv pip install -r requirements/common.txt
echo "✓ Common dependencies installed"

# Install vLLM in editable mode
echo "Installing vLLM in editable/development mode..."
uv pip install -e .
echo "✓ vLLM installed in editable mode"

# Verify installation
echo ""
echo "=========================================="
echo "Verifying installation..."
echo "=========================================="
python -c "import vllm; print(f'vLLM version: {vllm.__version__}')"
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"
python -c "import torch; print(f'Number of GPUs: {torch.cuda.device_count() if torch.cuda.is_available() else 0}')"

echo ""
echo "=========================================="
echo "Setup completed successfully!"
echo "=========================================="
echo ""
echo "To activate the environment in the future, run:"
echo "  source .venv/bin/activate"
echo ""
echo "CUDA_VISIBLE_DEVICES is set to 0 for this session."
echo "To persist this setting, add the following to your ~/.bashrc:"
echo "  export CUDA_VISIBLE_DEVICES=0"
echo ""
echo "Any code changes you make will be immediately reflected when you run vLLM."
echo "=========================================="

