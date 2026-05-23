#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="omnipose"
PYTHON_VERSION="3.10.12"

echo "==> Checking for conda..."
if ! command -v conda >/dev/null 2>&1; then
  echo "Error: conda was not found in PATH."
  echo "Please install Conda or Miniconda first."
  exit 1
fi

# Make 'conda activate' available in non-interactive shells
source "$(conda info --base)/etc/profile.d/conda.sh"

echo "==> Creating conda environment: ${ENV_NAME}"
if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "Environment '${ENV_NAME}' already exists; skipping creation."
else
  conda create -y -n "${ENV_NAME}" python="${PYTHON_VERSION}" pytorch
fi

echo "==> Activating environment"
conda activate "${ENV_NAME}"

echo "==> Removing old cellpose_omni install if present"
if python -m pip show cellpose_omni >/dev/null 2>&1; then
  python -m pip uninstall -y cellpose_omni
fi

python -m pip cache remove cellpose_omni >/dev/null 2>&1 || true

echo "==> Installing/upgrading Omnipose"
python -m pip install --upgrade pip
python -m pip install omnipose

echo "==> Installation complete"
echo "Launching Omnipose GUI..."
python -m omnipose
