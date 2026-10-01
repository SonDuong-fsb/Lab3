#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python -m pip install --upgrade pip
python -m pip install "setuptools==69.5.1" wheel
python -m pip install "$(grep -E '^numpy==' requirements.txt)"
python -m pip install --no-build-isolation "$(grep -E '^scikit-surprise==' requirements.txt)"
python -m pip install -r requirements.txt
