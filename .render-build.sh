#!/usr/bin/env bash
set -o errexit

# Upgrade pip and install packages
pip install --upgrade pip
pip install --prefer-binary -r requirements.txt

# Download compatible spaCy model
python -m spacy download en_core_web_sm==3.6.0

# Set compilation flags for blis
export BLIS_ARCH="generic"
export NO_BLAS=1