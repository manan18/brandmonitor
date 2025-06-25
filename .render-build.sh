#!/usr/bin/env bash
set -o errexit

pip install --upgrade pip
pip install --prefer-binary -r requirements.txt

# Add this to link the model
python -m spacy link en_core_web_sm en_core_web_sm --force