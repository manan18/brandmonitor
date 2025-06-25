#!/usr/bin/env bash
set -o errexit

pip install --upgrade pip
pip install --prefer-binary -r requirements.txt  # Now handles model install

export BLIS_ARCH="generic"
export NO_BLAS=1