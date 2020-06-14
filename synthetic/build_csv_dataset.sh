#!/usr/bin/env bash

if [ ! -d data ]; then
  mkdir -p data/train data/test data/validation
  export PYTHONPATH=$(realpath ..)
  cd ..
  python3 synthetic/generate_data.py
fi
