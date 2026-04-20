#!/bin/bash
# Run analyze.py on all 5 APD runs and print summary.
set -e
cd "$(dirname "$0")"/..
for r in plain_20f_5n plain_20f_2n plain_100f_10n embed_20f_5n_D80 embed_20f_2n_D40; do
  if [ -d "apd_decomposition/out/$r" ]; then
    python3 -m apd_decomposition.analyze --runs "apd_decomposition/out/$r"
  fi
done
