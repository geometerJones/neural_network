#!/bin/bash

rate="1.0 0.5 0.25 0.1 0.05 0.01"
type="custom"


for r in ${rate}; do
  for t in ${type}; do
    bsub -o logs/${t}.115.${r}.log \
      -q week \
      -M 8 \
      -R "rusage[mem=5000]" \
      -R "rusage[indium_io=50]" \
      python neural_net_main.py -e 115 -r ${r} -t ${t} \
      || exit 1
  done
done
