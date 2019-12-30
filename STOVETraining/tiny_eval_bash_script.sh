#!/bin/bash
for i in `seq 16 200`;
do
python check_inference.py experiments/constrained-different-mean/run_033/config.yml --load-run --run-number 5 --checkpoint-number $i
done
