#!/bin/bash
for i in `seq 1 200`;
do
python check_inference.py experiments/visdom-test/run_029/config.yml --load-run --run-number 29 --checkpoint-number $i
done
