#!/bin/bash
for i in `seq 1 200`;
do
	python check_inference.py experiments/slac-model/run_019/config.yml --load-run --run-number 19 --checkpoint-number $i --epochs 1
done
