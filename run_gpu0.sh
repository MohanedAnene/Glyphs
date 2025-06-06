#!/bin/bash
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=0

cound=0
for bins in 5 10; do
	for rot in 0 1 4 10; do
		for stepsize in 10 5 2; do
			for gamma in 1.0 0.9 0.5 0.1; do
				echo "Runnning on GPU 0: bins=$bins rot=$rot stepsize=$stepsize gamma=$gamma"
				python binned_regression_experiment.py cuda=0 num_bins=$bins rotation=$rot stepsize=$stepsize gamma=$gamma
			done
		done
	done
done
