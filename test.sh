#!/bin/sh
PARTITION=Segmentation

dataset=$1
exp_name=$2

save_dir=output/${dataset}/${exp_name}
result_dir=${save_dir}/result
config=config/${dataset}/${dataset}_${exp_name}.yaml
mkdir -p ${save_dir} ${result_dir}
now=$(date +"%Y%m%d_%H%M%S")


python3 -u test.py --config=${config} 2>&1 | tee ${result_dir}/test-$now.log
