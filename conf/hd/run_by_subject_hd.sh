#!/bin/sh

for i in {0..11}; do 
	echo "————— Subject ${i} —————" | tee -a logs/run_by_subject_hd.txt
	PYTHONPATH=. ./bin/qualia conf/hd/ResNet_float32_train.toml preprocess_data --bench.name="'HD_ResNet_float32_train_${i}'" --dataset.params.test_subjects="[${i}]" | tee -a logs/run_by_subject_hd.txt
	PYTHONPATH=. ./bin/qualia conf/hd/ResNet_float32_train.toml train --bench.name="'HD_ResNet_float32_train_${i}'" --dataset.params.test_subjects="[${i}]" | tee -a logs/run_by_subject_hd.txt
done

