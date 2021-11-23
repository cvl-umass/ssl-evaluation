#!/bin/bash
#
# run_train.sbatch
#
#SBATCH --job-name=hierarchy
#SBATCH --output=res_%j.txt           # output file
#SBATCH -N 1                          # Ensure that all cores are on one machine
#SBATCH --mem=100GB                     # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o hostname_%j.out            # File to which STDOUT will be written
#SBATCH -e hostname_%j.err            # File to which STDERR will be written
##SBATCH --mail-type=FAIL               # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --gres=gpu:1

srun -u python run_train_hierarchy.py --task ${task} --init ${init} --alg ${alg} --unlabel ${unlabel} \
	 --num_iter ${num_iter} --warmup ${warmup} --lr ${lr} --wd ${wd} --batch_size ${batch_size} \
	 --exp_dir ${exp_dir} --MoCo ${MoCo} --alpha ${alpha} --kd_T ${kd_T} --level ${level} 
exit
