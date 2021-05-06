#!/bin/bash
#
# run_train.sbatch
#
#SBATCH --job-name=CPL0504
#SBATCH --output=res_%j.txt           # output file
##SBATCH -n 6                          # Number of cores
#SBATCH -N 1                          # Ensure that all cores are on one machine
##SBATCH -t 0-04:00                    # Runtime in D-HH:MM
##SBATCH -p titanx-long               # Partition to submit to (serial_requeue)
#SBATCH --mem=100GB                     # Memory pool for all cores (see also --mem-per-cpu)
##SBATCH --mem-per-cpu=10240
#SBATCH -o hostname_%j.out            # File to which STDOUT will be written
#SBATCH -e hostname_%j.err            # File to which STDERR will be written
##SBATCH --mail-type=FAIL               # Type of email notification- BEGIN,END,FAIL,ALL
##SBATCH --mail-user=jcsu@cs.umass.edu # Email to which notifications will be sent
#SBATCH --gres=gpu:1
##SBATCH --nodelist=node022
#SBATCH --exclude=node124,node097,node051,node116

srun -u python run_curri_PL.py --task ${task} --init ${init} --alg ${alg} --unlabel ${unlabel} \
	 --num_iter ${num_iter} --warmup ${warmup} --lr ${lr} --wd ${wd} --batch_size ${batch_size} \
	 --exp_dir ${exp_dir} --MoCo ${MoCo} --alpha ${alpha} --kd_T ${kd_T} --trainval
exit

# srun -u python run_train.py --exp_dir ${exp_dir} --task ${task} --init ${init} --alg ${alg} --MoCo