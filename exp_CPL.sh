alg=supervised
batch_size=64
## unused args
kd_T=1.0
alpha=0.7
warmup=1

## PL
MoCo=false
for unlabel in in inout; do
  for task in semi_aves semi_fungi; do
    for init in imagenet inat; do

      if [ ${init} == scratch ]
      then
        ## From scratch ##
        init=scratch
        num_iter=50000
        lr=3e-2
        wd=3e-3
      elif [ ${init} == imagenet ]
      then
        ## From ImageNet ##
        init=imagenet
        num_iter=10000
        # lr=3e-3
        lr=1e-3
        wd=1e-4
      elif [ ${init} == inat ]
      then
        ## From iNat ##
        init=inat
        num_iter=10000
        # lr=3e-3
        lr=1e-3
        wd=1e-4
      fi

      exp_dir=${task}_CPL_${init}_${unlabel}_${lr}_${wd}_${num_iter}
      echo "${exp_dir}"
      out_path=slurm_out_0504/${exp_dir}
      err_path=slurm_err_0504/${exp_dir}
      export task init alg batch_size lr wd num_iter exp_dir unlabel MoCo kd_T alpha warmup
      sbatch --gres=gpu:1 -p 1080ti-long -o ${out_path}.out -e ${err_path}.err run_train_CPL.sbatch

    done
  done
done
