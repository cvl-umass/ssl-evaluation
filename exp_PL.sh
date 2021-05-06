alg=PL
batch_size=60
## unused args
kd_T=1.0
alpha=0.7

## PL
MoCo=false
for unlabel in in inout; do
  for task in semi_aves semi_fungi; do
    for init in imagenet inat; do

      if [ ${init} == scratch ]
      then
        ## From scratch ##
        init=scratch
        num_iter=100000
        warmup=10000
        lr=3e-2
        wd=1e-4
        if [ ${unlabel} == in ]
        then
          threshold=0.95
        else
          threshold=0.95
        fi
      elif [ ${init} == imagenet ]
      then
        ## From ImageNet ##
        init=imagenet
        num_iter=10000
        warmup=1000
        # lr=3e-3
        lr=1e-3
        wd=1e-4
        if [ ${unlabel} == in ]
        then
          threshold=0.95
        else
          threshold=0.95
        fi     
      elif [ ${init} == inat ]
      then
        ## From iNat ##
        init=inat
        num_iter=10000
        warmup=1000
        # lr=3e-3
        lr=1e-3
        wd=1e-4
        if [ ${unlabel} == in ]
        then
          threshold=0.95
        else
          threshold=0.95
        fi
      fi

      exp_dir=${task}_${alg}_${init}_${unlabel}_${lr}_${wd}_${num_iter}
      echo "${exp_dir}"
      out_path=slurm_out_0504/${exp_dir}
      err_path=slurm_err_0504/${exp_dir}
      export task init alg batch_size lr wd num_iter exp_dir unlabel MoCo kd_T alpha warmup
      sbatch --gres=gpu:1 -p 1080ti-long -o ${out_path}.out -e ${err_path}.err run_train.sbatch

    done
  done
done
