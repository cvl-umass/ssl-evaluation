task=semi_inat
batch_size=60
level=species
# only used for distillation
kd_T=1.0
alpha=1.0
# only used for PL
warmup=1

## Choose the algorithm
alg=sup_hie
# alg=PL_hie
# alg=MoCo_hie
# alg=ST_hie
# alg=MoCoST_hie
# alg=transfer
# level=species

################################
#### Supervised + hierarchy ####
################################
if [ ${alg} == sup_hie ]
then
alg=hierarchy
MoCo=false
for level in genus kingdom phylum class order family species; do
  for unlabel in in inout; do
    for init in imagenet; do

      if [ ${init} == scratch ]
      then
        ## From scratch ##
        init=scratch
        num_iter=100000
        lr=3e-3
        wd=1e-3
      elif [ ${init} == imagenet ]
      then
        ## From ImageNet ##
        init=imagenet
        num_iter=50000
        lr=3e-3
        wd=1e-4
      elif [ ${init} == inat ]
      then
        ## From iNat ##
        init=inat
        num_iter=50000
        lr=3e-3
        wd=1e-4
      fi

      ## only species loss for labeled data
      exp_dir=${task}_hierarchy_${level}_${init}_${unlabel}_${lr}_${wd}_${num_iter}
      echo "${exp_dir}"
      out_path=slurm_out_bmvc/${exp_dir}
      err_path=slurm_err_bmvc/${exp_dir}
      export task init alg batch_size lr wd num_iter exp_dir unlabel MoCo kd_T alpha warmup level
      sbatch --gres=gpu:1 -p 1080ti-long -o ${out_path}.out -e ${err_path}.err run_hierarchy.sbatch

    done
  done
done

################################
######## PL + hierarchy ########
################################
elif [ ${alg} == PL_hie ]
then
alg=PL_hierarchy
MoCo=false
kd_T=1.0
alpha=1.0
warmup=1
unlabel=inout
for level in phylum; do
  for unlabel in in inout; do
    for init in imagenet inat; do

      if [ ${init} == scratch ]
      then
        ## From scratch ##
        init=scratch
        num_iter=100000
        warmup=10000
        lr=3e-2
        wd=1e-4
        threshold=0.95
      elif [ ${init} == imagenet ]
      then
        ## From ImageNet ##
        init=imagenet
        num_iter=50000
        warmup=5000
        # lr=1e-3
        lr=3e-3
        wd=1e-4
        threshold=0.85
      elif [ ${init} == inat ]
      then
        ## From iNat ##
        init=inat
        num_iter=50000
        warmup=5000
        # lr=1e-3
        lr=3e-3
        wd=1e-4
        threshold=0.95
      fi

      exp_dir=${task}_PL_hie_${init}_${unlabel}_${lr}_${wd}_${num_iter}
      echo "${exp_dir}"
      out_path=slurm_out_bmvc/${exp_dir}
      err_path=slurm_err_bmvc/${exp_dir}

      export task init alg batch_size lr wd num_iter exp_dir unlabel MoCo kd_T alpha warmup level
      sbatch --gres=gpu:1 -p 1080ti-long -o ${out_path}.out -e ${err_path}.err run_hierarchy.sbatch

    done
  done
done

##################################
######## MoCo + hierarchy ########
##################################
elif [ ${alg} == MoCo_hie ]
then
alg=hierarchy
MoCo=true
unlabel=inout
for level in phylum; do
  for unlabel in in inout; do
    for init in imagenet inat; do

      if [ ${init} == scratch ]
      then
        ## From scratch ##
        init=scratch
        num_iter=100000
        lr=3e-3
        wd=1e-3
      elif [ ${init} == imagenet ]
      then
        ## From ImageNet ##
        init=imagenet
        num_iter=50000
        lr=1e-2
        wd=1e-4
      elif [ ${init} == inat ]
      then
        ## From iNat ##
        init=inat
        num_iter=50000
        lr=1e-2
        wd=1e-4
      fi

      exp_dir=${task}_MoCo_hie_${init}_${unlabel}_${lr}_${wd}_${num_iter}
      echo "${exp_dir}"
      out_path=slurm_out_bmvc/${exp_dir}
      err_path=slurm_err_bmvc/${exp_dir}
      export task init alg batch_size lr wd num_iter exp_dir unlabel MoCo kd_T alpha warmup level
      sbatch --gres=gpu:1 -p 1080ti-long -o ${out_path}.out -e ${err_path}.err run_hierarchy.sbatch

    done
  done
done

################################
######## ST + hierarchy ########
################################
elif [ ${alg} == ST_hie ]
then
alg=distill_hierarchy
MoCo=false
kd_T=1.0
alpha=0.7
warmup=1
unlabel=inout
for level in phylum; do
  for unlabel in in inout; do
    for init in imagenet inat; do

      if [ ${init} == scratch ]
      then
        ## From scratch ##
        init=scratch
        num_iter=150000
        lr=3e-2
        wd=1e-3
      elif [ ${init} == imagenet ]
      then
        ## From ImageNet ##
        init=imagenet
        num_iter=50000
        lr=3e-3
        wd=1e-4
      elif [ ${init} == inat ]
      then
        ## From iNat ##
        init=inat
        num_iter=50000
        lr=3e-3
        wd=1e-4
      fi

      exp_dir=${task}_ST_hie_${init}_${unlabel}_${lr}_${wd}_${num_iter}
      echo "${exp_dir}"
      out_path=slurm_out_bmvc/${exp_dir}
      err_path=slurm_err_bmvc/${exp_dir}

      export task init alg batch_size lr wd num_iter exp_dir unlabel MoCo kd_T alpha warmup level
      sbatch --gres=gpu:1 -p 1080ti-long -o ${out_path}.out -e ${err_path}.err run_hierarchy.sbatch

    done
  done
done

####################################
######## MoCoST + hierarchy ########
####################################
elif [ ${alg} == MoCoST_hie ]
then
alg=distill_hierarchy
MoCo=true
kd_T=1.0
alpha=0.7
warmup=1
unlabel=inout
for level in phylum; do
  for unlabel in in inout; do
    for init in imagenet inat; do

      if [ ${init} == scratch ]
      then
        ## From scratch ##
        init=scratch
        num_iter=150000
        lr=3e-2
        wd=1e-3
      elif [ ${init} == imagenet ]
      then
        ## From ImageNet ##
        init=imagenet
        num_iter=50000
        lr=3e-3
        wd=1e-4
      elif [ ${init} == inat ]
      then
        ## From iNat ##
        init=inat
        num_iter=50000
        lr=3e-3
        wd=1e-4
      fi

      exp_dir=${task}_MoCoST_hie_${init}_${unlabel}_${lr}_${wd}_${num_iter}
      echo "${exp_dir}"
      out_path=slurm_out_bmvc/${exp_dir}
      err_path=slurm_err_bmvc/${exp_dir}

      export task init alg batch_size lr wd num_iter exp_dir unlabel MoCo kd_T alpha warmup level
      sbatch --gres=gpu:1 -p 1080ti-long -o ${out_path}.out -e ${err_path}.err run_hierarchy.sbatch

    done
  done
done 

fi
