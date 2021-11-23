# Semi-Supervised Learning for Fine-Grained Classification

This repo contains the code of:

- **A Realistic Evaluation of Semi-Supervised Learning for Fine-Grained Classification**, Jong-Chyi Su, Zezhou Cheng, and Subhransu Maji, CVPR 2021. [[paper](https://arxiv.org/abs/2104.00679), [poster](https://people.cs.umass.edu/~jcsu/papers/ssl_evaluation/poster.pdf), [slides](https://people.cs.umass.edu/~jcsu/papers/ssl_evaluation/slides.pdf)]
- **Semi-Supervised Learning with Taxonomic Labels**, Jong-Chyi Su and Subhransu Maji, BMVC 2021. [[paper](), [slides](https://people.cs.umass.edu/~jcsu/papers/ssl_evaluation/slides_bmvc.pdf)]

## Preparing Datasets and Splits
We used the following datasets in the paper:
- **Semi-Aves**: dataset of the [Semi-Aves Challenge](https://github.com/cvl-umass/semi-inat-2020) at [FGVC7 workshop](https://sites.google.com/view/fgvc7) at CVPR 2020.
- **Semi-Fungi**: dataset build from the [2018 FGVCx Fungi Classification Challenge](https://github.com/visipedia/fgvcx_fungi_comp) at [FGVC5 workshop](https://sites.google.com/view/fgvc5) at CVPR 2018.
- **Semi-CUB**: dataset build from the [Caltech-UCSD
  Birds-200-2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html)
  dataset.
  

In addition the repository contains a new Semi-iNat dataset corresponding to
the FGVC8 semi-supervised challenge:
- **Semi-iNat**: This is a new dataset for the [Semi-iNat
  Challenge](https://github.com/cvl-umass/semi-inat-2021) at [FGVC8
  workshop](https://sites.google.com/view/fgvc8) at
  CVPR 2021. Different from Semi-Aves, Semi-iNat has more species from
  different kingdoms, and does not include in or out-of-domain label. 
  For more details please see
  the [challenge
  website](https://github.com/cvl-umass/semi-inat-2021).

The splits of each of these datasets can be found under
```data/${dataset}/${split}.txt``` corresponding to:
- l_train -- labeled in-domain data
- u_train_in -- unlabeled in-domain data
- u_train_out -- unlabeled out-of-domain data
- u_train (combines u_train_in and u_train_out)
- val -- validation set
- l_train_val (combines l_train and val)
- test -- test set

Each line in the text file has a filename and the corresponding class label.

Please download the datasets from the corresponding websites.
For Semi-Aves, put the data under `data/semi_aves`.
FFor Semi-Fungi and Semi-CUB, download the images and put them under
`data/semi_fungi/images` and `data/cub/images`.

**Note 1:** For the experiments on Semi-Fungi reported in the paper, the
images are resized to a maximum of 300px for each side.\
**Note 2:** We reported the results of another split of Semi-Aves in
the appendix (for cross-validation), but we do not release the labels
because it will leak the labels for unlabeled data. \
**Note 3:** We also provide the species names of Semi-Aves under
```data/semi_aves_species_names.txt```, and the species names of
Semi-Fungi. The names were not shared in the competetion.


## Training and Evaluation (CVPR paper)
We provide the code for all the methods included in the paper, except for FixMatch and MoCo. 
This includes methods of supervised training, self-training, PL, and curriculum PL.
This code is developed based on [this PyTorch implementation](https://github.com/perrying/realistic-ssl-evaluation-pytorch).

For FixMatch, we used the official [Tensorflow code](https://github.com/google-research/fixmatch) and an [unofficial PyTorch code](https://github.com/kekmodel/FixMatch-pytorch) to reproduce the results. 
For MoCo, we use this [PyContrast implementation](https://github.com/HobbitLong/PyContrast). 

To train the model, use the following command:
```
CUDA_VISIBLE_DEVICES=0 python run_train.py --task ${task} --init ${init} --alg ${alg} --unlabel ${unlabel} --num_iter ${num_iter} --warmup ${warmup} --lr ${lr} --wd ${wd} --batch_size ${batch_size} --exp_dir ${exp_dir} --MoCo ${MoCo} --alpha ${alpha} --kd_T ${kd_T} --trainval
```
For example, to train a supervised model initialized from a inat pre-trained model on semi-aves dataset with in-domain unlabeled data only, you will use:
```
CUDA_VISIBLE_DEVICES=0 python run_train.py --task semi_aves --init inat --alg supervised --unlabel in --num_iter 10000 --lr 1e-3 --wd 1e-4 --exp_dir semi_aves_supervised_in --MoCo false --trainval
```
Note that for experiments of Semi-Aves and Semi-Fungi in the paper, we combined the training and val set for training (use args `--trainval`).\
For all the hyper-parameters, please see the following shell scripts:
- `exp_sup.sh` for supervised training
- `exp_PL.sh` for pseudo-labeling
- `exp_CPL.sh` for curriculum pseudo-labeling
- `exp_MoCo.sh` for MoCo + supervised training
- `exp_distill.sh` for self-training and MoCo + self-training 

## Training and Evaluation (BMVC paper)
In our BMVC paper, we added the hierarchical supervision of coarse labels on top of semi-supervised learning.

To train the model, use the following command:
```
CUDA_VISIBLE_DEVICES=0 python run_train_hierarchy.py --task ${task} --init ${init} --alg ${alg} --unlabel ${unlabel} --num_iter ${num_iter} --warmup ${warmup} --lr ${lr} --wd ${wd} --batch_size ${batch_size} --exp_dir ${exp_dir} --MoCo ${MoCo} --alpha ${alpha} --kd_T ${kd_T} --level ${level}
```
The following are the arguments different from the above:
- ${level}: choose from {genus, kingdom, phylum, class, order, family, species}
- ${alg}: choose from {hierarchy, PL_hierarchy, distill_hierarchy}

For the settings and hyper-parameters, please see `exp_hierarchy.sh`.

## Pre-Trained Models
We provide supervised training models, MoCo pre-trained models, as
well as MoCo + supervised training models, for both Semi-Aves and
Semi-Fungi datasets. Here are the links to download the model:

```http://vis-www.cs.umass.edu/semi-inat-2021/ssl_evaluation/models/${method}/${dataset}_${initialization}_${unlabel}.pth.tar```

- ${method}: choose from {supervised, MoCo_init, MoCo_supervised}
- ${dataset}: choose from {semi_aves, semi_fungi}
- ${initialization}: choose from {scratch, imagenet, inat}
- ${unlabel}: choose from {in, inout}

You need these models for self-training mothods. For example, the
teacher model is initialized from `model/supervised` for
self-training. For MoCo + self-training, the teacher model is
initialized from `model/MoCo_supervised`, and the student model is
initialized from `model/MoCo_init`.

We also provide the [pre-trained ResNet-50 model of
iNaturalist-18](http://vis-www.cs.umass.edu/semi-inat-2021/ssl_evaluation/models/inat_resnet50.pth.tar). This
model was trained using this [github
code](https://github.com/macaodha/inat_comp_2018).

## Related Challenges
* Semi-iNat 2021 Competition at FGVC8: [[challenge website](https://github.com/cvl-umass/semi-inat-2021), [kaggle](https://www.kaggle.com/c/semi-inat-2021), [tech report](https://arxiv.org/abs/2106.01364)]
* Semi-Aves 2020 Competition at FGVC7: [[challenge website](https://github.com/cvl-umass/semi-inat-2020), [kaggle](https://www.kaggle.com/c/semi-inat-2020), [tech report](https://arxiv.org/abs/2103.06937)]

## Citation 
```
@inproceedings{su2021realistic,
  author    = {Jong{-}Chyi Su and Zezhou Cheng and Subhransu Maji},
  title     = {A Realistic Evaluation of Semi-Supervised Learning for Fine-Grained Classification},
  booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2021}
}

@inproceedings{su2021taxonomic,
  author    = {Jong{-}Chyi Su and Subhransu Maji},
  title     = {Semi-Supervised Learning with Taxonomic Labels},
  booktitle = {British Machine Vision Conference (BMVC)},
  year      = {2021}
}

@article{su2021semi_iNat,
      title={The Semi-Supervised iNaturalist Challenge at the FGVC8 Workshop}, 
      author={Jong-Chyi Su and Subhransu Maji},
      year={2021},
      journal={arXiv preprint arXiv:2106.01364}
}

@article{su2021semi_aves,
      title={The Semi-Supervised iNaturalist-Aves Challenge at FGVC7 Workshop}, 
      author={Jong-Chyi Su and Subhransu Maji},
      year={2021},
      journal={arXiv preprint arXiv:2103.06937}
}
```
