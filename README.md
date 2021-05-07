# A Realistic Evaluation of Semi-Supervised Learning for Fine-Grained Classification

Code for the paper "[A Realistic Evaluation of Semi-Supervised Learning for Fine-Grained Classification](https://arxiv.org/abs/2104.00679)" at CVPR 2021, by Jong-Chyi Su, Zezhou Cheng, and Subhransu Maji. 

## Preparing Datasets and Data Splits
We used the following datasets in the paper:
- **Semi-Aves**: This is the dataset of the [Semi-Aves Challenge](https://github.com/cvl-umass/semi-inat-2020) at [FGVC7 workshop](https://sites.google.com/view/fgvc7) at CVPR 2020.
- **Semi-Fungi**: We created a split from the dataset of [2018 FGVCx Fungi Classification Challenge](https://github.com/visipedia/fgvcx_fungi_comp) at [FGVC5 workshop](https://sites.google.com/view/fgvc5) at CVPR 2018.
- **CUB**: We created a split from the [Caltech-UCSD Birds-200-2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) dataset.

After this paper was submitted, we collected a new Semi-iNat dataset for the semi-supervised challenge this year:
- **Semi-iNat**: This is a new dataset for the [Semi-iNat Challenge](https://github.com/cvl-umass/semi-inat-2021) at [FGVC8 workshop](https://sites.google.com/view/fgvc8) at CVPR 2021. Different from Semi-Aves, Semi-iNat has more species from different kingdoms, has no domain label. For more details please see the [challenge website](https://github.com/cvl-umass/semi-inat-2021).

The splits of the above datasets can be found under ```data/${dataset}/${split}.txt```. Here are the splits for each datasets:
- l_train
- u_train_in
- u_train_out
- u_train (combining u_train_in and u_train_out)
- val
- l_train_val (combining l_train and val)
- test

Each line of the text files include the filename and the label.

Please download the datasets from the corresponding websites.\
For Semi-Aves, put the data under `data/semi_aves`.\
For Semi-Fungi and CUB, download images and put them under `data/semi_fungi/images` and `data/cub/images`.

Note 1: For experiments of Semi-Fungi in the paper, we first resize the images to a maximum of 300px for each side.\
Note 2: We reported the results of another split of Semi-Aves in the appendix (for cross-validation), but we do not release the labels because it will leak the labels for unlabeled data.\
Note 3: We also provided the species names of Semi-Aves under ```data/semi_aves_species_names.txt```, and the species names of Semi-Fungi.

## Training
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

## Pre-Trained Models
We provide supervised training models, MoCo pre-trained models, as well as MoCo + supervised training models, for both Semi-Aves and Semi-Fungi datasets. 
Here are the links to download the model:

```http://vis-www.cs.umass.edu/semi-inat-2021/ssl_evaluation/models/${method}/${dataset}_${initialization}_${unlabel}.pth.tar```

- ${method}: choose from {supervised, MoCo_init, MoCo_supervised}
- ${dataset}: choose from {semi_aves, semi_fungi}
- ${initialization}: choose from {scratch, imagenet, inat}
- ${unlabel}: choose from {in, inout}

You need these models for self-training mothods. For example, the teacher model is initialized from `model/supervised` for self-training. For MoCo + self-training, the teacher model is initialized from `model/MoCo_supervised`, and the student model is initialized from `model/MoCo_init`.

We also provide the [pre-trained ResNet-50 model of iNaturalist-18](http://vis-www.cs.umass.edu/semi-inat-2021/ssl_evaluation/models/inat_resnet50.pth.tar).
This model was trained using this [github code](https://github.com/macaodha/inat_comp_2018).

## Related Challenges
* [Semi-iNat 2021 Competition at FGVC8](https://github.com/cvl-umass/semi-inat-2021)
* [Semi-Aves 2020 Competition at FGVC7](https://github.com/cvl-umass/semi-inat-2020)

## Paper and Citation 
**A Realistic Evaluation of Semi-Supervised Learning for Fine-Grained Classification**\
Jong-Chyi Su, Zezhou Cheng, Subhransu Maji\
CVPR 2021 (oral)\
[arXiv link](https://arxiv.org/abs/2104.00679)\
Bibtex
```
@inproceedings{su2021realistic,
  author    = {Jong{-}Chyi Su and Zezhou Cheng and Subhransu Maji},
  title     = {A Realistic Evaluation of Semi-Supervised Learning for Fine-Grained Classification},
  booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2021}
}
```
