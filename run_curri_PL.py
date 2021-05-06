import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

import os, argparse, random, math
import copy, logging, sys, time, shutil, json
from tensorboardX import SummaryWriter

from lib import wrn, transform
from lib.initialize import initialize_model
from training import *
from lib.datasets.iNatDataset import iNatDataset

dset_root = {}
dset_root['cub'] = 'data/cub/images'
dset_root['semi_fungi'] = 'data/semi_fungi'
dset_root['semi_aves'] = 'data/semi_aves'
dset_root['semi_aves_2'] = 'data/semi_aves_2'


class RandomSampler(torch.utils.data.Sampler):
    """ sampling without replacement """
    def __init__(self, num_data, num_sample):
        iterations = num_sample // num_data + 1
        self.indices = torch.cat([torch.randperm(num_data) for _ in range(iterations)]).tolist()[:num_sample]

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def initializeLogging(log_filename, logger_name):
    log = logging.getLogger(logger_name)
    log.setLevel(logging.DEBUG)
    log.addHandler(logging.StreamHandler(sys.stdout))
    log.addHandler(logging.FileHandler(log_filename, mode='a'))
    return log


def main(args):

    log_dir = os.path.join(args.exp_prefix, args.exp_dir, 'log')
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    else:
        os.makedirs(log_dir)
    writer = SummaryWriter(log_dir)

    json.dump(dict(sorted(vars(args).items())), open(os.path.join(args.exp_prefix, args.exp_dir, 'configs.json'),'w'))

    checkpoint_folder = os.path.join(args.exp_prefix, args.exp_dir, 'checkpoints')
    if not os.path.isdir(checkpoint_folder):
        os.makedirs(checkpoint_folder)

    logger_name = 'train_logger'
    logger = initializeLogging(os.path.join(args.exp_prefix, args.exp_dir, 'train_history.txt'), logger_name)

    # ==================  Craete data loader ==================================
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(args.input_size), 
            transforms.RandomResizedCrop(args.input_size),
            # transforms.ColorJitter(Brightness=0.4, Contrast=0.4, Color=0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(args.input_size), 
            transforms.CenterCrop(args.input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
    data_transforms['l_train'] = data_transforms['train']
    data_transforms['u_train'] = data_transforms['train']
    data_transforms['val'] = data_transforms['test']

    root_path = dset_root[args.task]

    if args.trainval:
        ## use l_train + val for labeled training data
        l_train = 'l_train_val'
    else:
        l_train = 'l_train'

    if args.unlabel == 'in':
        u_train = 'u_train_in'
    elif args.unlabel == 'inout':
        u_train = 'u_train'

    ## set val to test when using l_train + val for training
    if args.trainval:
        split_fname = [l_train, u_train, 'test', 'test']
    else:
        split_fname = [l_train, u_train, 'val', 'test']

    image_datasets = {split: iNatDataset(root_path, split_fname[i], args.task,
        transform=data_transforms[split]) \
        for i,split in enumerate(['l_train', 'u_train', 'val', 'test'])}

    print("labeled data : {}, unlabeled data : {}".format(len(image_datasets['l_train']), len(image_datasets['u_train'])))
    print("validation data : {}, test data : {}".format(len(image_datasets['val']), len(image_datasets['test'])))

    if args.task == 'cifar10' or args.task == 'svhn' or args.task == 'stl10':
        num_classes = 10
    else:
        num_classes = image_datasets['l_train'].get_num_classes()
    
    print("#classes : {}".format(num_classes))

    dataloaders_dict = {}
    if args.alg != 'supervised':
        dataloaders_dict['l_train'] = DataLoader(image_datasets['l_train'],
                    batch_size=args.batch_size//2, num_workers=args.num_workers, drop_last=True,
                    sampler=RandomSampler(len(image_datasets['l_train']), args.num_iter * args.batch_size//2))
    else:
        dataloaders_dict['l_train'] = DataLoader(image_datasets['l_train'],
                    batch_size=args.batch_size, num_workers=args.num_workers, drop_last=True,
                    sampler=RandomSampler(len(image_datasets['l_train']), args.num_iter * args.batch_size))

    dataloaders_dict['u_train'] = DataLoader(image_datasets['u_train'],
                    batch_size=args.batch_size//2, num_workers=args.num_workers, drop_last=True,
                    sampler=RandomSampler(len(image_datasets['u_train']), args.num_iter * args.batch_size//2))
    dataloaders_dict['val'] = DataLoader(image_datasets['val'],
                    batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)
    dataloaders_dict['test'] = DataLoader(image_datasets['test'],
                    batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #======================= Initialize the model ==============================
    model_ft = initialize_model(args.model, num_classes, feature_extract=False, 
                    use_pretrained=args.init=='imagenet', logger=logger)
    
    #======================= Class Weight ==============================
    ## This could be added
    cls_weight = [1.0 for tt in range(num_classes)]
    cls_weight = torch.tensor(cls_weight,dtype=torch.float).cuda()

    #======================= Set the loss ==============================
    if args.alg == "distill":
        from KL import DistillKL
        ssl_obj = DistillKL(args.kd_T)
    elif args.alg == "PL":
        from lib.algs.pseudo_label import PL
        ssl_obj = PL(args.threshold, num_classes)
    elif args.alg == "supervised":
        ssl_obj = None
    else:
        raise ValueError("{} is unknown algorithm".format(args.alg))

    criterion = nn.CrossEntropyLoss(weight=cls_weight, ignore_index=-1)

    #====================== Initialize optimizer ==============================
    optimizer = optim.SGD(model_ft.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_iter)

    #====================== Initialize model ==============================
    start_iter = 0
    best_acc = 0.0
    # load from checkpoint if exists
    if args.continue_training or args.load_dir != '':
        if args.load_dir != '':
        	## Loading pre-trained model
            checkpoint_filename = args.load_dir
        else:
        	## Continue training, loading from previous checkpoint
            checkpoint_filename = os.path.join(checkpoint_folder, 'checkpoint.pth.tar')

        if os.path.isfile(checkpoint_filename):
            print("=> loading checkpoint '{}'".format(checkpoint_filename))
            checkpoint = torch.load(checkpoint_filename)
            
            if args.load_dir != '':
            	## Loading pre-trained model
                start_iter = 0
                best_acc = 0.0
                model_ft.fc = nn.Linear(2048, num_classes)

                if args.init == 'inat':
                    ## loading inat pre-trained model
                    model_ft = torch.nn.DataParallel(model_ft)
                    del checkpoint['state_dict']['module.fc.bias']
                    del checkpoint['state_dict']['module.fc.weight']
                    model_ft.load_state_dict(checkpoint['state_dict'], strict=False)
                else:
                    model_ft.load_state_dict(checkpoint['state_dict'])
            else:
            	## Continue training, loading from previous checkpoint
                start_iter = checkpoint['iteration']
                best_acc = checkpoint['best_acc']
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                model_ft.load_state_dict(checkpoint['model_state_dict'])

            if args.init == 'inat':
                print("=> loaded inat model from '{}'" .format(checkpoint_filename))
            else:
                print("=> loaded previous checkpoint from '{}' (iteration {})"
                      .format(checkpoint_filename, checkpoint['iteration']))
                
            del checkpoint
        else:
            print("=> Cannot find checkpoint '{}'" .format(checkpoint_filename))

    # parallelize the model if using multiple gpus
    print('using #GPUs:',torch.cuda.device_count())
    if torch.cuda.device_count() > 1:
        model_ft = torch.nn.DataParallel(model_ft)
    model_ft.to(device)

    # moving optimizer to gpu
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()

    model_teacher = None

    #====================== Init things for each round ==============================
    for rd in range(args.rounds):
        log_dir = os.path.join(args.exp_prefix, args.exp_dir, 'round'+str(rd+1), 'log')
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)
        else:
            os.makedirs(log_dir)
        writer = SummaryWriter(log_dir)
        logger = initializeLogging(os.path.join(args.exp_prefix, args.exp_dir, 'round'+str(rd+1), 
            'train_history.txt'), logger_name)

        logger.info('{} {}'.format('Round',rd+1))

        checkpoint_folder = os.path.join(args.exp_prefix, args.exp_dir, 'round'+str(rd+1), 'checkpoints')
        if not os.path.isdir(checkpoint_folder):
            os.makedirs(checkpoint_folder)
        #====================== Get Pseudo-labeles ==============================
        # Get PL images
        out_name = os.path.join(args.exp_prefix, args.exp_dir, 'round'+str(rd+1), 'pl.txt')
        if not os.path.exists(out_name):
            dataset = iNatDataset(root_path, u_train, args.task,
                                    transform=data_transforms['val'], return_name=True)
            dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, 
                                    drop_last=False, shuffle=False)
            get_pl(model_ft, dataloader, logger_name, thres=args.threshold, out_name=out_name, curriculum=True, ratio=(rd+1)/args.rounds)

        # Combine PL images
        print('read PL images from:', out_name)
        with open(out_name, 'r') as f:
            pl_list = f.readlines()

        pl_list = list(set(pl_list))
        print("Total number of PL images:", len(pl_list))

        dataset = iNatDataset(root_path, l_train, args.task, transform=data_transforms['train'], pl_list=pl_list)
        dataloaders_dict['l_train'] = DataLoader(image_datasets['l_train'],
                        batch_size=args.batch_size, num_workers=args.num_workers, drop_last=True,
                        sampler=RandomSampler(len(image_datasets['l_train']), args.num_iter * args.batch_size))

        ## Reload ImageNet model (train from it again instead of previous iter)
        del model_ft
        model_ft = initialize_model(args.model, num_classes, feature_extract=False, 
                        use_pretrained=args.init=='imagenet', logger=logger)
        ## load iNat pretrained model
        if args.init == 'inat':
            checkpoint = torch.load(args.load_dir)
            model_ft = torch.nn.DataParallel(model_ft)
            del checkpoint['state_dict']['module.fc.bias']
            del checkpoint['state_dict']['module.fc.weight']
            model_ft.load_state_dict(checkpoint['state_dict'], strict=False)

        # parallelize the model if using multiple gpus
        print('using #GPUs:',torch.cuda.device_count())
        if torch.cuda.device_count() > 1:
            model_ft = torch.nn.DataParallel(model_ft)

        model_ft.to(device)
        ## Reload optimizer
        optimizer = optim.SGD(model_ft.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_iter)

        # Train the model
        model_ft, val_acc_history = train_model(args, model_ft, model_teacher, dataloaders_dict, criterion, optimizer,
                logger_name=logger_name, checkpoint_folder=checkpoint_folder,
                start_iter=start_iter, best_acc=best_acc, writer=writer, ssl_obj=ssl_obj, scheduler=scheduler)

        logging.shutdown()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='semi_aves', type=str, 
            help='the name of the dataset')
    parser.add_argument('--model', default='resnet50', type=str,
            help='resnet50|resnet101|wrn')
    parser.add_argument('--batch_size', default=32, type=int,
            help='size of mini-batch')
    parser.add_argument('--num_iter', default=200, type=int,
            help='number of iterations')
    parser.add_argument('--exp_prefix', default='results', type=str,
            help='path to the chekcpoint folder for the experiment')
    parser.add_argument('--exp_dir', default='exp', type=str,
            help='path to the chekcpoint folder for the experiment')
    parser.add_argument('--continue_training', action='store_true',
            help='train the model from last checkpoint')
    parser.add_argument('--load_dir', default='', type=str,
            help='load pretrained model from')
    parser.add_argument('--input_size', default=224, type=int, 
            help='input image size')
    parser.add_argument("--alg", "-a", default="supervised", type=str, 
            help="ssl algorithm : [supervised, PL, distill]")
    parser.add_argument("--em", default=0, type=float, 
            help="coefficient of entropy minimization. If you try VAT + EM, set 0.06")
    parser.add_argument('--num_workers', default=12, type=int)
    parser.add_argument("--root", "-r", default="data", type=str, help="dataset dir for cifar and svhn")
    parser.add_argument('--val_freq', default=100, type=int,
            help='do val every x iter')
    parser.add_argument('--print_freq', default=100, type=int,
            help='show train loss/acc every x iter')
    parser.add_argument("--wd", default=1e-4, type=float, 
            help="weight decay")
    parser.add_argument('--trainval', action='store_true', 
            help='use {train+val,test,test} for {train,val,test}')
    

    ### learning rate setup ###
    parser.add_argument("--lr", default=1e-3, type=float, 
            help="learning rate")
    parser.add_argument('--warmup', default=1000, type=int, 
            help='warmup iterations, only used for SSL methods')

    ### Semi-supervised loss ###
    # SSL algorithms: [PI, MT, VAT, PL, ICT, MM]
    ## for all SSL
    parser.add_argument("--consis_coef", default=1.0, type=float)
    ## PL
    parser.add_argument("--threshold", default=0.95, type=float)
    # ## MM
    # parser.add_argument("--T", default=0.5, type=float)
    # parser.add_argument("--K", default=2, type=int)
    # ## VAT
    # parser.add_argument("--eps", default=1, type=float)
    # parser.add_argument("--xi", default=1, type=float)
    # ## MT and ICT
    # parser.add_argument("--ema_factor", default=0.95, type=float)
    ## MM and ICT
    parser.add_argument("--alpha", default=0.1, type=float)
    
    ### Optimizer ###    
    parser.add_argument('--unlabel', default='in', type=str, 
            choices=['in','inout'], help='U_in or U_in + U_out')

    ### Release ###
    parser.add_argument('--init', default='scratch', type=str, 
            choices=['scratch','imagenet','inat'], 
            help='flag on for using pre-trained model')
    # parser.add_argument('--MoCo', action='store_true', 
    #         help='Use MoCo pre-trained model for supervised or self-training')
    parser.add_argument('--MoCo', default='false', type=str,
            help='Use MoCo pre-trained model for supervised or self-training')

    ### Self-training ###
    parser.add_argument('--path_t', default='', type=str, 
            help='use iNat/MoCo pretrained model')
    parser.add_argument("--kd_T", default=1.0, type=float, 
            help='temperature for distillation')

    parser.add_argument("--rounds", default=5, type=int)

    args = parser.parse_args()

    if args.MoCo == 'true':
        args.MoCo = True
    elif args.MoCo == 'false':
        args.MoCo = False

    if args.init == 'inat':
        args.load_dir = 'models/inat_resnet50.pth.tar'

    if args.alg == 'distill':
        if args.MoCo:
            args.path_t = 'models/MoCo_supervised/' + args.task + '_' + args.init + '_' + args.unlabel
            args.load_dir = 'models/MoCo_init/' + args.task + '_' + args.init + '_' + args.unlabel
        else:
            args.path_t = 'models/supervised/' + args.task + '_' + args.init
        args.path_t += '.pth.tar'

    main(args)



