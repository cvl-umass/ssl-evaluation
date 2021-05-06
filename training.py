import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import os
import argparse, random, math
import copy, logging, sys, time, shutil, json
from tensorboardX import SummaryWriter
import torch.nn.functional as F


def save_checkpoint(state, is_best, checkpoint_folder='exp',
                filename='checkpoint.pth.tar'):
    filename = os.path.join(checkpoint_folder, filename)
    best_model_filename = os.path.join(checkpoint_folder, 'model_best.pth.tar')
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, best_model_filename)

def compute_correct(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k)
        return res


def get_pl(model, dataloader, logger_name, thres, out_name, curriculum=False, ratio=1.0):
    logger = logging.getLogger(logger_name)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()
    test_corrects_1 = 0
    test_corrects_5 = 0
    count = 0
    all_scores = []
    all_labels = []
    all_path = []
    f = open(out_name, 'w')
    for i,data in enumerate(dataloader):

        inputs, target, path = data
        inputs = inputs.to(device).float()
        target = target.to(device)
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            preds, labels = F.softmax(outputs, 1).max(1)

            ## Get PL and save to file
            count += sum(preds>thres).item()
            all_scores += preds.tolist()
            all_labels += labels.tolist()
            all_path += path

            if not curriculum:
                for j in range(preds.shape[0]):
                    if preds[j] > thres:
                        f.write('%s %d\n'%(path[j].replace('iNat_dump/',''), labels[j]))

            correct_1, correct_5 = compute_correct(outputs, target, topk=(1, 5))

        test_corrects_1 += correct_1.item()
        test_corrects_5 += correct_5.item()

    if curriculum:
        idx = np.argsort(-np.array(all_scores))
        num_pl = round(len(all_scores)*ratio)
        logger.info('Select {} PL Images'.format(num_pl))
        for i in range(num_pl):
            f.write('%s %d\n'%(all_path[idx[i]].replace('iNat_dump/',''), all_labels[idx[i]]))

    f.close()

    epoch_acc   = test_corrects_1 / len(dataloader.dataset)
    epoch_acc_5 = test_corrects_5 / len(dataloader.dataset)

    logger.info('{} Top1 Acc: {:.2f}% Top5 Acc: {:.2f}%'.format('Pseudo_label', epoch_acc*100, epoch_acc_5*100))


def test(model, dataloaders, args, logger, name="Best", criterion=nn.CrossEntropyLoss()):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()
    test_corrects_1 = 0
    test_corrects_5 = 0
    test_loss = 0
    for i,data in enumerate(dataloaders['test']):
        inputs, target = data
        inputs = inputs.to(device).float()
        target = target.to(device).long()

        ## upsample
        if args.input_size != inputs.shape[-1]:
            m = torch.nn.Upsample((args.input_size, args.input_size), mode='bilinear', align_corners=True)
            inputs = m(inputs)

        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            loss = criterion(outputs, target)

            correct_1, correct_5 = compute_correct(outputs, target, topk=(1, 5))
            test_loss += loss.item()

        test_corrects_1 += correct_1.item()
        test_corrects_5 += correct_5.item()

    epoch_loss  = test_loss / i
    epoch_acc   = test_corrects_1 / len(dataloaders['test'].dataset)
    epoch_acc_5 = test_corrects_5 / len(dataloaders['test'].dataset)

    logger.info('{} Loss: {:.4f} Top1 Acc: {:.2f}% Top5 Acc: {:.2f}%'.format('test'+name, epoch_loss, epoch_acc*100, epoch_acc_5*100))


def train_model(args, model, model_t, dataloaders, criterion, optimizer, 
    logger_name='train_logger', checkpoint_folder='exp',
    start_iter=0, best_acc=0.0, writer=None, ssl_obj=None, scheduler=None):

    ## for self-training
    if model_t is not None:
        model_t.eval()

    is_inception = (args.model=="inception")

    logger = logging.getLogger(logger_name)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())

    print_freq = args.print_freq

    iteration = start_iter
    running_loss = 0.0
    running_loss_cls = 0.0
    running_loss_ssl = 0.0
    running_corrects_1 = 0
    running_corrects_5 = 0

    ####################
    ##### Training #####
    ####################
    for l_data, u_data in zip(dataloaders['l_train'], dataloaders['u_train']):
        iteration += 1 

        model.train()
        
        l_input, target = l_data
        u_input, dummy_target = u_data

        l_input = l_input.to(device).float()
        u_input = u_input.to(device).float()
        dummy_target = dummy_target.to(device).long()
        target = target.to(device).long()

        ## upsample
        if args.input_size != l_input.shape[-1]:
            m = torch.nn.Upsample((args.input_size, args.input_size), mode='bilinear', align_corners=True)
            l_input = m(l_input)
            u_input = m(u_input)
        else:
            m = None
            
        # forward
        with torch.set_grad_enabled(True):                
            if args.alg == 'distill':
                inputs = torch.cat([l_input, u_input], 0)

                ## for self-training
                logit_s = model(inputs, is_feat=False)
                with torch.no_grad():
                    logit_t = model_t(inputs, is_feat=False)
                ssl_loss = ssl_obj(logit_s, logit_t)

                ## classification loss
                target = torch.cat([target, -torch.ones(args.batch_size//2).to(device).long()], 0)
                unlabeled_mask = (target == -1).float()
                outputs = logit_s
                cls_loss = criterion(outputs, target)

                loss = (1.0 - args.alpha) * cls_loss + args.alpha * ssl_loss

                correct_1, correct_5 = compute_correct(outputs[:args.batch_size//2,:], target[:args.batch_size//2], topk=(1, 5))

            elif args.alg != "supervised":
                ## for other semi-supervised methods
                target = torch.cat([target, -torch.ones(args.batch_size//2).to(device).long()], 0)
                unlabeled_mask = (target == -1).float()
                
                inputs = torch.cat([l_input, u_input], 0)
                
                outputs = model(inputs)

                coef = args.consis_coef * math.exp(-5 * (1 - min(iteration/args.warmup, 1))**2)
                writer.add_scalar('train/coef', coef, iteration)
                ssl_loss = ssl_obj(inputs, outputs.detach(), model, unlabeled_mask) * coef

                if args.em > 0:
                    ssl_loss -= args.em * ((outputs.softmax(1) * F.log_softmax(outputs, 1)).sum(1) * unlabeled_mask).mean()
                cls_loss = F.cross_entropy(outputs, target, reduction="none", ignore_index=-1).mean()
                loss = cls_loss + ssl_loss

                correct_1, correct_5 = compute_correct(outputs[:args.batch_size//2,:], target[:args.batch_size//2], topk=(1, 5))
            else:
                ## for supervised baseline
                outputs = model(l_input)
                loss = criterion(outputs, target)

                correct_1, correct_5 = compute_correct(outputs, target, topk=(1, 5))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        # if args.alg == "MT" or args.alg == "ICT":
        #     # parameter update with exponential moving average
        #     ssl_obj.moving_average(model.parameters())

        # statistics
        if args.alg != "supervised":
            running_loss_cls += cls_loss.item()
            running_loss_ssl += ssl_loss.item()
        else:
            running_loss += loss.item()
        
        running_corrects_1 += correct_1.item()
        running_corrects_5 += correct_5.item()

        ## Print training loss/acc ##
        if (iteration+1) % print_freq==0:
            if args.alg != "supervised":
                if args.alg == 'distill':
                    logger.info('{} | Iteration {:d}/{:d} | Cls Loss {:f} | Distillation Loss {:f} | Top1 Acc {:.2f}%'.format( \
                        'train', iteration+1, len(dataloaders['l_train']), running_loss_cls/print_freq, \
                        running_loss_ssl/print_freq, running_corrects_1*100/(print_freq*args.batch_size//2) ))
                else:
                    logger.info('{} | Iteration {:d}/{:d} | Cls Loss {:f} | SSL Loss {:f} | Top1 Acc {:.2f}% | coef {:f}'.format( \
                        'train', iteration+1, len(dataloaders['l_train']), running_loss_cls/print_freq, \
                        running_loss_ssl/print_freq, running_corrects_1*100/(print_freq*args.batch_size//2), coef ))
                writer.add_scalar('train/loss_'+args.alg, running_loss_ssl/print_freq, iteration)
                writer.add_scalar('train/loss_cls', running_loss_cls/print_freq, iteration)
            else:
                logger.info('{} | Iteration {:d}/{:d} | Loss {:f} | Top1 Acc {:.2f}% | Top5 Acc {:.2f}%'.format( \
                    'train', iteration+1, len(dataloaders['l_train']), running_loss/print_freq, \
                    running_corrects_1*100/(print_freq*l_input.size(0)), running_corrects_5*100/(print_freq*l_input.size(0)) ))
                writer.add_scalar('train/loss', running_loss/print_freq, iteration)
                
            writer.add_scalar('train/top1_acc', running_corrects_1*100/(print_freq*l_input.size(0)), iteration)
            writer.add_scalar('train/top5_acc', running_corrects_5*100/(print_freq*l_input.size(0)), iteration)

            running_loss = 0.0
            running_loss_cls = 0.0
            running_loss_ssl = 0.0
            running_corrects_1 = 0
            running_corrects_5 = 0

        ####################
        #### Validation ####
        ####################
        if ((iteration+1) % args.val_freq) == 0 or (iteration+1) == args.num_iter:

            ## Print val loss/acc ##
            model.eval()
            val_loss = 0.0
            val_corrects_1 = 0
            val_corrects_5 = 0
            for i,data in enumerate(dataloaders['val']):
                inputs, target = data
                inputs = inputs.to(device).float()
                target = target.to(device).long()

                ## upsample
                if m is not None:
                    inputs = m(inputs)

                optimizer.zero_grad()
                with torch.set_grad_enabled(False):
                    outputs = model(inputs)
                    loss = criterion(outputs, target)

                    correct_1, correct_5 = compute_correct(outputs, target, topk=(1, 5))
                    val_loss += loss.item()

                val_corrects_1 += correct_1.item()
                val_corrects_5 += correct_5.item()

            num_val = len(dataloaders['val'].dataset)
            logger.info('{} | Iteration {:d}/{:d} | Loss {:f} | Top1 Acc {:.2f}% | Top5 Acc {:.2f}%'.format( 'Val', iteration+1, \
                args.num_iter, val_loss/i, val_corrects_1*100/num_val, val_corrects_5*100/num_val ))

            epoch_acc = val_corrects_1*100/num_val
            writer.add_scalar('val/loss', val_loss/num_val, iteration)
            writer.add_scalar('val/top1_acc', val_corrects_1*100/num_val, iteration)
            writer.add_scalar('val/top5_acc', val_corrects_5*100/num_val, iteration)

            # deep copy the model with best val acc.
            is_best = epoch_acc > best_acc
            if epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            val_acc_history.append(epoch_acc)

            save_checkpoint({
                'iteration': iteration + 1,
                'best_acc': best_acc,
                'model': args.model,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict' : optimizer.state_dict(),
                }, is_best, checkpoint_folder=checkpoint_folder)
        
        ## my setting
        if scheduler is None:
            ## Manually decrease lr if not using scheduler
            if (iteration+1)%args.lr_decay_iter == 0:
                optimizer.param_groups[0]["lr"] *= args.lr_decay_factor
        writer.add_scalar('lr', optimizer.param_groups[0]["lr"], iteration)


    time_elapsed = time.time() - since
    logger.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    logger.info('Best val Acc: {:.2f}%'.format(best_acc))


    ##############
    #### Test ####
    ##############
    optimizer.zero_grad()
    test(model,dataloaders,args,logger,"Last")
    ## Load best model weights
    model.load_state_dict(best_model_wts)
    test(model,dataloaders,args,logger,"Best")

    writer.close()
    return model, val_acc_history