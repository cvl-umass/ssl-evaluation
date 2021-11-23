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

NLLoss = nn.NLLLoss()

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

            correct_1 = compute_correct(outputs, target, topk=(1, ))

        test_corrects_1 += correct_1[0].item()

    if curriculum:
        idx = np.argsort(-np.array(all_scores))
        num_pl = round(len(all_scores)*ratio)
        logger.info('Select {} PL Images'.format(num_pl))
        for i in range(num_pl):
            f.write('%s %d\n'%(all_path[idx[i]].replace('iNat_dump/',''), all_labels[idx[i]]))

    f.close()

    epoch_acc   = test_corrects_1 / len(dataloader.dataset)

    logger.info('{} Top1 Acc: {:.2f}%'.format('Pseudo_label', epoch_acc*100))

def forward_hierarchy(outputs, l_targets, u_outputs, u_targets, args, model, criterion=nn.CrossEntropyLoss()):
    loss = criterion(outputs, l_targets[0])
    correct_1 = compute_correct(outputs, l_targets[0], topk=(1, ))

    _outputs =  F.softmax(outputs, dim=1)

    outputs_g = torch.matmul(_outputs, model.W_s2g)
    outputs_f = torch.matmul(outputs_g, model.W_g2f)
    outputs_o = torch.matmul(outputs_f, model.W_f2o)
    outputs_c = torch.matmul(outputs_o, model.W_o2c)
    outputs_p = torch.matmul(outputs_c, model.W_c2p)
    outputs_k = torch.matmul(outputs_p, model.W_p2k)

    correct_1_p = compute_correct(outputs_p, l_targets[5], topk=(1,))
    correct_1_k = compute_correct(outputs_k, l_targets[6], topk=(1,))
    
    _u_outputs =  F.softmax(u_outputs, dim=1)

    u_outputs_g = torch.matmul(_u_outputs, model.W_s2g)
    u_outputs_f = torch.matmul(u_outputs_g, model.W_g2f)
    u_outputs_o = torch.matmul(u_outputs_f, model.W_f2o)
    u_outputs_c = torch.matmul(u_outputs_o, model.W_o2c)
    u_outputs_p = torch.matmul(u_outputs_c, model.W_c2p)
    u_outputs_k = torch.matmul(u_outputs_p, model.W_p2k)

    u_correct_1_p = compute_correct(u_outputs_p, u_targets[5], topk=(1,))
    u_correct_1_k = compute_correct(u_outputs_k, u_targets[6], topk=(1,))

    if args.level == 'species':
        u_loss_s = NLLoss(torch.log(_u_outputs + 1e-20) , u_targets[0])
        loss += u_loss_s
    elif args.level == 'genus':
        u_loss_g = NLLoss(torch.log(u_outputs_g + 1e-20) , u_targets[1])
        loss += u_loss_g
    elif args.level == 'family':
        u_loss_f = NLLoss(torch.log(u_outputs_f + 1e-20) , u_targets[2])
        loss += u_loss_f
    elif args.level == 'order':
        u_loss_o = NLLoss(torch.log(u_outputs_o + 1e-20) , u_targets[3])
        loss += u_loss_o
    elif args.level == 'class':
        u_loss_c = NLLoss(torch.log(u_outputs_c + 1e-20) , u_targets[4])
        loss += u_loss_c
    elif args.level == 'phylum':
        u_loss_p = NLLoss(torch.log(u_outputs_p + 1e-20) , u_targets[5])
        loss += u_loss_p
    elif args.level == 'kingdom':
        u_loss_k = NLLoss(torch.log(u_outputs_k + 1e-20) , u_targets[6])
        loss += u_loss_k

    return loss, correct_1, correct_1_p, correct_1_k, u_correct_1_p, u_correct_1_k


def test(model, dataloaders, args, logger, name="Best", criterion=nn.CrossEntropyLoss()):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()
    test_corrects_1 = 0
    test_loss = 0
    test_corrects_1_p = 0
    test_corrects_1_k = 0
    for i,data in enumerate(dataloaders['test']):
        inputs, target, l_target_k, l_target_p, l_target_c, l_target_o, l_target_f, l_target_g = data
        inputs = inputs.to(device).float()
        target = target.to(device).long()
        l_target_k = l_target_k.to(device).long()
        l_target_p = l_target_p.to(device).long()
        l_target_c = l_target_c.to(device).long()
        l_target_o = l_target_o.to(device).long()
        l_target_f = l_target_f.to(device).long()
        l_target_g = l_target_g.to(device).long()

        ## upsample
        if args.input_size != inputs.shape[-1]:
            m = torch.nn.Upsample((args.input_size, args.input_size), mode='bilinear', align_corners=True)
            inputs = m(inputs)

        with torch.set_grad_enabled(False):

            feature = model(inputs)
            outputs = model.fc(feature)
            loss = criterion(outputs, target)
            correct_1 = compute_correct(outputs, target, topk=(1, ))

            _outputs =  F.softmax(outputs, dim=1)

            outputs_g = torch.matmul(_outputs, model.W_s2g)
            loss_g = NLLoss(torch.log(outputs_g + 1e-20) , l_target_g)

            outputs_f = torch.matmul(outputs_g, model.W_g2f)
            loss_f = NLLoss(torch.log(outputs_f + 1e-20) , l_target_f)

            outputs_o = torch.matmul(outputs_f, model.W_f2o)
            loss_o = NLLoss(torch.log(outputs_o + 1e-20) , l_target_o)

            outputs_c = torch.matmul(outputs_o, model.W_o2c)
            loss_c = NLLoss(torch.log(outputs_c + 1e-20) , l_target_c)

            outputs_p = torch.matmul(outputs_c, model.W_c2p)
            loss_p = NLLoss(torch.log(outputs_p + 1e-20) , l_target_p)

            outputs_k = torch.matmul(outputs_p, model.W_p2k)
            loss_k = NLLoss(torch.log(outputs_k + 1e-20) , l_target_k)

            correct_1_p = compute_correct(outputs_p, l_target_p, topk=(1,))
            correct_1_k = compute_correct(outputs_k, l_target_k, topk=(1,))

            test_loss += loss.item() + loss_p.item() + loss_k.item()
                
        test_corrects_1 += correct_1[0].item()

    epoch_loss  = test_loss / i
    epoch_acc   = test_corrects_1 / len(dataloaders['test'].dataset)

    logger.info('{} Loss: {:.4f} Top1 Acc: {:.2f}%'.format( \
        'test'+name, epoch_loss, epoch_acc*100))


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
    running_corrects_1_p = 0
    running_corrects_1_k = 0

    ####################
    ##### Training #####
    ####################
    for l_data, u_data in zip(dataloaders['l_train'], dataloaders['u_train']):
        iteration += 1 

        model.train()
        
        l_input, l_target_s, l_target_k, l_target_p, l_target_c, l_target_o, l_target_f, l_target_g = l_data
        u_input, u_target_s, u_target_k, u_target_p, u_target_c, u_target_o, u_target_f, u_target_g = u_data

        l_input = l_input.to(device).float()
        u_input = u_input.to(device).float()

        l_target_s = l_target_s.to(device).long()
        l_target_k = l_target_k.to(device).long()
        l_target_p = l_target_p.to(device).long()
        l_target_c = l_target_k.to(device).long()
        l_target_o = l_target_p.to(device).long()
        l_target_f = l_target_k.to(device).long()
        l_target_g = l_target_p.to(device).long()

        u_target_s = u_target_s.to(device).long()
        u_target_k = u_target_k.to(device).long()
        u_target_p = u_target_p.to(device).long() 
        u_target_c = u_target_c.to(device).long()
        u_target_o = u_target_o.to(device).long() 
        u_target_f = u_target_f.to(device).long()
        u_target_g = u_target_g.to(device).long() 

        l_targets = [l_target_s, l_target_g, l_target_f, l_target_o, l_target_c, l_target_p, l_target_k]
        u_targets = [u_target_s, u_target_g, u_target_f, u_target_o, u_target_c, u_target_p, u_target_k]

        ## upsample
        if args.input_size != l_input.shape[-1]:
            m = torch.nn.Upsample((args.input_size, args.input_size), mode='bilinear', align_corners=True)
            l_input = m(l_input)
            u_input = m(u_input)
        else:
            m = None
            
        # forward
        with torch.set_grad_enabled(True):                
            if args.alg == 'distill_hierarchy':
                ## Distillation + hierarchy supervision
                l_feature = model(l_input)
                l_outputs = model.fc(l_feature)

                u_feature = model(u_input)
                u_outputs = model.fc(u_feature)

                cls_loss, correct_1, correct_1_p, correct_1_k, u_correct_1_p, u_correct_1_k = forward_hierarchy(l_outputs, l_targets, u_outputs, u_targets, args, model)

                ## for self-training
                logit_s = torch.cat([l_outputs, u_outputs], 0)
                with torch.no_grad():
                    # feature_t = model_t(torch.cat([l_input, u_input], 0), is_feat=False)
                    l_feature_t = model_t(l_input, is_feat=False)
                    u_feature_t = model_t(u_input, is_feat=False)
                    feature_t = torch.cat([l_feature_t, u_feature_t], 0)
                    if args.init == 'inat' and args.MoCo is False :
                        logit_t = model_t.module.fc(feature_t)
                    else:
                        logit_t = model_t.fc(feature_t)
                ssl_loss = ssl_obj(logit_s, logit_t)

                loss = (1.0 - args.alpha) * cls_loss + args.alpha * ssl_loss

            elif args.alg == "hierarchy":
                ## Supervised + hierarchical supervision
                l_feature = model(l_input)
                l_outputs = model.fc(l_feature)

                u_feature = model(u_input)
                u_outputs = model.fc(u_feature)

                loss, correct_1, correct_1_p, correct_1_k, u_correct_1_p, u_correct_1_k = forward_hierarchy(l_outputs, l_targets, u_outputs, u_targets, args, model)

            elif args.alg == "PL_hierarchy":
                ## PL + hierarchical supervision
                l_feature = model(l_input)
                l_outputs = model.fc(l_feature)

                u_feature = model(u_input)
                u_outputs = model.fc(u_feature)

                cls_loss, correct_1, correct_1_p, correct_1_k, u_correct_1_p, u_correct_1_k = forward_hierarchy(l_outputs, l_targets, u_outputs, u_targets, args, model)

                target = torch.cat([l_target_s, -torch.ones(args.batch_size//2).to(device).long()], 0)
                unlabeled_mask = (target == -1).float()
                
                inputs = torch.cat([l_input, u_input], 0)
                outputs = torch.cat([l_outputs, u_outputs], 0)
                
                coef = args.consis_coef * math.exp(-5 * (1 - min(iteration/args.warmup, 1))**2)
                writer.add_scalar('train/coef', coef, iteration)
                ssl_loss = ssl_obj(inputs, outputs.detach(), model, unlabeled_mask) * coef

                if args.em > 0:
                    ssl_loss -= args.em * ((outputs.softmax(1) * F.log_softmax(outputs, 1)).sum(1) * unlabeled_mask).mean()
                loss = cls_loss + ssl_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        # statistics
        if args.alg == "hierarchy":
            running_loss += loss.item()
        else:
            running_loss_cls += cls_loss.item()
            running_loss_ssl += ssl_loss.item()
        
        running_corrects_1 += correct_1[0].item()

        running_corrects_1_p += correct_1_p[0].item()
        running_corrects_1_k += correct_1_k[0].item()

        ## Print training loss/acc ##
        if (iteration+1) % print_freq==0:
            if args.alg == "hierarchy":
                logger.info('{} | Iteration {:d}/{:d} | Loss {:f} | Top1 Acc {:.2f}% | Top1 Kingdom Acc {:.2f} | Top1 Phylum Acc {:.2f}%'.format( \
                    'train', iteration+1, len(dataloaders['l_train']), running_loss/print_freq, \
                    running_corrects_1*100/(print_freq*l_input.size(0)), running_corrects_1_k*100/(print_freq*l_input.size(0)), \
                    running_corrects_1_p*100/(print_freq*l_input.size(0)) ))
                writer.add_scalar('train/loss', running_loss/print_freq, iteration)
            elif args.alg == 'distill_hierarchy':
                logger.info('{} | Iteration {:d}/{:d} | Cls Loss {:f} | Distillation Loss {:f} | Top1 Acc {:.2f}% | Top1 Kingdom Acc {:.2f} | Top1 Phylum Acc {:.2f}%'.format( \
                        'train', iteration+1, len(dataloaders['l_train']), running_loss_cls/print_freq, \
                        running_loss_ssl/print_freq, running_corrects_1*100/(print_freq*args.batch_size//2), \
                        running_corrects_1_k*100/(print_freq*l_input.size(0)), running_corrects_1_p*100/(print_freq*l_input.size(0)) ))
            elif args.alg == 'PL_hierarchy':
                logger.info('{} | Iteration {:d}/{:d} | Cls Loss {:f} | SSL Loss {:f} | Top1 Acc {:.2f}% | Top1 Kingdom Acc {:.2f} | Top1 Phylum Acc {:.2f}%'.format( \
                        'train', iteration+1, len(dataloaders['l_train']), running_loss_cls/print_freq, \
                        running_loss_ssl/print_freq, running_corrects_1*100/(print_freq*args.batch_size//2), \
                        running_corrects_1_k*100/(print_freq*l_input.size(0)), running_corrects_1_p*100/(print_freq*l_input.size(0)) ))
                
            writer.add_scalar('train/top1_acc', running_corrects_1*100/(print_freq*l_input.size(0)), iteration)

            running_loss = 0.0
            running_loss_cls = 0.0
            running_loss_ssl = 0.0
            running_corrects_1 = 0
            running_corrects_1_p = 0
            running_corrects_1_k = 0

        ####################
        #### Validation ####
        ####################
        if ((iteration+1) % args.val_freq) == 0 or (iteration+1) == args.num_iter:

            ## Print val loss/acc ##
            model.eval()
            val_loss = 0.0
            val_corrects_1 = 0
            val_corrects_1_p = 0
            val_corrects_1_k = 0
            for i,data in enumerate(dataloaders['val']):
                inputs, target, l_target_k, l_target_p, l_target_c, l_target_o, l_target_f, l_target_g = data
                inputs = inputs.to(device).float()
                target = target.to(device).long()
                l_target_k = l_target_k.to(device).long()
                l_target_p = l_target_p.to(device).long()
                l_target_c = l_target_c.to(device).long()
                l_target_o = l_target_o.to(device).long()
                l_target_f = l_target_f.to(device).long()
                l_target_g = l_target_g.to(device).long()

                ## upsample
                if m is not None:
                    inputs = m(inputs)

                optimizer.zero_grad()
                with torch.set_grad_enabled(False):

                    feature = model(inputs)
                    outputs = model.fc(feature)
                    loss = criterion(outputs, target)
                    correct_1 = compute_correct(outputs, target, topk=(1, ))

                    _outputs =  F.softmax(outputs, dim=1)

                    outputs_g = torch.matmul(_outputs, model.W_s2g)
                    loss_g = NLLoss(torch.log(outputs_g + 1e-20) , l_target_g)

                    outputs_f = torch.matmul(outputs_g, model.W_g2f)
                    loss_f = NLLoss(torch.log(outputs_f + 1e-20) , l_target_f)

                    outputs_o = torch.matmul(outputs_f, model.W_f2o)
                    loss_o = NLLoss(torch.log(outputs_o + 1e-20) , l_target_o)

                    outputs_c = torch.matmul(outputs_o, model.W_o2c)
                    loss_c = NLLoss(torch.log(outputs_c + 1e-20) , l_target_c)

                    outputs_p = torch.matmul(outputs_c, model.W_c2p)
                    loss_p = NLLoss(torch.log(outputs_p + 1e-20) , l_target_p)

                    outputs_k = torch.matmul(outputs_p, model.W_p2k)
                    loss_k = NLLoss(torch.log(outputs_k + 1e-20) , l_target_k)

                    correct_1_p = compute_correct(outputs_p, l_target_p, topk=(1,))
                    correct_1_k = compute_correct(outputs_k, l_target_k, topk=(1,))

                val_corrects_1 += correct_1[0].item()

                val_corrects_1_p += correct_1_p[0].item()
                val_corrects_1_k += correct_1_k[0].item()

            num_val = len(dataloaders['val'].dataset)
            logger.info('{} | Iteration {:d}/{:d} | Loss {:f} | Top1 Acc {:.2f}% | Top1 Kingdom Acc {:.2f} | Top1 Phylum Acc {:.2f}%'.format( 'Val', iteration+1, \
                args.num_iter, val_loss/i, val_corrects_1*100/num_val, val_corrects_1_k*100/num_val, val_corrects_1_p*100/num_val ))
            writer.add_scalar('val/top1_kingdom_acc', val_corrects_1_p*100/num_val, iteration)
            writer.add_scalar('val/top1_phylum_acc', val_corrects_1_k*100/num_val, iteration)

            epoch_acc = val_corrects_1*100/num_val
            writer.add_scalar('val/loss', val_loss/num_val, iteration)
            writer.add_scalar('val/top1_acc', val_corrects_1*100/num_val, iteration)

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