import argparse
import os
from os import path
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import resnet

from torch.utils.data import Dataset, DataLoader
import util
from warnings import simplefilter
from GradualWarmupScheduler import *

import autograd_hacks as ah

print(torch.cuda.current_device())
print(torch.cuda.device(0))
print(torch.cuda.get_device_name(0))
print(torch.cuda.is_available())

from pruner.GraSP import GraSP


# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
np.seterr(all='ignore')

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152

model_names = sorted(name for name in resnet.__dict__
    if name.islower() and not name.startswith("__")
                     and name.startswith("resnet")
                     and callable(resnet.__dict__[name]))

print(model_names)

parser = argparse.ArgumentParser(description='Propert ResNets for CIFAR10 in pytorch')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet20', #'resnet56', #
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) +
                         ' (default: resnet32)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', '-m', type=float, metavar='M', default=0.9,
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--print-freq', '-p', default=100, type=int,
                    metavar='N', help='print frequency (default: 20)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--half', dest='half', action='store_true',
                    help='use half-precision(16-bit) ')
parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='save_temp', type=str)
parser.add_argument('--save-every', dest='save_every',
                    help='Saves checkpoints at every specified number of epochs',
                    type=int, default=300)  # default=10)
parser.add_argument('--gpu', default='7', type=str, help='The GPU to be used')
parser.add_argument('--greedy', '-g', dest='greedy', action='store_true', default=False, help='greedy ordering')
parser.add_argument('--subset_size', '-s', dest='subset_size', type=float, help='size of the subset', default=1.0)
parser.add_argument('--random_subset_size', '-rs', type=float, help='size of the subset', default=1.0)
parser.add_argument('--st_grd', '-stg', type=float, help='stochastic greedy', default=0)
parser.add_argument('--smtk', type=int, help='smtk', default=1)
parser.add_argument('--ig', type=str, help='ig method', default='sgd', choices=['sgd, adam, adagrad'])
parser.add_argument('--lr_schedule', '-lrs', type=str, help='learning rate schedule', default='mile',
                    choices=['mile', 'exp', 'cnt', 'step', 'cosine'])
parser.add_argument('--gamma', type=float, default=-1, help='learning rate decay parameter')
parser.add_argument('--lag', type=int, help='update lags', default=1)
parser.add_argument('--runs', type=int, help='num runs', default=1)
parser.add_argument('--warm', '-w', dest='warm_start', action='store_true', help='warm start learning rate ')
parser.add_argument('--cluster_features', '-cf', dest='cluster_features', action='store_true', help='cluster_features')
parser.add_argument('--cluster_all', '-ca', dest='cluster_all', action='store_true', help='cluster_all')
parser.add_argument('--start-subset', '-st', default=0, type=int, metavar='N', help='start subset selection')
parser.add_argument('--save_subset', dest='save_subset', action='store_true', help='save_subset')
parser.add_argument('--searching', dest='searching', action='store_true', help='search EB or not')
parser.add_argument('--imp_type', type=int, help='implementation type', default=0)
parser.add_argument('--EB_path', type=str, help='path to EB checkpoint', default='None')
parser.add_argument('--prune_percent', '-ppct', type=float, help='percent of weights kept after pruning', default=0.5)

TRAIN_NUM = 60000
CLASS_NUM = 10


def main(subset_size=.1, greedy=0):

    global args, best_prec1
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    print(f'--------- subset_size: {subset_size}, method: {args.ig}, moment: {args.momentum}, '
          f'lr_schedule: {args.lr_schedule}, greedy: {greedy}, stoch: {args.st_grd}, rs: {args.random_subset_size} ---------------')

    # Check the save_dir exists or not
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    #model = torch.nn.DataParallel(resnet.__dict__[args.arch]())
    #model.cuda()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    normalize = transforms.Normalize(mean=[0.5],
                                     std=[0.5])

    train_loader__ = torch.utils.data.DataLoader(
        datasets.MNIST(root='./data', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=True),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    class IndexedDataset(Dataset):
        def __init__(self):
            self.MNIST = datasets.MNIST(root='./data', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=True)

        def __getitem__(self, index):
            data, target = self.MNIST[index]
            # Your transformations here (or set it in MNIST)
            return data, target, index

        def __len__(self):
            return len(self.MNIST)

    indexed_dataset = IndexedDataset()
    indexed_loader = DataLoader(
        indexed_dataset,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root='./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    train_val_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root='./data', train=True, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    train_criterion = nn.CrossEntropyLoss(reduction='none').cuda()  # (Note)
    val_criterion = nn.CrossEntropyLoss().cuda()

    if args.half:
        model.half()
        train_criterion.half()
        val_criterion.half()

    runs, best_run, best_run_loss, best_loss = args.runs, 0, 0, 1e10
    epochs = args.epochs
    train_loss, test_loss = np.zeros((runs, epochs)), np.zeros((runs, epochs))
    train_acc, test_acc = np.zeros((runs, epochs)), np.zeros((runs, epochs))
    train_time, data_time = np.zeros((runs, epochs)), np.zeros((runs, epochs))
    grd_time, sim_time = np.zeros((runs, epochs)), np.zeros((runs, epochs))
    pass_time = np.zeros((runs,epochs))
    not_selected = np.zeros((runs, epochs))
    best_bs, best_gs = np.zeros(runs), np.zeros(runs)
    times_selected = np.zeros((runs, len(indexed_loader.dataset)))

    if args.save_subset:
        B = int(args.subset_size * TRAIN_NUM)
        selected_ndx = np.zeros((runs, epochs, B))
        selected_wgt = np.zeros((runs, epochs, B))

    if (args.lr_schedule == 'mile' or args.lr_schedule == 'cosine') and args.gamma == -1:
        lr = args.lr
        b = 0.1
    else:
        lr = args.lr
        b = args.gamma

    print(f'lr schedule: {args.lr_schedule}, epochs: {args.epochs}')
    print(f'lr: {lr}, b: {b}')

    for run in range(runs):
        best_prec1_all, best_loss_all, prec1 = 0, 1e10, 0

        if subset_size < 1:
            # initialize a random subset
            B = int(args.random_subset_size * TRAIN_NUM)
            order = np.arange(0, TRAIN_NUM)
            np.random.shuffle(order)
            order = order[:B]
            print(f'Random init subset size: {args.random_subset_size}% = {B}')

        #model = torch.nn.DataParallel(resnet.__dict__[args.arch]())
        model = Net()
        model.cuda()

        #get cfg_mask/GraSP mask
        if not args.searching:
            prune_percent = args.prune_percent
            if args.imp_type == 2:
                EB_checkpoint = args.EB_path
                cfg_mask, cfg = get_cfg_mask(EB_checkpoint, prune_percent)
            elif args.imp_type in [3,4,5]:
                print("Doing grasp!")
                classes = {
                     'cifar10': 10,
                     'cifar100': 100,
                     'mnist': 10,
                     'tiny_imagenet': 200
                }
                dataset = 'mnist'
                batch_size = 512
                ratio = 1 - args.prune_percent
                sample_per_class = 25
                num_iters = 1
                cfg_mask = GraSP(model, ratio, indexed_loader, 'cuda',
                          num_classes=classes[dataset],
                          samples_per_class=sample_per_class,
                          num_iters=num_iters)

        best_prec1, best_loss = 0, 1e10

        if args.ig == 'adam':
            print('using adam')
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=args.weight_decay)
        elif args.ig == 'adagrad':
            optimizer = torch.optim.Adagrad(model.parameters(), lr, weight_decay=args.weight_decay)
        else:
            optimizer = torch.optim.SGD(model.parameters(),
                                        lr,
                                        momentum=args.momentum,
                                        weight_decay=args.weight_decay)

        if args.lr_schedule == 'exp':
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=b, last_epoch=args.start_epoch - 1)
        elif args.lr_schedule == 'step':
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=b)
        elif args.lr_schedule == 'mile':
            milestones = np.array([100, 150])
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=milestones, last_epoch=args.start_epoch - 1, gamma=b)
        elif args.lr_schedule == 'cosine':
            # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
        else:  # constant lr
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.epochs, gamma=1.0)

        if args.warm_start:
            print('Warm start learning rate')
            lr_scheduler_f = GradualWarmupScheduler(optimizer, 1.0, 20, lr_scheduler)
        else:
            print('No Warm start')
            lr_scheduler_f = lr_scheduler

        if args.arch in ['resnet1202', 'resnet110']:
            # for resnet1202 original paper uses lr=0.01 for first 400 minibatches for warm-up
            # then switch back. In this setup it will correspond for first epoch.
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr*0.1

        if args.evaluate:
            validate(val_loader, model, val_criterion)
            return

        flag_1 = True
        flag_30 = True
        flag_50 = True
        flag_70 = True
        flag_95 = True
        early_bird_1 = EarlyBird(0.01)
        early_bird_30 = EarlyBird(0.3)
        early_bird_50 = EarlyBird(0.5)
        early_bird_70 = EarlyBird(0.7)
        early_bird_95 = EarlyBird(0.95)

        for epoch in range(args.start_epoch, args.epochs):

            # train for one epoch
            print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))

            #print(masks)
            #############################
            weight = None
            if subset_size >= 1 or epoch < args.start_subset:
                print('Training on all the data')
                train_loader = indexed_loader

            elif subset_size < 1 and \
                    (epoch % (args.lag + args.start_subset) == 0 or epoch == args.start_subset):
                B = int(subset_size * TRAIN_NUM)
                if args.imp_type == 0:
                    # order = np.arange(0, TRAIN_NUM)
                    np.random.shuffle(order)
                    subset = order[:B]
                    weights = np.zeros(len(indexed_loader.dataset))
                    weights[subset] = np.ones(B)
                    print(f'Selecting {B} element from the pre-selected random subset of size: {len(subset)}')
                else:  # Note: warm start
                    if args.cluster_features:
                        print(f'Selecting {B} elements greedily from features')
                        data = datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomCrop(32, 4),
                            transforms.ToTensor(),
                            normalize,
                        ]), download=True)
                        preds, labels = np.reshape(data.data, (len(data.targets), -1)), data.targets
                    elif args.imp_type in [2,3]: #Use EB or GraSP
                        mask_type = "EB" if args.imp_type == 2 else "grasp"
                        print(f'Selecting {B} elements greedily from {mask_type} gradients')
                        preds, labels, batch_time = get_grads(indexed_loader, model, val_criterion, cfg_mask, mask_type) #still use name 'preds' for consistency 
                        pass_time[run,epoch] = batch_time
                    elif args.imp_type == 1:
                        print(f'Selecting {B} elements greedily from predictions')
                        preds, labels, get_pred_time = predictions(indexed_loader, model)
                        preds -= np.eye(CLASS_NUM)[labels]
                        pass_time[run, epoch] = (get_pred_time)
                    elif args.imp_type in [4,5]:
                        print(f'Selecting {B} elements greedily from predictions + second to last layer gradient')
                        preds, labels, get_pred_time = predictions(indexed_loader, model)
                        preds -= np.eye(CLASS_NUM)[labels]
                        pass_time[run, epoch] += (get_pred_time)
                        grasp1 = args.imp_type == 5 
                        mask_type = "grasp1" if grasp1 else "final"
                        preds2, labels2, get_pred2_time = get_grads(indexed_loader, model, val_criterion, cfg_mask, mask_type = mask_type)
                        preds = np.concatenate((preds, preds2),axis=1)
                        pass_time[run, epoch] += (get_pred2_time)



                    fl_labels = np.zeros(np.shape(labels), dtype=int) if args.cluster_all else labels
                    subset, subset_weight, _, _, ordering_time, similarity_time = util.get_orders_and_weights(
                        B, preds, 'euclidean', smtk=args.smtk, no=0, y=fl_labels, stoch_greedy=args.st_grd,
                        equal_num=True)

                    weights = np.zeros(len(indexed_loader.dataset))
                    # weights[subset] = np.ones(len(subset))
                    subset_weight = subset_weight / np.sum(subset_weight) * len(subset_weight)
                    if args.save_subset:
                        selected_ndx[run, epoch], selected_wgt[run, epoch] = subset, subset_weight

                    weights[subset] = subset_weight
                    weight = torch.from_numpy(weights).float().cuda()
                    # weight = torch.tensor(weights).cuda()
                    # np.random.shuffle(subset)
                    print(f'FL time: {ordering_time:.3f}, Sim time: {similarity_time:.3f}')
                    grd_time[run, epoch], sim_time[run, epoch] = ordering_time, similarity_time

                times_selected[run][subset] += 1
                print(f'{np.sum(times_selected[run] == 0) / len(times_selected[run]) * 100:.3f} % not selected yet')
                not_selected[run, epoch] = np.sum(times_selected[run] == 0) / len(times_selected[run]) * 100
                indexed_subset = torch.utils.data.Subset(indexed_dataset, indices=subset)
                train_loader = DataLoader(
                    indexed_subset,
                    batch_size=args.batch_size, shuffle=True,
                    num_workers=args.workers, pin_memory=True)
            else:
                print('Using the previous subset')
                not_selected[run, epoch] = not_selected[run, epoch - 1]
                print(f'{not_selected[run, epoch]:.3f} % not selected yet')
                #############################

            data_time[run, epoch], train_time[run, epoch] = train(
                train_loader, model, train_criterion, optimizer, epoch, weight)

            lr_scheduler_f.step()

            # evaluate on validation set
            prec1, loss = validate(val_loader, model, val_criterion)

            # remember best prec@1 and save checkpoint
            is_best = prec1 > best_prec1
            # best_run = run if is_best else best_run
            best_prec1 = max(prec1, best_prec1)
            if best_prec1 > best_prec1_all:
                best_gs[run], best_bs[run] = lr, b
                best_prec1_all = best_prec1
            test_acc[run, epoch], test_loss[run, epoch] = prec1, loss

            ta, tl = validate(train_val_loader, model, val_criterion)
            # best_run_loss = run if tl < best_loss else best_run_loss
            best_loss = min(tl, best_loss)
            best_loss_all = min(best_loss_all, best_loss)
            train_acc[run, epoch], train_loss[run, epoch] = ta, tl

            if epoch > 0 and epoch % args.save_every == 0:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_prec1': best_prec1,
                }, is_best, filename=os.path.join(args.save_dir, 'checkpoint.th'))

            # save_checkpoint({
            # 'state_dict': model.state_dict(),
            # 'best_prec1': best_prec1,
            # }, is_best, filename=os.path.join(args.save_dir, 'model.th'))

            print(f'run: {run}, subset_size: {subset_size}, epoch: {epoch}, prec1: {prec1}, loss: {tl:.3f}, '
                  f'g: {lr:.3f}, b: {b:.3f}, '
                  f'best_prec1_gb: {best_prec1}, best_loss_gb: {best_loss:.3f}, best_run: {best_run};  '
                  f'best_prec_all: {best_prec1_all}, best_loss_all: {best_loss_all:.3f}, '
                  f'best_g: {best_gs[run]:.3f}, best_b: {best_bs[run]:.3f}, '
                  f'not selected %:{not_selected[run][epoch]}')

            grd = 'grd_w' if args.greedy else f'rand_rsize_{args.random_subset_size}'
            grd += f'_st_{args.st_grd}' if args.st_grd > 0 else ''
            grd += f'_warm' if args.warm_start > 0 else ''
            grd += f'_feature' if args.cluster_features else ''
            grd += f'_ca' if args.cluster_all else ''
            folder = args.save_dir + '/'
            subfolders = ['random',"pred","EB_grad","GraSP_grad","pred_first_last","pred_last"]
            if greedy == 0:
                folder += subfolders[0]
            else:
                folder += subfolders[args.imp_type]
            if not path.isdir(folder):
                os.mkdir(folder)

            if args.save_subset:
                print(
                    f'Saving the results to {folder}_{args.ig}_moment_{args.momentum}_{args.arch}_{subset_size}'
                    f'_{grd}_{args.lr_schedule}_start_{args.start_subset}_lag_{args.lag}_subset')

                np.savez(f'{folder}_{args.ig}_moment_{args.momentum}_{args.arch}_{subset_size}'
                         f'_{grd}_{args.lr_schedule}_start_{args.start_subset}_lag_{args.lag}_prune_{args.prune_percent}_subset',
                         train_loss=train_loss, test_acc=test_acc, train_acc=train_acc, test_loss=test_loss,
                         data_time=data_time, train_time=train_time, grd_time=grd_time, sim_time=sim_time,pass_time = pass_time,
                         best_g=best_gs, best_b=best_bs, not_selected=not_selected, times_selected=times_selected,
                         subset=selected_ndx, weights=selected_wgt)
            else:
                print(
                    f'Saving the results to {folder}_{args.ig}_moment_{args.momentum}_{args.arch}_{subset_size}'
                    f'_{grd}_{args.lr_schedule}_start_{args.start_subset}_lag_{args.lag}')

                np.savez(f'{folder}_{args.ig}_moment_{args.momentum}_{args.arch}_{subset_size}'
                         f'_{grd}_{args.lr_schedule}_start_{args.start_subset}_lag_{args.lag}',
                         train_loss=train_loss, test_acc=test_acc, train_acc=train_acc, test_loss=test_loss,
                         data_time=data_time, train_time=train_time, grd_time=grd_time, sim_time=sim_time,pass_time = pass_time,
                         best_g=best_gs, best_b=best_bs, not_selected=not_selected,
                         times_selected=times_selected)

            if args.searching:
                if early_bird_1.early_bird_emerge(model):
                    print("[early_bird_1] Find EB!!!!!!!!!, epoch: "+str(epoch))
                    if flag_1:
                        save_checkpoint_EB({
                        'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'best_prec1': best_loss_all,
                        'optimizer': optimizer.state_dict(),
                        }, is_best, 'EB-1-'+str(epoch+1), filepath=args.save_dir)
                        flag_1 = False
                if early_bird_30.early_bird_emerge(model):
                    print("[early_bird_30] Find EB!!!!!!!!!, epoch: "+str(epoch))
                    if flag_30:
                        save_checkpoint_EB({
                        'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'best_prec1': best_loss_all,
                        'optimizer': optimizer.state_dict(),
                        }, is_best, 'EB-30-'+str(epoch+1), filepath=args.save_dir)
                        flag_30 = False
                if early_bird_50.early_bird_emerge(model):
                    print("[early_bird_50] Find EB!!!!!!!!!, epoch: "+str(epoch))
                    if flag_50:
                        save_checkpoint_EB({
                        'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'best_prec1': best_loss_all,
                        'optimizer': optimizer.state_dict(),
                        }, is_best, 'EB-50-'+str(epoch+1), filepath=args.save_dir)
                        flag_50 = False
                if early_bird_70.early_bird_emerge(model):
                     print("[early_bird_70] Find EB!!!!!!!!!, epoch: "+str(epoch))
                     if flag_70:
                         save_checkpoint_EB({
                         'epoch': epoch + 1,
                         'state_dict': model.state_dict(),
                         'best_prec1': best_loss_all,
                         'optimizer': optimizer.state_dict(),
                         }, is_best, 'EB-70-'+str(epoch+1), filepath=args.save_dir)
                         flag_70 = False 
                if early_bird_95.early_bird_emerge(model):
                     print("[early_bird_95] Find EB!!!!!!!!!, epoch: "+str(epoch))
                     if flag_95:
                         save_checkpoint_EB({
                         'epoch': epoch + 1,
                         'state_dict': model.state_dict(),
                         'best_prec1': best_loss_all,
                         'optimizer': optimizer.state_dict(),
                         }, is_best, 'EB-95-'+str(epoch+1), filepath=args.save_dir)
                         flag_95 = False 

    print(np.max(test_acc, 1), np.mean(np.max(test_acc, 1)),
          np.min(not_selected, 1), np.mean(np.min(not_selected, 1)))


def train(train_loader, model, criterion, optimizer, epoch, weight=None):
    """
        Run one train epoch
    """
    if weight is None:
        weight = torch.ones(TRAIN_NUM).cuda()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target, idx) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda()
        input_var = input.cuda()
        target_var = target
        if args.half:
            input_var = input_var.half()

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)
        loss = (loss * weight[idx.long()]).mean()  # (Note)

        # compute gradient and do SGD step
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # if i % args.print_freq == 0:
        #     print('Epoch: [{0}][{1}/{2}]\t'
        #           'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        #           'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
        #           'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
        #           'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
        #               epoch, i, len(train_loader), batch_time=batch_time,
        #               data_time=data_time, loss=losses, top1=top1))
    return data_time.sum, batch_time.sum


def validate(val_loader, model, criterion):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda()
            input_var = input.cuda()
            target_var = target.cuda()

            if args.half:
                input_var = input_var.half()

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # if i % args.print_freq == 0:
            #     print('Test: [{0}/{1}]\t'
            #           'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            #           'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            #           'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
            #               i, len(val_loader), batch_time=batch_time, loss=losses,
            #               top1=top1))

    print(' * Prec@1 {top1.avg:.3f}' .format(top1=top1))

    return top1.avg, losses.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def get_grads(loader, model, criterion, cfg_mask=None, mask_type="EB"):
    """
    Get gradients per example
    """
    batch_time = AverageMeter()

    #add gradient hooks
    ah.add_hooks(model)
    total_weights = 0
    if cfg_mask is not None and mask_type == "EB":
        total_weights = sum([sum(mask) for mask in cfg_mask])
        total_weights = int(total_weights.cpu().numpy())
    elif mask_type == "grasp":
        for layer in model.modules():
            if layer in cfg_mask.keys():
                #print(sm.shape, cfg_mask[layer].shape)
                total_weights += torch.sum(cfg_mask[layer])
        total_weights = int(total_weights.cpu().numpy())
    elif mask_type == "grasp1":
        first_layer_found = False
        for layer in model.modules():
            if layer in cfg_mask.keys() and not first_layer_found:
                #print(sm.shape, cfg_mask[layer].shape)
                total_weights += torch.sum(cfg_mask[layer])
                cfg_mask = {layer:cfg_mask[layer]}
                first_layer_found = True
        total_weights = int(total_weights.cpu().numpy())
    elif mask_type == "final":
        cfg_mask = []
        layer_count = 0
        for layer in model.modules():
            if ah._layer_type(layer) in ['Conv2d','Linear', "BatchNorm2d"]:
                cfg_mask.append(np.ones(layer.weight.size()))
                layer_count += 1
        cfg_mask = cfg_mask[-2]
        total_weights = int(np.sum(cfg_mask))
        target_layer = layer_count - 1
    elif mask_type == 'first':
        cfg_mask = []
        layer_count = 0
        for layer in model.modules():
            if ah._layer_type(layer) in ['Conv2d','Linear', "BatchNorm2d"]:
                cfg_mask.append(np.ones(layer.weight.size()))
                layer_count += 1
        cfg_mask = cfg_mask[0]
        total_weights = int(sum([mask.shape[0] for mask in cfg_mask]))
        target_layer = [0,layer_count - 1]
    else:
        for layer in model.modules():
            if ah._layer_type(layer) == "Conv2d":
                total_weights += layer.weight.shape[0]
    print("Total weights: ", total_weights)

    grads = torch.zeros(TRAIN_NUM, total_weights).cuda()
    labels = torch.zeros(TRAIN_NUM, dtype=torch.int)
    end = time.time()
    for i, (input, target, idx) in enumerate(loader):
        #print(i)
        input_var = input.cuda()
        target_var = target.cuda()

        if args.half:
            input_var = input_var.half()

        output = model(input_var)
        labels[idx] = target.int()
        labels_var = labels.cuda()
        loss = criterion(output, target_var).mean()
        
        ah.clear_backprops(model)
        loss.backward()

        ah.compute_grad1(model)

        current_ind = 0
        end_ind = 0
        current_layer = 1
        if cfg_mask is not None and mask_type == "EB":
            layer_id_in_cfg = 0
            end_mask = cfg_mask[layer_id_in_cfg]
        for m0 in model.modules():
            if cfg_mask is None:
                if not ah._layer_type(m0) is "BatchNorm2d":
                    continue
                grad1 = m0.weight.grad1
                #print(m0.weight.grad.size())
                #print(grad1.shape)
                grad1 = torch.mean(grad1,tuple(range(2,len(grad1.shape))))
                #print(k," th layer",grad1.shape)
                end_ind += m0.weight.size()[0]
                grads[idx,current_ind:end_ind] = grad1
                current_ind += m0.weight.size()[0]
            elif mask_type == "EB":
                if isinstance(m0, nn.Conv2d):
                    if torch.sum(end_mask) == 0:
                        continue
                    idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
                    if idx1.size == 1:
                        idx1 = np.resize(idx1, (1,))
                    grad1 = m0.weight.grad1[:, idx1.tolist(), :, :, :].clone()
                    grad1 = torch.mean(grad1,(2,3,4))
                    end_ind += int(torch.sum(end_mask))
                    grads[idx,current_ind:end_ind] = grad1
                    current_ind += int(torch.sum(end_mask))
                    layer_id_in_cfg += 1
                    if layer_id_in_cfg < len(cfg_mask):
                        end_mask = cfg_mask[layer_id_in_cfg]
            elif mask_type == "grasp" or mask_type == "grasp1":
                if m0 not in cfg_mask.keys():
                    continue
                end_mask = cfg_mask[m0].bool().cpu().numpy()
                if ah._layer_type(m0) in ah._supported_layers:
                    if np.sum(end_mask) == 0:
                        continue
                    idx1 = np.array([end_mask] * len(idx))
                    grad1 = m0.weight.grad1.clone().cpu().numpy()
                    grad1 = grad1[idx1].reshape(len(idx),-1) #Reshape so nrow = batch size.
                    if len(grad1.shape) > 2:
                        grad1 = torch.mean(grad1,(2,3,4))
                    end_ind += int(np.sum(end_mask))
                    grads[idx,current_ind:end_ind] = torch.tensor(grad1).cuda()
                    current_ind += int(np.sum(end_mask))
            elif mask_type == "final":
                if current_layer == target_layer:
                    grad1 = m0.weight.grad1.clone()
                    grad1 = torch.mean(grad1,(2,3,4))
                    end_ind += int(grad1.shape[1])
                    grads[idx,current_ind:end_ind] = torch.tensor(grad1)
                    current_ind += int(grad1.shape[1])
                current_layer += 1
            elif mask_type == "final2":
                if current_layer in target_layer:
                    grad1 = m0.weight.grad1.clone()
                    grad1 = torch.mean(grad1,(2,3,4))
                    end_ind += int(grad1.shape[1])
                    grads[idx,current_ind:end_ind] = torch.tensor(grad1)
                    current_ind += int(grad1.shape[1])
                current_layer += 1 

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    #Remove handle after use so it do not accumulate during the actual training section.
    ah.clear_backprops(model)
    if not hasattr(model, 'autograd_hacks_hooks'):
        print("Warning, asked to remove hooks, but no hooks found")
    else:
        for handle in model.autograd_hacks_hooks:
            handle.remove()
        del model.autograd_hacks_hooks
    if hasattr(model, 'autograd_hacks_hooks'):
            print("Still has hook after cleaning!")

    return grads.cpu().numpy(), labels.cpu().data.numpy(), batch_time.sum

def get_cfg_mask(checkpoint_path, percent=0.01,):

    model = Net()
    model.cuda()

    if os.path.isfile(checkpoint_path):
        print("=> loading checkpoint '{}'".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
    else:
        print("=> no checkpoint found at '{}'".format(checkpoint_path))
        exit()

    total = 0
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            total += m.weight.data.shape[0]

    bn = torch.zeros(total)
    index = 0
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            size = m.weight.data.shape[0]
            bn[index:(index+size)] = m.weight.data.abs().clone()
            index += size

    p_flops = 0
    y, i = torch.sort(bn)
    p_flops += total * np.log2(total) * 3
    thre_index = int(total * percent)
    thre = y[thre_index]

    pruned = 0
    cfg = []
    cfg_mask = []
    for k, m in enumerate(model.modules()):
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            weight_copy = m.weight.data.abs().clone()
            mask = weight_copy.gt(thre.cuda()).float().cuda()
            pruned = pruned + mask.shape[0] - torch.sum(mask)
            m.weight.data.mul_(mask)
            m.bias.data.mul_(mask)
            if int(torch.sum(mask)) > 0:
                cfg.append(int(torch.sum(mask)))
            cfg_mask.append(mask.clone())
            print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
                format(k, mask.shape[0], int(torch.sum(mask))))
        elif isinstance(m, nn.MaxPool2d):
            cfg.append('M')

    return cfg_mask, cfg

def get_connection_mask(loader, model,criterion, percent=0.01,):

    model.eval()

    total = 0
    for m in model.modules():
        if isinstance(m,nn.Conv2d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
            sp = m.weight.data.shape
            sz = sp[0]
            if len(sp) > 1:
               sz = sz * sp[1]
            total += sz
            c = torch.ones(m.weight.data.shape, requires_grad=True).cuda()
            setattr(m, "c", c)
            m.weight.data = c * m.weight.data  

    print(total)      

    preds = torch.zeros(TRAIN_NUM, CLASS_NUM).cuda()
    labels = torch.zeros(TRAIN_NUM, dtype=torch.int)
    end = time.time()
    for i, (input, target, idx) in enumerate(loader):
        if not i == 0:
            continue
        input_var = input.cuda()
        target_var = target.cuda()

        if args.half:
            input_var = input_var.half()

        output = model(input_var)
        labels[idx] = target.int()
        labels_var = labels.cuda()
        loss = criterion(output, target_var).mean()
        
        loss.backward()

    
    bn = torch.zeros(total)
    index = 0
    for m in model.modules():
        if isinstance(m,nn.Conv2d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
            sp = m.weight.data.shape
            size = sp[0]
            if len(sp) > 1:
               size = size * sp[1]
            grad = tf.gradients(loss.detach().cpu(),m.c.data.cpu().clone())
            mean_range = list(range(2,len(sp)))
            print(grad)
            bn[index:(index+size)] = torch.mean(grad,mean_range)
            index += size

    p_flops = 0
    y, i = torch.sort(bn)
    # comparsion and permutation (sort process)
    p_flops += total * np.log2(total) * 3
    thre_index = int(total * percent)
    thre_index = 10
    thre = y[thre_index]

    pruned = 0
    cfg = []
    cfg_mask = []
    for k, m in enumerate(model.modules()):
        if not isinstance(m, nn.MaxPool2d):
            c_grad = tf.gradients(loss, m.c.data.clone())
            mean_range = list(range(2,len(sp)))
            c_grad = torch.means(c_grad, mean_range)
            mask = c_grad.gt(thre.cuda()).float().cuda()
            pruned = pruned + mask.shape[0] - torch.sum(mask)
            m.c.data.mul_(mask)
            if int(torch.sum(mask)) > 0:
                cfg.append(int(torch.sum(mask)))
            cfg_mask.append(mask.clone())
            print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
                format(k, mask.shape[0], int(torch.sum(mask))))
        elif isinstance(m, nn.MaxPool2d):
            cfg.append('M')

    return cfg_mask, cfg


def predictions(loader, model):
    """
    Get predictions
    """
    batch_time = AverageMeter()

    # switch to evaluate mode
    model.eval()

    preds = torch.zeros(TRAIN_NUM, CLASS_NUM).cuda()
    labels = torch.zeros(TRAIN_NUM, dtype=torch.int)
    end = time.time()
    with torch.no_grad():
        for i, (input, target, idx) in enumerate(loader):
            input_var = input.cuda()

            if args.half:
                input_var = input_var.half()

            preds[idx, :] = nn.Softmax(dim=1)(model(input_var))
            labels[idx] = target.int()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # if i % args.print_freq == 0:
            #     print('Predict: [{0}/{1}]\t'
            #           'Time {batch_time.val:.3f} ({batch_time.avg:.3f})'
            #           .format(i, len(loader), batch_time=batch_time))

    return preds.cpu().data.numpy(), labels.cpu().data.numpy(), batch_time.sum


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

# Setup basic CNN model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        in_channels = 1
        default_cfg = [64,128]
        cfg = default_cfg
        self.conv1 = nn.Conv2d(in_channels, cfg[0], kernel_size=3)
        self.bn1 = nn.BatchNorm2d(cfg[0])
        self.conv2 = nn.Conv2d(cfg[0], cfg[1], kernel_size=3)
        self.bn2 = nn.BatchNorm2d(cfg[1])
        self.fc = nn.Linear(cfg[1], 10)
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = F.max_pool2d(x,2)
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = F.avg_pool2d(x, x.size()[3])
        x = x.view(x.size(0),-1)
        return self.fc(x)

class EarlyBird():
    def __init__(self, percent, epoch_keep=5):
        self.percent = percent
        self.epoch_keep = epoch_keep
        self.masks = []
        self.dists = [1 for i in range(1, self.epoch_keep)]

    def pruning(self, model, percent):
        total = 0
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                total += m.weight.data.shape[0]

        bn = torch.zeros(total)
        index = 0
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                size = m.weight.data.shape[0]
                bn[index:(index+size)] = m.weight.data.abs().clone()
                index += size

        y, i = torch.sort(bn)
        thre_index = int(total * percent)
        thre = y[thre_index]
        # print('Pruning threshold: {}'.format(thre))

        mask = torch.zeros(total)
        index = 0
        for k, m in enumerate(model.modules()):
            if isinstance(m, nn.BatchNorm2d):
                size = m.weight.data.numel()
                weight_copy = m.weight.data.abs().clone()
                _mask = weight_copy.gt(thre.cuda()).float().cuda()
                mask[index:(index+size)] = _mask.view(-1)
                # print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.format(k, _mask.shape[0], int(torch.sum(_mask))))
                index += size

        # print('Pre-processing Successful!')
        return mask

    def put(self, mask):
        if len(self.masks) < self.epoch_keep:
            self.masks.append(mask)
        else:
            self.masks.pop(0)
            self.masks.append(mask)

    def cal_dist(self):
        if len(self.masks) == self.epoch_keep:
            for i in range(len(self.masks)-1):
                mask_i = self.masks[-1]
                mask_j = self.masks[i]
                self.dists[i] = 1 - float(torch.sum(mask_i==mask_j)) / mask_j.size(0)
            return True
        else:
            return False

    def early_bird_emerge(self, model):
        mask = self.pruning(model, self.percent)
        self.put(mask)
        flag = self.cal_dist()
        if flag == True:
            print(self.dists)
            for i in range(len(self.dists)):
                if self.dists[i] > 0.1:
                    return False
            return True
        else:
            return False

def save_checkpoint_EB(state, is_best, epoch, filepath):
    if epoch == 'init':
        filepath = os.path.join(filepath, 'init.pth.tar')
        torch.save(state, filepath)
    elif 'EB' in str(epoch):
        filepath = os.path.join(filepath, epoch+'.pth.tar')
        torch.save(state, filepath)
    else:
        filename = os.path.join(filepath, 'ckpt'+str(epoch)+'.pth.tar')
        torch.save(state, filename)
        # filename = os.path.join(filepath, 'ckpt.pth.tar')
        # torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, os.path.join(filepath, 'model_best.pth.tar'))


if __name__ == '__main__':
    args = parser.parse_args()
    main(subset_size=args.subset_size, greedy=args.greedy)

