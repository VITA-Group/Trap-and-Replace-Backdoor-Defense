import argparse, os, time
from skimage.io import imsave
from skimage.util import img_as_ubyte

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, ConcatDataset
import torch.distributed as dist
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder

from datasets.gtsrb import GTSRB
from datasets.cifar import CIFAR_BadNet
from datasets.imagenet import BackDoorImageFolder, subset_by_class_id
from models.wideresnet import WRN28, WRN16
from torchvision.models import resnet34

from utils.utils import *

def get_args_parser():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch CIFAR Example')
    parser.add_argument('--gpu', default='2')
    parser.add_argument('--num_workers', '--cpus', default=16, type=int, help='number of threads for data loader')
    parser.add_argument('--data_root_path', '--drp', default='/ssd1/haotao/datasets', help='data root path')
    parser.add_argument('--dataset', '--ds', default='gtsrb', choices=['cifar10', 'cifar100', 'gtsrb', 'imagenet12', 'imagenet'])
    parser.add_argument('--model', '--md', default='WRN16', choices=['WRN16', 'WRN28', 'ResNet34'], help='which model to use')
    parser.add_argument('--pooling', default='avgpool', choices=['avgpool', 'maxpool'], help='which pooling layer to use')
    parser.add_argument('--ratio_holdout', default=0.1, type=float, help='size of holdout set')
    # training params:
    parser.add_argument('--batch_size', '-b', type=int, default=256, help='input batch size for training')
    parser.add_argument('--test_batch_size', '--tb', type=int, default=1000, help='input batch size for testing')
    parser.add_argument('--epochs', '-e', type=int, default=200, help='number of epochs to train')
    parser.add_argument('--opt', default='adam', choices=['sgd', 'adam'], help='optimizer')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--wd', type=float, default=5e-4, help='weight decay')
    # attack params:
    parser.add_argument('--target', default=0, type=int, help='target class')
    parser.add_argument('--triggered_ratio', '--ratio', default=0.1, type=float, help='ratio of poisoned data in training set')
    parser.add_argument('--trigger_pattern', '--pattern', default='badnet_grid', 
        choices=['badnet_sq', 'badnet_grid', 'trojan_3x3', 'trojan_8x8', 'trojan_wm', 'l0_inv', 'l2_inv', 'blend', 'smooth', 'sig', 'cl'], 
        help='pattern of trigger'
    )
    # 
    parser.add_argument('--resume', action='store_true', help='If true, resume from early stopped ckpt')
    parser.add_argument('--save_root_path', '--srp', default='/ssd1/haotao/BackDoorBlocker_results', help='data root path')
    parser.add_argument('--densely_save_ckpt', '--dsc', action='store_true')
    parser.add_argument('--ddp', action='store_true', help='If true, use distributed data parallel')
    parser.add_argument('--ddp_backend', '--ddpbed', default='nccl', choices=['nccl', 'gloo', 'mpi'], help='If true, use distributed data parallel')
    parser.add_argument('--num_nodes', default=1, type=int, help='Number of nodes')
    parser.add_argument('--node_id', default=0, type=int, help='Node ID')
    parser.add_argument('--dist_url', default='tcp://localhost:23456', type=str, help='url used to set up distributed training')
    args = parser.parse_args()
    return args

def create_save_path():
    # mkdirs:
    attack_type = args.trigger_pattern
    attack_str = 'target%d-ratio%s' % (args.target, args.triggered_ratio)
    opt_str = 'e%d-b%d-%s-lr%s-wd%s-cos-holdout%s' % (args.epochs, args.batch_size, args.opt, args.lr, args.wd, args.ratio_holdout)
    exp_str = '%s_%s' % (attack_str, opt_str)
    model_str = '%s' % (args.model)
    save_dir = os.path.join(args.save_root_path, 'normal_training', args.dataset, model_str, attack_type, exp_str)
    create_dir(save_dir)
    return save_dir

def setup(rank, ngpus_per_node, args):
    # initialize the process group
    world_size = ngpus_per_node * args.num_nodes
    dist.init_process_group(args.ddp_backend, init_method=args.dist_url, rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train(gpu_id, ngpus_per_node, args):

    save_dir = args.save_dir

    # get globale rank (thread id):
    rank = args.node_id * ngpus_per_node + gpu_id

    print(f"Running on rank {rank}.")

    # Initializes ddp:
    if args.ddp:
        setup(rank, ngpus_per_node, args)

    # intialize device:
    device = gpu_id if args.ddp else 'cuda'
    torch.backends.cudnn.benchmark = True

    # get batch size:
    train_batch_size = args.batch_size if not args.ddp else int(args.batch_size/ngpus_per_node/args.num_nodes)
    num_workers = args.num_workers if not args.ddp else int((args.num_workers+ngpus_per_node)/ngpus_per_node)

    fp_val = open(os.path.join(save_dir, 'val.txt'), 'a+')

    # data:
    if args.dataset in ['cifar10', 'cifar100', 'gtsrb']:
        if args.dataset == 'cifar10':
            num_classes = 10
            CIFAR = CIFAR10
        elif args.dataset == 'cifar100':
            num_classes = 100
            CIFAR = CIFAR100
        elif args.dataset == 'gtsrb':
            num_classes = 43
            CIFAR = GTSRB
            
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        train_set = CIFAR_BadNet(data_root_path=args.data_root_path, dataset_name=args.dataset, ratio_holdout=args.ratio_holdout, 
            split='train', triggered_ratio=args.triggered_ratio, trigger_pattern=args.trigger_pattern, target=args.target, transform=train_transform)

        holdout_set = CIFAR_BadNet(data_root_path=args.data_root_path, dataset_name=args.dataset, ratio_holdout=args.ratio_holdout, 
            split='holdout', triggered_ratio=0, trigger_pattern=args.trigger_pattern, target=args.target, transform=train_transform)

        train_set = ConcatDataset([train_set, holdout_set])

        test_poisoned_set = CIFAR_BadNet(data_root_path=args.data_root_path, dataset_name=args.dataset, 
            split='test', triggered_ratio=0, trigger_pattern=args.trigger_pattern, target=args.target, transform=test_transform)

        test_clean_set = CIFAR(args.data_root_path, train=False, transform=test_transform, download=False)

    elif 'imagenet' in args.dataset:
        if args.dataset == 'imagenet12':
            num_classes = 12
        elif args.dataset == 'imagenet':
            num_classes = 1000

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        train_set = BackDoorImageFolder(os.path.join(args.data_root_path, 'imagenet', 'train'), 
            split='train', num_classes=num_classes, ratio_holdout=args.ratio_holdout, 
            triggered_ratio=args.triggered_ratio, 
            trigger_pattern=args.trigger_pattern, attack_target=args.target, transform=train_transform
        )

        holdout_set = BackDoorImageFolder(os.path.join(args.data_root_path, 'imagenet', 'train'), 
            split='holdout', num_classes=num_classes, ratio_holdout=args.ratio_holdout, 
            triggered_ratio=args.triggered_ratio, 
            trigger_pattern=args.trigger_pattern, attack_target=args.target, transform=train_transform
        )

        train_set = ConcatDataset([train_set, holdout_set])

        test_poisoned_set = BackDoorImageFolder(os.path.join(args.data_root_path, 'imagenet', 'val'), 
            split='val',num_classes=num_classes, 
            triggered_ratio=args.triggered_ratio, 
            trigger_pattern=args.trigger_pattern, attack_target=args.target, transform=test_transform
        )

        test_clean_set = subset_by_class_id(
            ImageFolder(os.path.join(args.data_root_path, 'imagenet', 'val'), transform=test_transform), 
            selected_classes=np.arange(num_classes)
        )
        
    if args.ddp:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
    else:
        train_sampler = None
    train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=(train_sampler is None), num_workers=num_workers,
                                drop_last=True, pin_memory=True, sampler=train_sampler)
    test_poisoned_loader = DataLoader(test_poisoned_set, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_clean_loader = DataLoader(test_clean_set, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # model:
    if args.model == 'WRN16':
        model = WRN16(num_classes=num_classes, widen_factor=1).to(device)
    elif args.model == 'WRN28':
        model = WRN28(num_classes=num_classes).to(device)
    elif args.model == 'ResNet34':
        model = resnet34(num_classes=num_classes).to(device)
    if args.ddp:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu_id], broadcast_buffers=False, find_unused_parameters=True)

    # optimizer:
    if args.opt == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    elif args.opt == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=0.9)
    if 'imagenet' in args.dataset:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [30,60], gamma=0.1)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # train:
    if args.resume:
        ckpt = torch.load(os.path.join(save_dir, 'latest.pth'))
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])  
        start_epoch = ckpt['epoch']+1 
        best_clean_acc = ckpt['best_clean_acc']
        training_losses = ckpt['training_losses']
        test_clean_losses = ckpt['test_clean_losses']
        test_clean_accs = ckpt['test_clean_accs']
        test_poisoned_ASRs =ckpt['test_poisoned_ASRs']
    else:
        training_losses, test_clean_losses = [], []
        test_clean_accs, test_poisoned_ASRs = [], []
        best_clean_acc = 0
        start_epoch = 0
        clean_loss_mean_list, clean_loss_std_list, poisoned_loss_mean_list, poisoned_loss_std_list = [], [], [], []

    for epoch in range(start_epoch, args.epochs):

        # reset sampler when using ddp:
        start_time = time.time()
        if args.ddp:
            train_sampler.set_epoch(epoch)
        time1 = time.time() - start_time

        start_time = time.time()
        model.train()
        training_loss_meter = AverageMeter()
        for batch_idx, (data, labels, triggered, _) in enumerate(train_loader):
            data, labels = data.to(device), labels.to(device)

            # forward:
            logits = model(data)
            loss = F.cross_entropy(logits, labels)

            # backward:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # append:
            training_loss_meter.append(loss.item())

            if batch_idx % 100 == 0 and rank == 0:
                print('Epoch %d, batch %d: loss %.4f' % (epoch, batch_idx, loss.item()))

        # lr update:
        scheduler.step()
        time2 = time.time() - start_time

        start_time = time.time()
        if rank == 0:
            # eval on clean set:
            model.eval()
            test_acc_meter, test_loss_meter = AverageMeter(), AverageMeter()
            with torch.no_grad():
                for data, labels in test_clean_loader:
                    data, labels = data.to(device), labels.to(device)
                    logits = model(data)
                    pred = logits.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                    loss = F.cross_entropy(logits, labels)
                    test_acc_meter.append((logits.argmax(1) == labels).float().mean().item())
                    test_loss_meter.append(loss.item())
            test_clean_losses.append(test_loss_meter.avg)
            test_clean_accs.append(test_acc_meter.avg)

            # eval on poisoned set:
            model.eval()
            ASR_meter = AverageMeter()
            with torch.no_grad():
                for batch_idx, (data, labels, _, _) in enumerate(test_poisoned_loader):
                    data, labels = data.to(device), labels.to(device)
                    logits = model(data)
                    pred = logits.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                    loss = F.cross_entropy(logits, labels)
                    ASR_meter.append((logits.argmax(1) == args.target).float().mean().item())

                    # save poisoned image:
                    if epoch==0 and batch_idx==0:
                        _img = data[0].cpu().numpy()
                        _img = np.moveaxis(_img, 0, -1)
                        if 'imagenet' in args.dataset:
                            _img = _img * np.array(std) + np.array(mean)
                            _img = np.clip(_img, 0,1)
                        _img = img_as_ubyte(_img)
                        imsave(os.path.join(save_dir, 'poisoned_img.png'), _img)

            test_poisoned_ASRs.append(ASR_meter.avg)

            time3 = time.time() - start_time

            val_str = 'epoch %d (test): clean ACC %.4f, poisoned ASR %.4f | time %d+%d+%d' % (epoch, test_clean_accs[-1], test_poisoned_ASRs[-1], time1, time2, time3)
            print(val_str)
            fp_val.write(val_str + '\n')
            fp_val.flush()

            # save curves:
            training_losses.append(training_loss_meter.avg)
            plt.plot(training_losses, 'b', label='training_losses')
            plt.plot(test_clean_losses, 'g', label='test_clean_losses')
            plt.grid()
            plt.legend()
            plt.savefig(os.path.join(save_dir, 'losses.png'))
            plt.close()

            plt.plot(test_clean_accs, 'g', label='test_clean_accs')
            plt.grid()
            plt.legend()
            plt.savefig(os.path.join(save_dir, 'test_clean_accs.png'))
            plt.close()

            plt.plot(test_poisoned_ASRs, 'r', label='test_poisoned_ASRs')
            plt.grid()
            plt.legend()
            plt.savefig(os.path.join(save_dir, 'test_poisoned_ASRs.png'))
            plt.close()

            # save best model:
            if test_clean_accs[-1] > best_clean_acc:
                best_clean_acc = test_clean_accs[-1]
                torch.save(model.state_dict(), os.path.join(save_dir, 'best_clean_acc.pth'))


            # save pth:
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch, 
                'best_clean_acc': best_clean_acc,
                'training_losses': training_losses, 'test_clean_losses': test_clean_losses, 
                'test_clean_accs': test_clean_accs, 'test_poisoned_ASRs': test_poisoned_ASRs
                }, 
                os.path.join(save_dir, 'latest.pth'))

    # Clean up ddp:
    if args.ddp:
        cleanup()


if __name__ == '__main__':
    # get args:
    args = get_args_parser()

    # mkdirs:
    save_dir = create_save_path()
    args.save_dir = save_dir
    
    # set CUDA:
    if args.num_nodes == 1: # When using multiple nodes, we assume all gpus on each node are available.
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu 

    if args.ddp:
        ngpus_per_node = torch.cuda.device_count()
        torch.multiprocessing.spawn(train, args=(ngpus_per_node,args), nprocs=ngpus_per_node, join=True)
    else:
        train(0, 0, args)