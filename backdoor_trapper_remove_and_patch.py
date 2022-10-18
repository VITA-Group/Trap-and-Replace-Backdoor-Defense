'''
Remove the poisoned branch and patch it with a new one trained on clean holdout set.
'''
import argparse, os, copy
import itertools
from skimage.io import imsave
from skimage.util import img_as_ubyte

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder
from timm.loss.cross_entropy import LabelSmoothingCrossEntropy

from datasets.gtsrb import GTSRB
from datasets.cifar import CIFAR_BadNet
from datasets.imagenet import BackDoorImageFolder, subset_by_class_id
from models.wideresnet_aux import WRN16AUX
from models.resnet_aux import ResNet34AUX

from utils.utils import *
from utils.loss_fn import *
from utils.sampler import RandomSampler, BatchSampler

# Training settings
parser = argparse.ArgumentParser(description='PyTorch CIFAR Example')
parser.add_argument('--gpu', default='0')
parser.add_argument('--num_workers', '--cpus', type=int, default=8, help='number of threads for data loader')
parser.add_argument('--data_root_path', '--drp', default='/ssd1/haotao/datasets', help='data root path')
parser.add_argument('--dataset', '--ds', default='cifar10', choices=['cifar10', 'cifar100', 'gtsrb', 'imagenet12'])
parser.add_argument('--model', '--md', default='WRN16', choices=['WRN16', 'ResNet34'], help='which model to use')
parser.add_argument('--ratio_holdout', default=0.05, type=float, help='size of holdout set')
parser.add_argument('--stem_end_block', '--stem', default=5, type=int, help='where the stem ends in model')
# training params:
parser.add_argument('--batch_size', '-b', type=int, default=64, help='input batch size for training')
parser.add_argument('--test_batch_size', '--tb', type=int, default=1000, help='input batch size for testing')
parser.add_argument('--epochs', '-e', type=int, default=100, help='number of epochs to train') # original FixMatch uses 1024 epochs
parser.add_argument('--lr', type=float, default=0.03, help='learning rate')
parser.add_argument('--wd', type=float, default=5e-4, help='weight decay')
parser.add_argument('--opt', default='adam', choices=['sgd', 'adam'], help='optimizer')
parser.add_argument('--decay', default='cos', choices=['cos', 'multisteps'], help='lr schedular')
parser.add_argument('--dropout', type=float, default=0.5, help='aux branch dropout rate')
parser.add_argument('--no_trap', action='store_true', help='If true, start from normal training')
parser.add_argument('--beta', default=1.0, type=float, help='hyperparameter beta')
parser.add_argument('--cutmix_prob', default=1.0, type=float, help='cutmix probability')
parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
# attack params:
parser.add_argument('--target', default=0, type=int, help='target class')
parser.add_argument('--triggered_ratio', '--ratio', default=0.1, type=float, help='ratio of poisoned data in training set')
parser.add_argument('--trigger_pattern', '--pattern', default='badnet_grid', 
    choices=['badnet_sq', 'badnet_grid', 'trojan_3x3', 'trojan_8x8', 'trojan_wm', 'l0_inv', 'l2_inv', 'blend', 'smooth', 'sig', 'cl'], 
    help='pattern of trigger'
)
parser.add_argument('--lambda_str', default='10.0-1', help='stage 1 lambda_str')
# 
parser.add_argument('--resume', action='store_true', help='If true, resume from early stopped ckpt')
parser.add_argument('--save_root_path', '--srp', default='/ssd1/haotao/BackDoorBlocker_results', help='data root path')
parser.add_argument('--densely_save_ckpt', '--dsc', action='store_true')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
torch.backends.cudnn.benchmark = True

# mkdirs:
attack_type = args.trigger_pattern
attack_str = 'target%d-ratio%s' % (args.target, args.triggered_ratio)
opt_str = 'e%d-b%d-%s-lr%s-wd%s-%s-holdout%s-dp%s' % (args.epochs, args.batch_size, args.opt, args.lr, args.wd, args.decay, args.ratio_holdout, args.dropout)
loss_str = 'mixup-beta%s-prob%s-smoothing%s' % (args.beta, args.cutmix_prob, args.smoothing)
exp_str = '%s_%s_%s' % (attack_str, opt_str, loss_str)
model_str = '%s-stem%d' % (args.model, args.stem_end_block)

if 'imagenet' in args.dataset:
    if args.no_trap:
        stage1_ckpt_dir = os.path.join(args.save_root_path, 'normal_training', args.dataset, args.model, attack_type, 
            'target%d-ratio0.1_e90-b256-sgd-lr0.1-wd0.0005-cos-holdout%s' % (args.target, args.ratio_holdout))
    else:
        # stage1_ckpt_dir = os.path.join(args.save_root_path, 'backdoor_trapper_bait_and_trap', args.dataset, model_str, attack_type, 
        #     'target%d-ratio0.1_e90-b256-sgd-lr0.1-wd0.0005-cos-holdout%s_Lambda10.0-0.0' % (args.target, args.ratio_holdout))
        stage1_ckpt_dir = os.path.join(args.save_root_path, 'backdoor_trapper_bait_and_trap', args.dataset, model_str, attack_type, 
            'target%d-ratio0.1_e90-b256-sgd-lr0.1-wd0.0005-cos-holdout0.05_Lambda%s' % (args.target, args.lambda_str))
else:
    if args.no_trap:
        stage1_ckpt_dir = os.path.join(args.save_root_path, 'normal_training', args.dataset, args.model, attack_type, 
            'target%d-ratio0.1_e200-b256-adam-lr0.001-wd0.0005-cos-holdout%s' % (args.target, args.ratio_holdout))
    else:
        stage1_ckpt_dir = os.path.join(args.save_root_path, 'backdoor_trapper_bait_and_trap', args.dataset, model_str, attack_type, 
            'target%d-ratio%s_e200-b256-adam-lr0.001-wd0.0005-cos-holdout%s_Lambda%s' % (args.target, args.triggered_ratio, args.ratio_holdout, args.lambda_str))


if not os.path.exists(stage1_ckpt_dir):
    raise Exception('stage1_ckpt_dir does not exist %s' % stage1_ckpt_dir)

save_dir = os.path.join(stage1_ckpt_dir, 'remove_and_patch', exp_str)
# if os.path.exists(save_dir) and not args.resume:
#     raise Exception('Save dir already exists! Please make sure not overwriting previous results!')
create_dir(save_dir)

fp_train = open(os.path.join(save_dir, 'train.txt'), 'a+')
fp_val = open(os.path.join(save_dir, 'val.txt'), 'a+')
fp_benign_pred = open(os.path.join(save_dir, 'benign_pred.txt'), 'a+')
fp_poisoned_pred = open(os.path.join(save_dir, 'poisoned_pred.txt'), 'a+')

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

    holdout_set = CIFAR_BadNet(data_root_path=args.data_root_path, dataset_name=args.dataset, ratio_holdout=args.ratio_holdout, 
        split='holdout', triggered_ratio=0, trigger_pattern=args.trigger_pattern, target=args.target, transform=train_transform)

    combined_train_set = ConcatDataset([holdout_set])

    test_poisoned_set = CIFAR_BadNet(data_root_path=args.data_root_path, dataset_name=args.dataset, 
        split='test', triggered_ratio=0, trigger_pattern=args.trigger_pattern, target=args.target, transform=test_transform)

    test_clean_set = CIFAR(args.data_root_path, train=False, transform=test_transform, download=False)

    detect_train_set = CIFAR_BadNet(data_root_path=args.data_root_path, dataset_name=args.dataset, ratio_holdout=args.ratio_holdout, 
        split='train', triggered_ratio=args.triggered_ratio, trigger_pattern=args.trigger_pattern, target=args.target, transform=test_transform)

elif 'imagenet' in args.dataset:
    if args.dataset == 'imagenet12':
        num_classes = 12
    elif args.dataset == 'imagenet':
        num_classes = 1000

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    std_tensor = torch.Tensor(std).cuda().view((1,3,1,1))
    mean_tensor = torch.Tensor(mean).cuda().view((1,3,1,1))
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

    holdout_set = BackDoorImageFolder(os.path.join(args.data_root_path, 'imagenet', 'train'), 
        split='holdout', num_classes=num_classes, ratio_holdout=args.ratio_holdout, 
        triggered_ratio=args.triggered_ratio, 
        trigger_pattern=args.trigger_pattern, attack_target=args.target, transform=train_transform
    )

    combined_train_set = ConcatDataset([holdout_set])

    test_poisoned_set = BackDoorImageFolder(os.path.join(args.data_root_path, 'imagenet', 'val'), 
        split='val',num_classes=num_classes, 
        triggered_ratio=args.triggered_ratio, 
        trigger_pattern=args.trigger_pattern, attack_target=args.target, transform=test_transform
    )

    test_clean_set = subset_by_class_id(
        ImageFolder(os.path.join(args.data_root_path, 'imagenet', 'val'), transform=test_transform), 
        selected_classes=np.arange(num_classes)
    )

    detect_train_set = BackDoorImageFolder(os.path.join(args.data_root_path, 'imagenet', 'train'), 
        split='train', num_classes=num_classes, ratio_holdout=args.ratio_holdout, 
        triggered_ratio=args.triggered_ratio, 
        trigger_pattern=args.trigger_pattern, attack_target=args.target, transform=test_transform
    )

train_loader = DataLoader(combined_train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                            drop_last=True, pin_memory=True)

detect_train_loader = DataLoader(detect_train_set, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers,
                            drop_last=False, pin_memory=True)

test_poisoned_loader = DataLoader(test_poisoned_set, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

test_clean_loader = DataLoader(test_clean_set, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)


# model:
if args.model == 'WRN16':
    model = WRN16AUX(num_classes=num_classes, stem_end_block=args.stem_end_block, aux_drop_rate=args.dropout).cuda()
elif args.model == 'ResNet34':
    model = ResNet34AUX(num_classes=num_classes, stem_end_block=args.stem_end_block, aux_drop_rate=args.dropout).cuda()

# optimizer:
if args.opt == 'adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
elif args.opt == 'sgd':
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=0.9)
if args.decay == 'cos':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
elif args.decay == 'multisteps':
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [int(0.5*args.epochs), int(0.75*args.epochs)], gamma=0.1)

loss_fn = LabelSmoothingCrossEntropy(smoothing=args.smoothing)

# train:
if args.resume:
    ckpt = torch.load(os.path.join(save_dir, 'latest.pth'))
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    scheduler.load_state_dict(ckpt['scheduler'])  
    start_epoch = ckpt['epoch']+1 
    best_aux_acc = ckpt['best_aux_acc']
    training_losses = ckpt['training_losses']
    test_clean_losses = ckpt['test_clean_losses']
    training_ASRs = ckpt['training_ASRs']
    training_ASRs_aux = ckpt['training_ASRs_aux']
    test_clean_accs = ckpt['test_clean_accs']
    test_clean_accs_aux = ckpt['test_clean_accs_aux']
    test_poisoned_ASRs = ckpt['test_poisoned_ASRs']
    test_poisoned_ASRs_aux = ckpt['test_poisoned_ASRs_aux']
    detection_TPRs = ckpt['detection_TPRs']
    detection_FPRs = ckpt['detection_FPRs']
    final_triggered_preds = ckpt['final_triggered_preds']
    final_benign_preds = ckpt['final_benign_preds']
else:
    training_losses, test_clean_losses = [], []
    training_ASRs, training_ASRs_aux = [], []
    test_clean_accs, test_clean_accs_aux = [], []
    test_poisoned_ASRs, test_poisoned_ASRs_aux = [], []
    detection_TPRs, detection_FPRs = [], []
    clean_loss_mean_list, clean_loss_min_list, clean_loss_max_list = [], [], []
    poisoned_loss_mean_list, poisoned_loss_min_list, poisoned_loss_max_list = [], [], []
    best_aux_acc = 0
    start_epoch = 0
    final_triggered_preds = torch.zeros(len(detect_train_set)).cuda().bool()
    final_benign_preds = torch.zeros(len(detect_train_set)).cuda().bool()

    # load from normal_training models:
    stage1_ckpt_path = os.path.join(stage1_ckpt_dir, 'latest.pth')
    model.load_state_dict(torch.load(stage1_ckpt_path)['model'], strict=False)

# set correct require_grad:
for name, p in model.named_parameters():
    if 'aux' in name:
        p.requires_grad = True
    else:
        p.requires_grad = False
    # print(name, p.shape, p.requires_grad)

for epoch in range(start_epoch, args.epochs):

    for name, m in model.named_modules():
        if 'aux' in name:
            m.train()
        else:
            m.eval()
        # print(name, m.training)

    training_loss_meter = AverageMeter()
    _N_triggered = 0
    _N_total = len(detect_train_set)

    for batch_idx, (data, labels, triggered, _) in enumerate(train_loader):
        
        data, labels = data.cuda(), labels.cuda()

        # forward:
        r = np.random.rand(1)
        if args.beta > 0 and r < args.cutmix_prob:
            # generate mixed sample
            _lambda = np.random.beta(args.beta, args.beta)
            rand_index = torch.randperm(data.size()[0]).to(data.device)
            target_a = labels
            target_b = labels[rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), _lambda)
            data[:, :, bbx1:bbx2, bby1:bby2] = data[rand_index, :, bbx1:bbx2, bby1:bby2]
            # adjust lambda to exactly match pixel ratio
            _lambda = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))
            # compute output
            _, logits_aux = model(data)
            loss = loss_fn(logits_aux, target_a) * _lambda + loss_fn(logits_aux, target_b) * (1. - _lambda)
        else:
            _, logits_aux = model(data)

            loss = F.cross_entropy(logits_aux, labels)
        
        # backward:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # append:
        training_loss_meter.append(loss.item())

        if batch_idx % 100 == 0:
            train_log_str = 'epoch %d, batch %d: loss %s' % (
                epoch, batch_idx, loss.item())
            print(train_log_str)
            fp_train.write(train_log_str + '\n')
            fp_train.flush()

        _N_triggered += torch.sum(triggered).item()

    # eval on clean set:
    model.eval()

    # # detection on training set:
    # detection_TP, detection_P, detection_FP, detection_N = 0, 0, 0, 0
    # N_poisoned = 0
    # N_AS, N_AS_aux = 0, 0
    # with torch.no_grad():
    #     training_acc_meter, training_mask_acc_meter = AverageMeter(), AverageMeter()
    #     all_clean_losses, all_poisoned_losses = [], []
    #     for i, (data, labels, triggered, index) in enumerate(detect_train_loader):
    #         data, labels, triggered = data.to('cuda'), labels.to('cuda'), triggered.to('cuda')

    #         N_poisoned += triggered.float().sum().item()

    #         logits, aux_logits = model(data)

    #         preds = logits.argmax(dim=1)
    #         N_AS += torch.logical_and(preds == args.target, triggered).float().sum().item()

    #         aux_preds = aux_logits.argmax(dim=1)
    #         N_AS_aux += torch.logical_and(aux_preds == args.target, triggered).float().sum().item()

    #         triggered_preds = aux_preds != preds
    #         detection_TP += torch.logical_and(triggered_preds, triggered).float().sum().item()
    #         detection_P += (triggered).float().sum().item()
    #         detection_FP += torch.logical_and(triggered_preds, ~triggered).float().sum().item()
    #         detection_N += (~triggered).float().sum().item()

    #         final_triggered_preds[index] = triggered_preds

    #         loss = F.cross_entropy(aux_logits, labels, reduction='none')
    #         if torch.sum(~triggered) != 0:
    #             all_clean_losses.append(loss[~triggered])
    #         if torch.sum(triggered) != 0:
    #             all_poisoned_losses.append(loss[triggered])

            # # filter by loss value:
            # if epoch>50:
            #     if args.th>0:
            #         th_idx = int(len(loss)*args.th)
            #         loss_th = torch.sort(loss)[th_idx]
            #         loss_smaller_than_th = loss < loss_th
            #         benign_preds = torch.logical_and(torch.logical_not(triggered_preds), loss_smaller_than_th)
            #     else:
            #         benign_preds = torch.logical_not(triggered_preds)
            #     final_benign_preds[index] = benign_preds

        # training_ASR = N_AS / N_poisoned
        # training_ASR_aux = N_AS_aux / N_poisoned
        # detection_TPR = detection_TP / detection_P
        # detection_FPR = detection_FP / detection_N

    # training_ASRs.append(training_ASR)
    # training_ASRs_aux.append(training_ASR_aux)

    # # save loss curves with error bar:
    # all_clean_losses = torch.cat(all_clean_losses, dim=0)
    # all_poisoned_losses = torch.cat(all_poisoned_losses, dim=0)

    # clean_loss_mean = all_clean_losses.mean(0)
    # clean_loss_min = all_clean_losses.min(0).values
    # clean_loss_max = all_clean_losses.max(0).values
    # poisoned_loss_mean = all_poisoned_losses.mean(0)
    # poisoned_loss_min = all_poisoned_losses.min(0).values
    # poisoned_loss_max = all_poisoned_losses.max(0).values

    # clean_loss_mean_list.append(clean_loss_mean.item())
    # clean_loss_min_list.append(clean_loss_min.item())
    # clean_loss_max_list.append(clean_loss_max.item())
    # poisoned_loss_mean_list.append(poisoned_loss_mean.item())
    # poisoned_loss_min_list.append(poisoned_loss_min.item())
    # poisoned_loss_max_list.append(poisoned_loss_max.item())

    # plt.plot(np.arange(epoch+1), clean_loss_mean_list, marker='^', color='blue', label='clean sample loss')
    # plt.fill_between(np.arange(epoch+1), 
    #     np.array(clean_loss_max_list), np.array(clean_loss_min_list), 
    #     color='blue', alpha=0.5
    # )
    # plt.plot(np.arange(epoch+1), poisoned_loss_mean_list, marker='^', color='orange', label='poisoned sample loss')
    # plt.fill_between(np.arange(epoch+1), 
    #     np.array(poisoned_loss_max_list), np.array(poisoned_loss_min_list), 
    #     color='orange', alpha=0.5
    # )
    # plt.legend()
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss value')
    # plt.savefig(os.path.join(save_dir, 'losses_error_bar.png'))
    # plt.close()

    # get predictions on clean test set:
    test_acc_meter, test_acc_meter_aux, test_loss_meter = AverageMeter(), AverageMeter(), AverageMeter()
    with torch.no_grad():
        for data, labels in test_clean_loader:
            data, labels = data.cuda(), labels.cuda()

            logits, aux_logits = model(data)
            loss = F.cross_entropy(aux_logits, labels)

            acc = (logits.argmax(1) == labels).float().mean().item()
            test_acc_meter.append(acc)

            mask_acc = (aux_logits.argmax(1) == labels).float().mean().item()
            test_acc_meter_aux.append(mask_acc)
            test_loss_meter.append(loss.item())
    
    test_clean_accs.append(test_acc_meter.avg)
    test_clean_accs_aux.append(test_acc_meter_aux.avg)
    test_clean_losses.append(test_loss_meter.avg)

    # eval on poisoned set:
    model.eval()
    ASR_meter, ASR_meter_aux = AverageMeter(), AverageMeter()
    with torch.no_grad():
        for batch_idx, (data, labels, _, _) in enumerate(test_poisoned_loader):
            data, labels = data.cuda(), labels.cuda()
            logits, aux_logits = model(data)
            ASR_meter.append((logits.argmax(1) == args.target).float().mean().item())
            ASR_meter_aux.append((aux_logits.argmax(1) == args.target).float().mean().item())

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
    test_poisoned_ASRs_aux.append(ASR_meter_aux.avg)

    # lr update:
    scheduler.step()

    # val_str:
    val_str = 'epoch %d (test): Main ACC %.4f | Aux ACC %.4f | Main ASR %.4f | Aux ASR %.4f' % (
        epoch, test_clean_accs[-1], test_clean_accs_aux[-1], test_poisoned_ASRs[-1], test_poisoned_ASRs_aux[-1]) 
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

    # plt.plot(training_ASRs, 'r', label='main branch')
    # plt.plot(training_ASRs_aux, 'r--', label='aux branch')
    # plt.grid()
    # plt.legend()
    # plt.savefig(os.path.join(save_dir, 'training_poisoned_ASRs.png'))
    # plt.close()


    plt.plot(test_clean_accs, 'g', label='main branch')
    plt.plot(test_clean_accs_aux, 'g--', label='aux branch')
    plt.grid()
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'test_clean_accs.png'))
    plt.close()

    # detection_TPRs.append(detection_TPR)
    # detection_FPRs.append(detection_FPR)
    # plt.plot(detection_TPRs, 'r', label='detection_TPRs')
    # plt.plot(detection_FPRs, 'r--', label='detection_FPRs')
    # plt.grid()
    # plt.legend()
    # plt.savefig(os.path.join(save_dir, 'detection_metrics.png'))
    # plt.close()

    # save best model:
    if test_clean_accs_aux[-1] > best_aux_acc:
        best_aux_acc = test_clean_accs_aux[-1]
        torch.save(model.state_dict(), os.path.join(save_dir, 'best_aux_acc.pth'))

    # save pth:
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': epoch, 
        'best_aux_acc': best_aux_acc,
        'training_losses': training_losses, 
        'test_clean_losses': test_clean_losses, 
        'training_ASRs': training_ASRs, 
        'training_ASRs_aux': training_ASRs_aux, 
        'test_clean_accs': test_clean_accs, 
        'test_clean_accs_aux': test_clean_accs_aux, 
        'test_poisoned_ASRs': test_poisoned_ASRs, 
        'test_poisoned_ASRs_aux': test_poisoned_ASRs_aux, 
        'detection_TPRs': detection_TPRs,
        'detection_FPRs': detection_FPRs,
        'final_triggered_preds': final_triggered_preds
        }, 
        os.path.join(save_dir, 'latest.pth'))