import argparse, os, time
from skimage.io import imsave
from skimage.util import img_as_ubyte

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder

from datasets.gtsrb import GTSRB
from datasets.cifar import CIFAR_BadNet
from datasets.imagenet import BackDoorImageFolder, subset_by_class_id
from models.wideresnet_ae import WRN16AE
from models.resnet_ae import ResNet34AE

from utils.utils import *


# Training settings
parser = argparse.ArgumentParser(description='PyTorch CIFAR Example')
parser.add_argument('--gpu', default='2')
parser.add_argument('--num_workers', '--cpus', type=int, default=16, help='number of threads for data loader')
parser.add_argument('--data_root_path', '--drp', default='/ssd1/haotao/datasets', help='data root path')
parser.add_argument('--dataset', '--ds', default='cifar10', choices=['cifar10', 'cifar100', 'gtsrb', 'imagenet12'])
parser.add_argument('--model', '--md', default='WRN16', choices=['WRN16', 'WRN28', 'ResNet34'], help='which model to use')
parser.add_argument('--pooling', default='avgpool', choices=['avgpool', 'maxpool'], help='which pooling layer to use')
parser.add_argument('--ratio_holdout', default=0.05, type=float, help='size of holdout set')
parser.add_argument('--stem_end_block', '--stem', default=5, type=int, help='where the stem ends in model')
# training params:
parser.add_argument('--batch_size', '-b', type=int, default=256, help='input batch size for training')
parser.add_argument('--test_batch_size', '--tb', type=int, default=1000, help='input batch size for testing')
parser.add_argument('--epochs', '-e', type=int, default=200, help='number of epochs to train')
parser.add_argument('--opt', default='adam', choices=['sgd', 'adam'], help='optimizer')
parser.add_argument('--decay', default='cos', choices=['cos', 'multisteps'], help='lr schedular')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--wd', type=float, default=5e-4, help='weight decay')
parser.add_argument('--Lambda', type=float, default=10., help='img reconstruction loss weight')
parser.add_argument('--Lambda2', type=float, default=1, help='TV loss weight')
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
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
torch.backends.cudnn.benchmark = True

# mkdirs:
attack_type = args.trigger_pattern
attack_str = 'target%d-ratio%s' % (args.target, args.triggered_ratio)
opt_str = 'e%d-b%d-%s-lr%s-wd%s-%s-holdout%s' % (args.epochs, args.batch_size, args.opt, args.lr, args.wd, args.decay, args.ratio_holdout)
loss_str = 'Lambda%s-%s' % (args.Lambda, args.Lambda2)
exp_str = '%s_%s_%s' % (attack_str, opt_str, loss_str)
model_str = '%s-stem%d' % (args.model, args.stem_end_block)
save_dir = os.path.join(args.save_root_path, 'backdoor_trapper_bait_and_trap', args.dataset, model_str, attack_type, exp_str)
create_dir(save_dir)

fp_train = open(os.path.join(save_dir, 'train.txt'), 'a+')
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
    
train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                            drop_last=True, pin_memory=True)
test_poisoned_loader = DataLoader(test_poisoned_set, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
test_clean_loader = DataLoader(test_clean_set, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

# dataset_info_str = 'Training on %d images (excluding %d clean holdout images) with %d poisoned images' % (train_set.N, train_set.N_holdout, train_set.N_triggered)
# print(dataset_info_str)
# fp_val.write(dataset_info_str + '\n')

# model:
if args.model == 'WRN16':
    model = WRN16AE(num_classes=num_classes, stem_end_block=args.stem_end_block).cuda()
elif args.model == 'ResNet34':
    model = ResNet34AE(num_classes=num_classes, stem_end_block=args.stem_end_block).cuda()

# optimizer:
if args.opt == 'adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
elif args.opt == 'sgd':
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=0.9)
if args.decay == 'cos':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
elif args.decay == 'multisteps':
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [int(0.5*args.epochs), int(0.75*args.epochs)], gamma=0.1)

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
    start_time = time.time()
    model.train()
    training_loss_meter = AverageMeter()
    for batch_idx, (data, labels, triggered, _) in enumerate(train_loader):
        data, labels = data.cuda(), labels.cuda()

        # forward:
        logits, imgs_rec = model(data)
        loss_clf = F.cross_entropy(logits, labels)
        if 'imagenet' in args.dataset:
            data = data * std_tensor + mean_tensor
        loss_rec = F.mse_loss(imgs_rec, data)
        loss_tv = total_variation_loss(imgs_rec)

        loss = loss_clf + args.Lambda * loss_rec + args.Lambda2 * loss_tv

        # backward:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # append:
        training_loss_meter.append(loss.item())

        if batch_idx % 100 == 0:
            train_log_str = 'epoch %d, batch %d: loss %s (%.4f, %.4f, %.4f)' % (
                epoch, batch_idx, loss.item(), loss_clf.item(), loss_rec.item(), loss_tv.item())
            print(train_log_str)
            fp_train.write(train_log_str + '\n')
            fp_train.flush()


    # eval on clean set:
    model.eval()
    test_acc_meter, test_loss_meter = AverageMeter(), AverageMeter()
    with torch.no_grad():
        for data, labels in test_clean_loader:
            data, labels = data.cuda(), labels.cuda()
            logits, _ = model(data)
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
            data, labels = data.cuda(), labels.cuda()
            logits, imgs_rec = model(data)
            pred = logits.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            loss = F.cross_entropy(logits, labels)
            ASR_meter.append((logits.argmax(1) == args.target).float().mean().item())

            # save poisoned image:
            if batch_idx==0:
                _img = data[0].cpu().numpy()
                _img = np.moveaxis(_img, 0, -1)
                if 'imagenet' in args.dataset:
                    _img = _img * np.array(std) + np.array(mean)
                    _img = np.clip(_img, 0,1)
                _img = img_as_ubyte(_img)
                imsave(os.path.join(save_dir, 'poisoned_img.png'), _img)

                _img = imgs_rec[0].cpu().numpy()
                _img = np.moveaxis(_img, 0, -1)
                # if 'imagenet' in args.dataset:
                #     _img = _img * np.array(std) + np.array(mean)
                #     _img = np.clip(_img, 0,1)
                _img = img_as_ubyte(_img)
                imsave(os.path.join(save_dir, 'poisoned_img_rec.png'), _img)
                

    test_poisoned_ASRs.append(ASR_meter.avg)

    val_str = 'epoch %d (test): clean ACC %.4f, poisoned ASR %.4f | time %s' % (epoch, test_clean_accs[-1], test_poisoned_ASRs[-1], time.time()-start_time)
    print(val_str)
    fp_val.write(val_str + '\n')
    fp_val.flush()

    # lr update:
    scheduler.step()

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

