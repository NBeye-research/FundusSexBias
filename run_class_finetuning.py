
import argparse
import os
import random
import time
import warnings
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from engine_for_finetuning import train_one_epoch, evaluate, evaluate_sex, evaluate_sex_multi_label
import datasets
import models
import utils
import utils.lr_decay as lrd


def main_worker(args):
    
    print('参数:', args)

    if args.multi_label:
        args.dataset='sex-baias-multi-label'
    
    if not args.evaluate:
        # train root_path, tag_path, input_size, label_columns, partition_tag='train', gender_tag=None
        train_dataset = datasets.make(args.dataset, root_path=args.data_path, tag_path = args.tag_path,
                                    input_size=args.image_size, label_columns = args.labels, partition_tag='train', sex_tag = args.sex)
        print('train dataset root path:{}, shape: {} (x{})'.format(train_dataset.rootpath, train_dataset[0][0].shape, len(train_dataset)))
        val_dataset = datasets.make(args.dataset, root_path=args.data_path, tag_path = args.tag_path,
                                    input_size=args.image_size, label_columns = args.labels, partition_tag='val', sex_tag = args.sex)

        print('val dataset root path:{}, shape: {} (x{})'.format(val_dataset.rootpath, val_dataset[0][0].shape, len(val_dataset)))

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
        
        dataset_name = os.path.basename(args.data_path)
        model_store_dir = os.path.join(args.output_dir, args.arch, args.arch + '_' + dataset_name + '_' + args.tag)
        os.makedirs(model_store_dir, exist_ok=True)
        args.model_store_dir = model_store_dir
        print('model store path:', model_store_dir)
    
    partition_tag = 'test'
    if  args.test_data_path and args.test_tag_path:
        root_path = args.test_data_path
        tag_path = args.test_tag_path
        partition_tag = 'outer_test'
    
    test_dataset = datasets.make(args.dataset, root_path=args.data_path, tag_path = args.tag_path,
                                    input_size=args.image_size, label_columns = args.labels, partition_tag=partition_tag, sex_tag = args.sex)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    print('test dataset root path:{}, shape: {} (x{})'.format(test_dataset.rootpath, test_dataset[0][0].shape, len(test_dataset)))
    
    # Get number of labels
    n_classes = test_dataset.n_classes
    print('batch_size:{}, num_workers:{}， n_classes:{}'.format(args.batch_size, args.workers, n_classes))

    if args.evaluate:
        args.pretrained = False
    model = models.make(args.arch, pretrained=args.pretrained, n_classes=n_classes)
    # print(model.state_dict().keys())
    
    if args.fine_tuning:
        utils.load_state_dict_from_pretrained(model, args)
        # exit(0)

    model_without_ddp = model
    if args.gpu is not None:
        device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
        model.to(device)
        if args.parallel:
            model = torch.nn.DataParallel(model)
            model_without_ddp = model.module
    
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of model params (M): %.2f' % (n_parameters / 1.e6))

    if args.arch in ('RETFound_mae', 'vit_large_patch16_224', 'swin_large_patch16'):

        no_weight_decay = model_without_ddp.no_weight_decay() if hasattr(model_without_ddp, 'no_weight_decay') else []
        param_groups = lrd.param_groups_lrd(model_without_ddp, args.weight_decay,
                                            no_weight_decay_list=no_weight_decay,
                                            layer_decay=args.layer_decay
                                            )
        optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    else :
        # optimizer = torch.optim.SGD(model.parameters(),lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        optimizer = torch.optim.AdamW(model_without_ddp.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    if args.multi_label:
        criterion = torch.nn.BCEWithLogitsLoss().cuda()
    else:
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=args.smoothing).cuda()
    
    cudnn.benchmark = True
    begin_time = time.time()
    if args.evaluate:
        res = evaluate(val_loader, model, criterion, args)
        print('预测耗时：{}'.format(int(time.time() - begin_time)))
        return

    #auto resume
    utils.auto_load_model(args, model, model_without_ddp, optimizer)
    
    # times = []
    epoch_list, train_loss_list, test_loss_list, auc_list, step_losses = [], [], [], [], []
    max_auc = 0.
    min_loss = 1000.0
    
    for epoch in range(args.start_epoch, args.epochs):

        # adjust_learning_rate(optimizer, epoch, args)
        # train for one epoch
        losses, step_loss, acc = train_one_epoch(train_loader, model, criterion, optimizer, epoch, args)

        print('Epoch:{}, loss:{:.4f}'.format(epoch, losses.avg))
        train_loss_list.append(losses.avg)
        epoch_list.append(epoch+1)

        # evaluate on validation set
        if args.multi_label:
            acc1,test_losses,auc = evaluate_sex_multi_label(test_loader, criterion, model, device, args, num_class=n_classes, mode = 'val')
        else:
            acc1,test_losses,auc = evaluate_sex(test_loader, criterion, model, device, args, num_class=n_classes, mode = 'val')
        # acc1,test_losses,auc = evaluate(val_loader, model, criterion, args)
        test_loss_list.append(test_losses.avg)
        auc_list.append(auc)
        step_losses.extend(step_loss)
        
        if args.parallel:
            model_ = model.module
        else:
            model_ = model
            
        save_obj = {
            'epoch': epoch + 1,
            'arch': args.arch,
            'model': model_.state_dict(),
            'train_loss':train_loss_list,
            'train_step_loss':step_losses,
            'val_loss':test_loss_list,
            'val_auc':auc_list,
            'max_auc': max_auc,
            'optimizer' : optimizer.state_dict(),
            'model_args': vars(args),
        }
        
        #保留最近的模型训练文件
        torch.save(save_obj, os.path.join(model_store_dir, 'checkpoint.pth'))
        # remember best auc and save checkpoint
        if auc > max_auc:
            max_auc = auc
            torch.save(save_obj, os.path.join(model_store_dir, 'checkpoint-best.pth'))
        
        if epoch == (args.epochs - 1):
            checkpoint = torch.load(os.path.join(args.model_store_dir, 'checkpoint-best.pth'), map_location='cpu')
            model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
            model.to(device)
            print("Test with the best model, epoch = %d:" % checkpoint['epoch'])
            if args.multi_label:
                evaluate_sex_multi_label(test_loader, criterion, model, device, args, num_class=n_classes, mode = 'test')
            else:
                evaluate_sex(test_loader, criterion, model, device, args, num_class=n_classes, mode = 'test')

    print('Finished Training best_auc: ', max_auc)


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    # lr = args.lr * (0.1 ** (epoch // 20))
    lr_decay = 0.1 ** (epoch // 20)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * lr_decay
        # param_group['lr'] = lr


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    
    parser.add_argument('--dataset', type=str, default='sex-baias',help='path to dataset')
    parser.add_argument('--data_path', default='./data/', type=str,
                        help='dataset path')
    parser.add_argument('--output_dir', metavar='DIR', default='', help='path to model file.')
    parser.add_argument('--tag', metavar='TAG', default='', help='path to dataset')
    parser.add_argument('--fine_tuning',default='', help='pretrained model path.')

    parser.add_argument('-a', '--arch', metavar='ARCH', default='vit_large_patch16_224',
                        help='model architecture (convnext_base、densenet121、vit_base_patch16_224_dino、vit_large_patch16_224)')

    def parse_weights(weights):
        return [float(w) for w in weights.split(',')]

    parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=50, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch_size', default=32, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                             'batch size of all GPUs on the current node when '
                             'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=5e-5, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--auto_resume', action='store_true', default=True)
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', default=False, 
                        help='evaluate moinrtdel on validation set')
    parser.add_argument('--pretrained', default=True, dest='pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--model_key', default='model|module', type=str)

    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--image_size', default=224, type=int,
                        help='image size')
    parser.add_argument('--advprop', default=False, action='store_true',
                        help='use advprop or not')
    parser.add_argument('--gpu', default='0,1,2,3', help='gpu ids.')
    parser.add_argument('--store_misclassification',  action='store_true', help='测试时候，是否保存错误分类，默认不保存。', default=False)
    parser.add_argument('--store_metrics', action='store_true', help='测试时候，保存模型训练过程指标。', default=False)

    # Optimizer parameters
    parser.add_argument('--drop_path', type=float, default=0.2, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--blr', type=float, default=5e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--layer_decay', type=float, default=0.65,
                        help='layer-wise lr decay from ELECTRA/BEiT')
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N',
                        help='epochs to warmup LR')
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=None, metavar='PCT',
                        help='Color jitter factor (enabled only when not using Auto/RandAug)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
    
    # Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    parser.add_argument('--tag_path', metavar='DIR', default='')
    parser.add_argument('--sex', type=str, default=None)
    parser.add_argument('--config_log_path', metavar='DIR', default='')

    parser.add_argument('--test_data_path', default=None, type=str,help='dataset path')
    parser.add_argument('--test_tag_path', metavar='DIR', default=None, help='存储数据标签路径.')
    def parse_names(names_str):
        return sorted(names_str.split(','))   
    parser.add_argument('--labels', type=parse_names, default=None)
    parser.add_argument('--multi_label', action='store_true', default=False) 
    
    args = parser.parse_args()
    
    utils.set_gpu(args.gpu)
    
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
        
    args.parallel = False
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
        #使用多个gpu，采用并行的模式
        if len(args.gpu.split(',')) > 1:
            args.parallel = True
    
    main_worker(args)