import os
import shutil
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import MultiStepLR

from pathlib import Path
import os
import glob
import torch.distributed as dist


_log_path = None

def set_log_path(path):
    global _log_path
    _log_path = path

def log(obj, filename='log.txt'):
    print(obj)
    if _log_path is not None:
        with open(os.path.join(_log_path, filename), 'a') as f:
            print(obj, file=f)


class Averager():

    def __init__(self):
        self.n = 0.0
        self.v = 0.0

    def add(self, v, n=1.0):
        self.v = (self.v * self.n + v * n) / (self.n + n)
        self.n += n

    def item(self):
        return self.v
    
    def count(self):
        return self.n


class Timer():

    def __init__(self):
        self.v = time.time()

    def s(self):
        self.v = time.time()

    def t(self):
        return time.time() - self.v


def set_gpu(gpu):
    print('set gpu:', gpu)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu


def ensure_path(path, remove=True):
    basename = os.path.basename(path.rstrip('/'))
    if os.path.exists(path):
        if remove and (basename.startswith('_')
                or input('{} exists, remove? ([y]/n): '.format(path)) != 'n'):
            shutil.rmtree(path)
            os.makedirs(path)
    else:
        os.makedirs(path)


def time_str(t):
    if t >= 3600:
        return '{:.1f}h'.format(t / 3600)
    if t >= 60:
        return '{:.1f}m'.format(t / 60)
    return '{:.1f}s'.format(t)


def compute_logits(feat, proto, metric='dot', temp=1.0, nearest_point = False):
    if nearest_point:
        assert feat.dim() + 1 == proto.dim()
    else:
        assert feat.dim() == proto.dim()

    if feat.dim() == 2:
        if metric == 'dot':
            logits = torch.mm(feat, proto.t())
        elif metric == 'cos':
            logits = torch.mm(F.normalize(feat, dim=-1),
                              F.normalize(proto, dim=-1).t())
        elif metric == 'sqr':
            logits = -(feat.unsqueeze(1) -
                       proto.unsqueeze(0)).pow(2).sum(dim=-1)

    elif feat.dim() == 3:
        if metric == 'dot':
            if nearest_point:
                #实现方式3：优雅的方式
                pre_logits = torch.einsum('bik,bjlk->jbil', feat, proto)
                tmp_distance, _ = torch.max(pre_logits, dim=-1, keepdim=True)
                logits = tmp_distance.squeeze(-1).permute(1, 2, 0)

                # #维度变化为[classes_num,batch,shot,feature]
                # proto = proto.permute(1, 0, 2, 3)
                # class_dim = proto.size(0)
                # #实现方式1：复制feat使其维度保持和proto一致
                # expand_feat = torch.stack([feat for _ in range(class_dim)], dim=0)
                # print('step5- expand_feat shape:{}'.format(expand_feat.shape))
                # pre_logits = torch.bmm(expand_feat,proto.permute(0, 1, 3, 2))
                # print('step6- pre_logits shape:{}'.format(pre_logits.shape))
                # tmp_distance, _ = torch.max(pre_logits, dim=-1, keepdim=True)
                # logits = tmp_distance.squeeze(-1).permute(1, 2, 0)
                # #实现方式2：遍历（暴力求解,慢）
                # max_distance_tensor_list = []
                # #遍历处理每个类别
                # for i in range(class_dim):
                #     sigle_class_shot_tensor = proto[i]
                #     # print('step5- sigle_class_shot_tensor shape:{}'.format(sigle_class_shot_tensor.shape))
                #     #计算queryset与某个分类每张图片的距离
                #     sigle_class_shot_distance = torch.bmm(feat,sigle_class_shot_tensor.permute(0, 2, 1))
                #     # print('step6- tmp_shot shape:{}'.format(sigle_class_shot_distance.shape))
                #     #取max作为最终距离
                #     tmp_distance, _ = torch.max(sigle_class_shot_distance, dim=-1, keepdim=True)
                #     # print('step7- tmp_distance shape:{}'.format(sigle_class_shot_distance.shape))
                #     max_distance_tensor_list.append(tmp_distance)
                #     # print('step8- tmp_max_distance shape:{}'.format(tmp_distance.shape))
                # logits = torch.cat(max_distance_tensor_list, dim=-1)
            else:
                logits = torch.bmm(feat, proto.permute(0, 2, 1))
            # print('step9- logits shape:{}'.format(logits.shape))
        elif metric == 'cos':
            
            logits = torch.bmm(F.normalize(feat, dim=-1),
                            F.normalize(proto, dim=-1).permute(0, 2, 1))
        elif metric == 'sqr':
            logits = -(feat.unsqueeze(2) -
                       proto.unsqueeze(1)).pow(2).sum(dim=-1)

    return logits * temp


def compute_acc(logits, label, reduction='mean'):
    ret = (torch.argmax(logits, dim=1) == label).float()
    if reduction == 'none':
        return ret.detach()
    elif reduction == 'mean':
        return ret.mean().item()


def compute_n_params(model, return_str=True):
    tot = 0
    for p in model.parameters():
        w = 1
        for x in p.shape:
            w *= x
        tot += w
    if return_str:
        if tot >= 1e6:
            return '{:.1f}M'.format(tot / 1e6)
        else:
            return '{:.1f}K'.format(tot / 1e3)
    else:
        return tot


def make_optimizer(params, name, lr, weight_decay=None, milestones=None):
    if weight_decay is None:
        weight_decay = 0.
    if name == 'sgd':
        optimizer = SGD(params, lr, momentum=0.9, weight_decay=weight_decay)
    elif name == 'adam':
        optimizer = Adam(params, lr, weight_decay=weight_decay)
    if milestones:
        lr_scheduler = MultiStepLR(optimizer, milestones)
    else:
        lr_scheduler = None
    return optimizer, lr_scheduler


def visualize_dataset(dataset, name, writer, n_samples=16):
    demo = []
    for i in np.random.choice(len(dataset), n_samples):
        demo.append(dataset.convert_raw(dataset[i][0]))
    writer.add_images('visualize_' + name, torch.stack(demo))
    writer.flush()


def freeze_bn(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()

def load_state_dict_from_pretrained(model, args):
    print("load pretrained model from:", args.fine_tuning)
    checkpoint = torch.load(args.fine_tuning, map_location=torch.device('cpu'))
    print('checkpoint keys:', checkpoint.keys())
    #特殊情况加载
    if 'vision_FM' in args.arch:
        print('begin to loda dino pretrained model.')
        # print(checkpoint['teacher'].keys())
        new_model_dict = {k.replace('module.','').replace('backbone.','encoder.') : v for k, v in checkpoint['teacher'].items()}
    else:
        for model_key in args.model_key.split('|'):
           if model_key in checkpoint:
               new_model_dict = checkpoint[model_key]
               print("Load state_dict by model_key = %s" % model_key)
               break
    load_state_dict(model, new_model_dict)
    #加载模型文件中的其他信息
    if args.store_metrics:
        args.train_loss_list = checkpoint['train_loss']
        args.test_loss_list = checkpoint['val_loss']
        args.aucs_list = checkpoint['val_aucs']
        args.train_step_losses = checkpoint['train_step_loss']

def load_state_dict_from_pretrained_path(arch, model, fine_tuning):
    
    #兼容使用
    model_keys = ['model','module']

    print("load pretrained model from:", fine_tuning)
    checkpoint = torch.load(fine_tuning, map_location=torch.device('cpu'))
    print('checkpoint keys:', checkpoint.keys())
    #特殊情况加载
    if 'dino' in arch:
        print('begin to loda dino pretrained model.')
        new_model_dict = {k.replace('module.','').replace('backbone.','') : v for k, v in checkpoint['student'].items()}
    else:
        for model_key in model_keys:
           if model_key in checkpoint:
               new_model_dict = checkpoint[model_key]
               print("Load state_dict by model_key = %s" % model_key)
               break
    load_state_dict(model, new_model_dict)


def load_state_dict(model, state_dict, prefix='', ignore_missing="relative_position_index"):
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix=prefix)

    warn_missing_keys = []
    ignore_missing_keys = []
    for key in missing_keys:
        keep_flag = True
        for ignore_key in ignore_missing.split('|'):
            if ignore_key in key:
                keep_flag = False
                break
        if keep_flag:
            warn_missing_keys.append(key)
        else:
            ignore_missing_keys.append(key)

    missing_keys = warn_missing_keys

    if len(missing_keys) > 0:
        print("Weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, missing_keys))
    if len(unexpected_keys) > 0:
        print("Weights from pretrained model not used in {}: {}".format(
            model.__class__.__name__, unexpected_keys))
    if len(ignore_missing_keys) > 0:
        print("Ignored weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, ignore_missing_keys))
    if len(error_msgs) > 0:
        print('\n'.join(error_msgs))

#分离编码层和分类层的参数，微调的时候可能对两部分分别设置学习率
def splits_encoder_classifier_param(arch, model):
    if arch.find('alexnet') != -1:
        fine_tune_parameters =model.classifier[6].parameters()
    elif arch.find('inception_v3') != -1 or arch.find('xception') != -1 or arch.find('resnet') != -1:
        fine_tune_parameters = model.module.fc.parameters()
    elif arch.find('densenet') != -1 or arch.find('inceptionresnet') != -1:
        fine_tune_parameters = model.module.classifier.parameters()
    elif arch.find('convnext') != -1:
        fine_tune_parameters = model.module.head.fc.parameters()
    elif arch.find('RepVGG') != -1:
        fine_tune_parameters = model.module.linear.parameters()
    elif (arch.find('Transform_large') != -1) or arch.find('Transform_base') != -1:
        fine_tune_parameters = model.module.head.parameters()
    elif arch.find('pre_fusion') != -1:
        fine_tune_parameters = model.module.model.head.parameters()
    elif arch.find('vit_large') != -1 or arch.find('vit_base') != -1  or arch.find('vit_small') != -1:
        fine_tune_parameters = model.module.head.parameters()
    else:
        print('### 模型结构:{},待补充。 ###'.format(arch))
        exit(-1) 

    ignored_params = list(map(id, fine_tune_parameters))
    base_params = filter(lambda p: id(p) not in ignored_params,
                             model.module.parameters())
    return base_params, fine_tune_parameters

def auto_load_model(args, model, model_without_ddp, optimizer):
    output_dir = Path(args.model_store_dir)
    
    # torch.amp
    if args.auto_resume and len(args.resume) == 0:
        all_checkpoints = glob.glob(os.path.join(output_dir, 'checkpoint.pth'))
        if len(all_checkpoints) > 0:
            args.resume = os.path.join(output_dir, 'checkpoint.pth')
        else:
            all_checkpoints = glob.glob(os.path.join(output_dir, 'checkpoint-*.pth'))
            latest_ckpt = -1
            for ckpt in all_checkpoints:
                t = ckpt.split('-')[-1].split('.')[0]
                if t.isdigit():
                    latest_ckpt = max(int(t), latest_ckpt)
            if latest_ckpt >= 0:
                args.resume = os.path.join(output_dir, 'checkpoint-%d.pth' % latest_ckpt)
        print("Auto resume checkpoint: %s" % args.resume)

    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model']) # strict: bool=True, , strict=False
        print("Resume checkpoint %s" % args.resume)
        if 'optimizer' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            print(f"Resume checkpoint at epoch {checkpoint['epoch']}")
            args.start_epoch = checkpoint['epoch'] + 1
            print("With optim & sched!")

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0