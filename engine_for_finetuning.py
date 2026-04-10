# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 08:39:52 2024

@author: Administrator
"""

import torch
import codecs
from sklearn import metrics
from sklearn.utils import resample
import os
from sklearn.metrics import precision_score, accuracy_score, average_precision_score, roc_auc_score, precision_recall_curve, auc, f1_score, recall_score, confusion_matrix
from sklearn.preprocessing import label_binarize
import numpy as np
import time
from tqdm import tqdm
import shutil
from collections import Counter
import torch.nn.functional as F

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, *meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def print(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = min(max(topk), output.size()[1])
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:min(k, maxk)].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]

def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs 
    else:
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr

def calculate_macro_accuracy(probs, targets, threshold=0.5):
    predictions = (probs > threshold).float()
    
    num_classes = probs.shape[1]
    per_class_accuracy = []
    
    for i in range(num_classes):
        correct = ((predictions[:, i] == targets[:, i])).sum().item()
        total = targets.shape[0]
        class_acc = correct / total
        per_class_accuracy.append(class_acc)
    
    macro_accuracy = sum(per_class_accuracy) / num_classes
    
    return macro_accuracy, per_class_accuracy
    
def train_one_epoch(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    step_loss = []
    progress = ProgressMeter(len(train_loader), batch_time, data_time, losses, top1,
                            prefix="Epoch: [{}]".format(epoch))

    topk=5
    if train_loader.dataset.n_classes < 5:
        topk = train_loader.dataset.n_classes
    # switch to train mode
    model.train()
    end = time.time()
    optimizer.zero_grad()
 
    for i, (images, targets, _) in tqdm(enumerate(train_loader), desc='train', leave=False):
        
        adjust_learning_rate(optimizer, i / len(train_loader) + epoch, args)
        # measure data loading time
        data_time.update(time.time() - end)
        if args.gpu :
            images, targets = images.cuda(), targets.cuda()

        # compute output
        output = model(images)
        loss = criterion(output, targets)
        
        # measure accuracy and record loss
        if args.multi_label:
            acc5 = 1.0
            probs = torch.sigmoid(output)
            acc1, _ = calculate_macro_accuracy(probs, targets)
        else:
            acc1, acc5 = accuracy(output, targets, topk=(1, topk))
            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))
        losses.update(loss.item(), images.size(0))
        step_loss.append(loss.item())

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.print(i)

    return losses, step_loss, top1.avg
 
@torch.no_grad()    
def evaluate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(len(val_loader), batch_time, losses, top1,
                             prefix='')

    # switch to evaluate mode
    model.eval()
    
    n_classes = len(val_loader.dataset.classes)
    preds, labels, scores, features = [], [], [], []
    begin_time = time.time()
    for i, (images, targets, _) in tqdm(enumerate(val_loader), desc='eval', leave=False):
        tmp_time = time.time()
        if args.gpu :
            images, targets = images.cuda(), targets.cuda()

        # compute output
        with torch.no_grad():
            output = model(images)
            loss = criterion(output, targets)
        # torch.cuda.synchronize()

        # measure accuracy and record loss
        score = torch.softmax(output, dim=1)
        predict = torch.max(output, dim=1)[1]

        acc1 = accuracy(output, targets, topk=(1,))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0].item(), images.size(0))
        labels.append(targets)
        scores.append(score)
        preds.append(predict)

        # measure elapsed time
        batch_time.update(time.time() - tmp_time)
        

        if i % args.print_freq == 0:
            progress.print(i)
    end_time = time.time()   
     
    true_labels = torch.cat(labels, dim=0).cpu().numpy()
    scores = torch.cat(scores, dim=0).cpu().numpy()
    predicts = torch.cat(preds, dim=0).cpu().numpy()

    if n_classes >= 3:
        auroc = roc_auc_score(true_labels_onehot, scores, multi_class='ovr' , average='macro')
    else:
        true_labels_onehot = np.eye(n_classes)[true_labels]
        auroc = roc_auc_score(true_labels_onehot, scores , average=None)
        precisions = calculate_precision(true_labels, predicts, 2)

    # TODO: this should also be done with the ProgressMeter
    print('Test Time {}s , Acc@1 {:.4f}, AUROC  {:.4f}'.format(int(end_time - begin_time) ,top1.avg, auroc))

    return top1.avg ,losses ,auroc

def calculate_accuracy(y_true, y_pred):
    # 计算样本数和正确分类样本数
    n = len(y_true)
    k = np.sum(y_true == y_pred)

    # 计算准确率
    accuracy = k / n

    return accuracy

def calculate_accs(true_labels, predicts, n_classes):

    accs = []
    for i in range(n_classes):
        # 设置当前类别为正类，其余类别为负类
        y_true = np.where(true_labels == i, 1, 0)
        y_pred = np.where(predicts == i, 1, 0)
        
        # 计算当前类别的准确率
        accuracy = accuracy_score(y_true, y_pred)
        accs.append(accuracy)

    return accs

def calculate_sensitivity(y_true, y_pred):

    confusion_matrix_original = confusion_matrix(y_true, y_pred)

    TP = np.diag(confusion_matrix_original)
    FN = np.sum(confusion_matrix_original, axis=1) - TP
    sensitivities = TP / (TP + FN)

    return sensitivities

def calculate_specificity(y_true, y_pred):
    specificities = []

    confusion_matrix_original = confusion_matrix(y_true, y_pred)
    TN = np.sum(confusion_matrix_original) - np.sum(confusion_matrix_original, axis=0) - np.sum(confusion_matrix_original, axis=1) + np.diag(confusion_matrix_original)
    FP = np.sum(confusion_matrix_original, axis=0) - np.diag(confusion_matrix_original)
    specificities = TN / (TN + FP)

    return specificities

def calculate_precision(y_true, y_pred, n_classes):
    
    # 将真实标签进行one-hot编码
    true_labels_onehot = label_binarize(y_true, classes=np.arange(n_classes))
    # 计算每个类别的 Precision
    precisions = precision_score(true_labels_onehot, label_binarize(y_pred, classes=range(n_classes)), average=None, zero_division=1)
    
    return precisions

def cal_all_metrics(true_labels, scores, predicts, n_classes, id2classes):

    if n_classes > 2:
        # 将真实标签进行one-hot编码
        true_labels_onehot = label_binarize(true_labels, classes=np.arange(n_classes))
    else:
        true_labels_onehot = true_labels_onehot = np.eye(n_classes)[true_labels]
    
    ####################################################
    sensitivities = calculate_sensitivity(true_labels, predicts)
    specificities = calculate_specificity(true_labels, predicts)
    precisions = calculate_precision(true_labels, predicts, n_classes)
    accs = calculate_accs(true_labels, predicts, n_classes)
    if n_classes >= 3:
        aucs = roc_auc_score(true_labels_onehot, scores, multi_class='ovr' , average=None)
    else:
        aucs = roc_auc_score(true_labels_onehot, scores , average=None)
    
    aupr_scores = []
    if n_classes >= 3:
        for i in range(n_classes):
            precision, recall, _ = precision_recall_curve(true_labels_onehot[:, i], scores[:, i])
            aupr = auc(recall, precision)
            aupr_scores.append(aupr)
        macro_aupr = np.mean(list(aupr_scores))
    else:
        precision, recall, _ = precision_recall_curve(true_labels_onehot[:, 0], scores[:, 0])
        macro_aupr = auc(recall, precision)
        aupr_scores.append(macro_aupr)

    #计算macro-AUC的值
    if n_classes >= 3:
        macro_auc = metrics.roc_auc_score(true_labels_onehot, scores)
    else:
        #默认0类别是正例
        macro_auc = roc_auc_score(true_labels_onehot[:, 0], scores[:, 0])
    #计算 整体acc 值
    acc = calculate_accuracy(true_labels, predicts)

    report_dict = metrics.classification_report(true_labels, predicts, target_names=['{}'.format(x) for x in range(n_classes)],digits=4, labels=range(n_classes), output_dict=True)
    f1_list = [report_dict[str(i)]['f1-score'] for i in range(n_classes)]

    ##打印各指标
    for label_index in range(n_classes):
        print()
        print("Class\t", id2classes[label_index])
        print("\tSensitivity \t{:.2f}".format(sensitivities[label_index]*100))
        print("\tSpecificity \t{:.2f}".format(specificities[label_index]*100))
        print("\tPrecision \t{:.2f}".format(precisions[label_index]*100))
        print("\tAUROC\t{:.2f}".format(aucs[label_index]))
        print("\tF1\t{:.2f}".format(f1_list[label_index]))
    
    print()
    
    print("acc\t{:.4f}".format(acc))
    print("macro_auc\t{:.4f}".format(macro_auc)) 
    print("macro_aupr\t{:.4f}".format(macro_aupr))

    report = metrics.classification_report(true_labels, predicts, target_names=['{}'.format(x) for x in range(n_classes)],digits=4, labels=range(n_classes))
    confusion = confusion_matrix(true_labels, predicts)
    print(report)
    print(confusion)

    return acc, macro_auc

@torch.no_grad()    
def evaluate_sex(data_loader, criterion, model, device, args, num_class, mode):
    """Evaluate the model diff sex."""

    losses = AverageMeter('Loss', ':.4e')
    # os.makedirs(os.path.join(args.output_dir, args.task), exist_ok=True)
    
    model.eval()
    true_onehot, pred_onehot, true_labels, pred_labels, pred_softmax = [], [], [], [], []
    preds, labels, scores, sex_list = [], [], [], []
    
    for i, (images, targets, sexs) in tqdm(enumerate(data_loader), desc=mode, leave=False):
        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        target_onehot = F.one_hot(targets.to(torch.int64), num_classes=num_class)
        
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, targets)
        # output_ = nn.Softmax(dim=1)(output)
        output_ = torch.softmax(output, dim=1)
        output_label = output_.argmax(dim=1)

        scores.append(output_)
        labels.append(targets)
        preds.append(output_label)
        sex_list.extend(sexs)
        losses.update(loss.item(), images.size(0))
    
    true_labels = torch.cat(labels, dim=0).cpu().numpy()
    scores = torch.cat(scores, dim=0).cpu().numpy()
    predicts = torch.cat(preds, dim=0).cpu().numpy()
    
    sex_list = np.array(sex_list)
    male_idx = sex_list == 'Male'
    female_idx = sex_list == 'Female'

    # 转换为对应的列表
    male_true_labels = true_labels[male_idx]
    male_scores = scores[male_idx]
    male_predicts = predicts[male_idx]

    female_true_labels = true_labels[female_idx]
    female_scores = scores[female_idx]
    female_predicts = predicts[female_idx]

    acc, macro_auc = cal_all_metrics(true_labels, scores, predicts, num_class, data_loader.dataset.classes)
    
    if 'test' == mode:
        print('Male########################################################################################')
        cal_all_metrics(male_true_labels, male_scores, male_predicts, num_class, data_loader.dataset.classes)
        print('Female########################################################################################\n')
        cal_all_metrics(female_true_labels, female_scores, female_predicts, num_class, data_loader.dataset.classes)
    return acc, losses, macro_auc

def cal_all_metrics_multilabel(y_true, y_pred, id2classes, threshold=0.5):

    num_classes = len(id2classes)
    
    aucs = []
    auprs = []
    f1s = []
    accs = []

    for i in range(num_classes):
        label_name = id2classes[i]

        y_true_i = y_true[:, i]
        y_pred_i = y_pred[:, i]
        # print(y_true_i[:20])
        
        try:
            # 计算AUC
            auc = roc_auc_score(y_true_i, y_pred_i)
            aupr = average_precision_score(y_true_i, y_pred_i)
            y_pred_bin = (y_pred_i >= threshold).astype(int)
            f1 = f1_score(y_true_i, y_pred_bin)
            sensitivity = recall_score(y_true_i, y_pred_bin)

            # 先计算TN, FP
            TP = np.sum((y_true_i == 1) & (y_pred_bin == 1))
            TN = np.sum((y_true_i == 0) & (y_pred_bin == 0))
            FP = np.sum((y_true_i == 0) & (y_pred_bin == 1))
            FN = np.sum((y_true_i == 1) & (y_pred_bin == 0))
            # 计算Specificity
            if (TN + FP) > 0:
                specificity = TN / (TN + FP)
            else:
                specificity = np.nan
            
            # 计算准确率
            acc = accuracy_score(y_true_i, y_pred_bin)
            # 计算混淆矩阵（返回TN, FP, FN, TP）
            cm = confusion_matrix(y_true_i, y_pred_bin, labels=[0,1])
        except Exception as e:
            print(f"Exception：{e}")
            sensitivity = np.nan
            specificity = np.nan
            auc = np.nan
            aupr = np.nan
            f1 = np.nan
            acc = np.nan
            cm = np.nan
    
        print()
        print("Class\t", id2classes[i])
        print("\tACC \t{:.4f}".format(acc))
        print("\tSensitivity \t{:.4f}".format(sensitivity))
        print("\tSpecificity \t{:.4f}".format(specificity))
        print("\tAUROC\t{:.4f}".format(auc))
        print("\tAURRC\t{:.4f}".format(aupr))
        print("\tF1\t{:.4f}".format(f1))
        print("\tConfusion Matrix:\n", cm)
        
        aucs.append(auc)
        auprs.append(aupr)
        f1s.append(f1)
        accs.append(acc)
    
    # 计算宏平均指标，忽略NaN
    macro_auc = np.nanmean(aucs)
    macro_aupr = np.nanmean(auprs)
    macro_f1 = np.nanmean(f1s)
    macro_acc = np.nanmean(accs)
    print()
    print("macro_acc\t{:.4f}".format(macro_acc))
    print("macro_auc\t{:.4f}".format(macro_auc))
    print("macro_aupr\t{:.4f}".format(macro_aupr))
    print("macro_f1\t{:.4f}".format(macro_f1))

    return macro_acc, macro_auc


@torch.no_grad()
def evaluate_sex_multi_label(data_loader, criterion, model, device, args, num_class, mode):
    """Evaluate the model diff sex."""

    losses = AverageMeter('Loss', ':.4e')
    
    model.eval()
    preds_score, labels, sex_list = [], [], []
    
    for i, (images, targets, sexs) in tqdm(enumerate(data_loader), desc=mode, leave=False):
        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        target_onehot = F.one_hot(targets.to(torch.int64), num_classes=num_class)
        
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, targets)
            probs = torch.sigmoid(output)
        labels.append(targets.cpu().numpy())
        preds_score.append(probs.cpu().numpy())
        sex_list.extend(sexs)
        losses.update(loss.item(), images.size(0))
    # 计算指标
    y_true = np.concatenate(labels, axis=0)
    y_pred = np.concatenate(preds_score, axis=0)
    
    macro_acc, macro_auc = cal_all_metrics_multilabel(y_true, y_pred, data_loader.dataset.classes)
    
    if 'test' == mode:
        sex_list = np.array(sex_list)
        male_idx = sex_list == 'Male'
        female_idx = sex_list == 'Female'

        male_true_labels = y_true[male_idx]
        male_predicts = y_pred[male_idx]

        female_true_labels = y_true[female_idx]
        female_predicts = y_pred[female_idx]

        print('Male########################################################################################')
        cal_all_metrics_multilabel(male_true_labels, male_predicts, data_loader.dataset.classes, log_writer)
        print('Female########################################################################################')
        cal_all_metrics_multilabel(female_true_labels, female_predicts, data_loader.dataset.classes, log_writer)
    
    return macro_acc, losses, macro_auc  