import os
from torchvision import datasets, transforms
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import random
import numpy as np
from .datasets import register

@register('sex-baias-multi-label')
class SexBaiasMultiLabelDataset(Dataset):

    def parse_label_info(self, label_info_dir):

        df = pd.read_excel(label_info_dir, header=0)
        
        paient_id_list = df['ID'].tolist()
        image_name_list = df['Image'].tolist()
        sex_ori_list = df['Sex'].tolist()
        age_list = df['Age'].tolist()

        outer_test = False
        if 'partition' in df:
            partition_tag_list = df['partition'].tolist()
        else:
            partition_tag_list = None
            outer_test = True
        
        if self.partition_tag == 'outer_test':
            outer_test = True

        data_total_len = len(paient_id_list)

        #Make sure the length is consistent
        assert len(paient_id_list) == data_total_len and len(image_name_list) == data_total_len
        assert len(sex_ori_list) == data_total_len and len(age_list) == data_total_len
        self.sex_list = []
        for index in range(data_total_len):
            
            paient_id = paient_id_list[index]
            image_name = image_name_list[index].strip()
            sex = sex_ori_list[index].strip()
            tmp_labels = df.iloc[index][self.classes].values.astype(float)
            age = age_list[index]
            
            if not outer_test:
                tmp_partition_tag = partition_tag_list[index].strip()
                if not tmp_partition_tag == self.partition_tag:
                    continue
            
            if self.sex_tag:
                if sex != self.sex_tag:
                    continue
            
            img_dir = os.path.join(self.rootpath, 'Images', image_name)

            if not os.path.exists(img_dir):
                # print('image path:{} is not exist.'.format(img_dir))
                continue
            self.sex_list.append(sex)
            self.filepaths.append(img_dir)
            self.labels.append(tmp_labels)


    def __init__(self, root_path, tag_path, input_size, label_columns, partition_tag='train', sex_tag=None,
                color_jitter = None, aa = 'rand-m9-mstd0.5-inc1', reprob = 0.25, remode = 'pixel', recount = 1):

        print('Dataset: tag_path:{}, partition_tag:{}, sex_tag:{}'.format(tag_path, partition_tag, sex_tag))

        self.filepaths = []
        self.labels = []
        self.classes = label_columns
        self.id2classes = {}
        self.rootpath = root_path
        self.name = os.path.basename(root_path)
        self.partition_tag = partition_tag
        self.sex_tag = sex_tag

        label_file_dir = tag_path
        self.parse_label_info(label_file_dir)
        self.n_classes = len(self.classes)
        self.transform = build_transform(partition_tag, input_size, color_jitter, aa, reprob, remode, recount)

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, i):
        img = Image.open(self.filepaths[i]).convert('RGB')

        return self.transform(img), self.labels[i], self.sex_list[i]

@register('sex-baias')
class SexBaiasDataset(Dataset):

    def print_gender_distribution(self):
        """
        统计并打印每个类别中男女性别的样本个数
        """
        print("\n" + "="*60)
        print("类别性别分布统计")
        print("="*60)
        
        # 统计每个类别中男女性别的样本个数
        class_gender_stats = {}
        for label, sex in zip(self.labels, self.sex_list):
            if label not in class_gender_stats:
                class_gender_stats[label] = {'Male': 0, 'Female': 0}
            if sex == 'Male':
                class_gender_stats[label]['Male'] += 1
            elif sex == 'Female':
                class_gender_stats[label]['Female'] += 1
        
        # 打印统计结果
        for label in sorted(class_gender_stats.keys()):
            male_count = class_gender_stats[label]['Male']
            female_count = class_gender_stats[label]['Female']
            total_count = male_count + female_count
            male_ratio = male_count / total_count * 100 if total_count > 0 else 0
            female_ratio = female_count / total_count * 100 if total_count > 0 else 0
            
            print(f"\n类别: {label}")
            print(f"  男性: {male_count:4d} 例 ({male_ratio:5.2f}%)")
            print(f"  女性: {female_count:4d} 例 ({female_ratio:5.2f}%)")
            print(f"  总计: {total_count:4d} 例")
        
        print("\n" + "="*60)
        print(f"总样本数: {len(self.labels)}")
        print(f"总男性样本: {self.sex_list.count('Male')} 例")
        print(f"总女性样本: {self.sex_list.count('Female')} 例")
        print("="*60)
        
        return class_gender_stats

    def parse_label_info(self, label_info_dir):

        df = pd.read_excel(label_info_dir, header=0)
        paient_id_list = df['ID'].tolist()
        image_name_list = df['Image'].tolist()
        label_list = df['Label'].tolist()
        sex_ori_list = df['Sex'].tolist()
        age_list = df['Age'].tolist()
        
        self.classes = sorted(df['Label'].unique())

        outer_test = False
        if 'partition' in df:
            partition_tag_list = df['partition'].tolist()
        else:
            partition_tag_list = None
            outer_test = True
        
        if self.partition_tag == 'outer_test':
            outer_test = True

        data_total_len = len(paient_id_list)

        label_sex_buckets = {}
        valid_indices = []
        for index in range(data_total_len):
            # partition filter
            if not outer_test:
                tmp_partition_tag = partition_tag_list[index].strip()
                if not tmp_partition_tag == self.partition_tag:
                    continue
        
            sex = str(sex_ori_list[index]).strip()
            # sex filter
            if self.sex_tag and sex != self.sex_tag:
                    continue
            label = str(label_list[index]).strip()

            valid_indices.append(index)
            bucket = label_sex_buckets.setdefault(label, {'Male': [], 'Female': []})
            if sex == 'Male':
                bucket['Male'].append(index)
            elif sex == 'Female':
                bucket['Female'].append(index)

        # 进行原有的逐行处理（保持其他逻辑不变）
        self.sex_list = []
        for index in valid_indices:
            
            paient_id = paient_id_list[index]
            image_name = image_name_list[index].strip()
            sex = sex_ori_list[index].strip()
            label_name = label_list[index].strip()
            age = age_list[index]
            
            img_dir = os.path.join(self.rootpath, 'Images', image_name)

            if not os.path.exists(img_dir):
                print('image path:{} is not exist.'.format(img_dir))
                continue
            self.sex_list.append(sex)
            self.filepaths.append(img_dir)
            self.labels.append(label_name)
        # self.gender_stats = self.print_gender_distribution()


    def __init__(self, root_path, tag_path, input_size, label_columns, partition_tag='train', sex_tag=None,
                color_jitter = None, aa = 'rand-m9-mstd0.5-inc1', reprob = 0.25, remode = 'pixel', recount = 1):

        print('Dataset: tag_path:{}, partition_tag:{}, sex_tag:{}'.format(tag_path, partition_tag, sex_tag))

        self.filepaths = []
        self.labels = []
        self.classes = []
        self.id2classes = {}
        self.rootpath = root_path
        self.name = os.path.basename(root_path)
        self.partition_tag = partition_tag
        self.sex_tag = sex_tag

        label_file_dir = tag_path
        # 以各个类别的最小值进行类别平衡
        self.parse_label_info(label_file_dir)
        self.n_classes = len(self.classes)
        
        self.transform = build_transform(partition_tag, input_size, color_jitter, aa, reprob, remode, recount)


    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, i):
        img = Image.open(self.filepaths[i]).convert('RGB')

        return self.transform(img), self.classes.index(self.labels[i]), self.sex_list[i]


def build_transform(is_train, input_size, color_jitter, aa, reprob, remode, recount):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train == 'train':
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=input_size,
            is_training=True,
            color_jitter=color_jitter,
            auto_augment=aa,
            interpolation='bicubic',
            re_prob=reprob,
            re_mode=remode,
            re_count=recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
    )
    t.append(transforms.CenterCrop(input_size))
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)
