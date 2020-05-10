import torch
import os
from PIL import Image
import random
import numpy as np
import pickle
import torchvision.transforms as transforms

from .celeba import CelebADataset
from .base_dataset import BaseDataset
from .face import FaceDataset


def create_dataloader(opt):  # 根据配置创建一个DataLoader，有两种：FaceDataset || CelebADataset
    return DataLoader(opt)


class DataLoader:  # emotion和celeb都使用CelebA; 自定义的数据集需要重载BaseDataSet！
    def __init__(self, opt):
        self.initialize(opt)

    def initialize(self, opt):
        self.opt = opt
        self.dataset = self.create_dataset()
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batch_size,
            shuffle=not opt.serial_batches,  # 是否随机打乱数据
            num_workers=int(opt.n_threads)  # 线程数
        )

    def create_dataset(self):  # 创建指定的数据集
        loaded_dataset = os.path.basename(self.opt.data_root.strip('/')).lower()
        if 'celeba' in loaded_dataset or 'emotion' in loaded_dataset:  # celebA 和 emotionNet 数据集使用 CelebADataset
            dataset = CelebADataset(self.opt)
        elif 'face' in loaded_dataset:  # 自定义测试或训练，使用face数据集（自定义）
            dataset = FaceDataset(self.opt)
        else:
            dataset = BaseDataset(self.opt)
        return dataset

    def name(self):
        return self.dataset.name() + "_Loader"

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)  # max_dataset_size的限制数据集中的图片数量

    def __iter__(self):  # 为了施加max_dataset_size这个限制
        for i, data in enumerate(self.dataloader):
            if i * self.opt.batch_size >= self.opt.max_dataset_size:  # max_dataset_size的限制数据集中的图片数量
                break
            yield data
