import torch
import os
from PIL import Image
import pickle
import torchvision.transforms as transforms


class BaseDataset(torch.utils.data.Dataset):
    """ 数据集基类 """
    def __init__(self, opt):
        super(BaseDataset, self).__init__()
        self._initialize(opt)

    def _initialize(self, opt):
        """ 初始化所有数据集的公操作，此方法将被继承 """
        self.opt = opt
        self.imgs_dir = os.path.join(self.opt.data_root, self.opt.imgs_dir)  # 数据集路径+图片路径
        self.is_train = (self.opt.mode == "train")

        filename = self.opt.train_csv if self.is_train else self.opt.test_csv  # 根据模式选择对应的id文件（csv）
        self.imgs_name_file = os.path.join(self.opt.data_root, filename)  # 拼接成id文件的完全路径
        self.imgs_path = self.make_dataset()  # 返回包含所有图像完全路径的list

        aus_pkl_file = os.path.join(self.opt.data_root, self.opt.aus_pkl)  # 拼接成完全路径
        self.aus_dict = self.load_dict(aus_pkl_file)  # 加载AU向量字典

        self.img2tensor = self.img_transformer()  # 图像预处理（整形、裁剪、翻转），image -> tensor

    def name(self):
        """ 数据集名 """
        return os.path.basename(self.opt.data_root.strip('/'))
        
    def make_dataset(self):
        """ 创建数据集，重载这个方法 """
        return None

    def load_dict(self, pkl_path):
        """ 加载pkl文件，此方法将被继承 """
        saved_dict = {}
        with open(pkl_path, 'rb') as f:  # 打开pickle文件必需以二进制形式
            saved_dict = pickle.load(f, encoding='latin1')
        return saved_dict

    def get_img_by_path(self, img_path):
        """ 根据指定路径打开单张图片（返回Image），此方法将被继承 """
        assert os.path.isfile(img_path), "Cannot find image file: %s" % img_path
        img_type = 'L' if self.opt.img_nc == 1 else 'RGB'  # L-灰度图，RGB-三通道
        return Image.open(img_path).convert(img_type)  # 返回打开并转化后的图片

    def get_aus_by_path(self, img_path):
        """ 根据指定路径加载AUs，重载这个方法 """
        return None

    def img_transformer(self):
        """ 图像预处理（整形、裁剪、翻转），此方法将被继承 """
        transform_list = []  # 处理后的图像
        if self.opt.resize_or_crop == 'resize_and_crop':  # 整形+裁剪
            transform_list.append(transforms.Resize([self.opt.load_size, self.opt.load_size], Image.BICUBIC))
            transform_list.append(transforms.RandomCrop(self.opt.final_size))
        elif self.opt.resize_or_crop == 'crop':  # 仅裁剪
            transform_list.append(transforms.RandomCrop(self.opt.final_size))
        elif self.opt.resize_or_crop == 'none':
            pass
            # transform_list.append(transforms.Lambda(lambda image: image))  #  自定义图像处理操作，这里直接返回原图
        else:
            raise ValueError("--resize_or_crop %s is not a valid option." % self.opt.resize_or_crop)

        if self.is_train and (not self.opt.no_flip):
            transform_list.append(transforms.RandomHorizontalFlip())  # 随机水平翻转

        transform_list.append(transforms.ToTensor())  # 转化为Tensor
        transform_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))  # 正则化
        img2tensor = transforms.Compose(transform_list)  # 打包transform
        return img2tensor

    def __len__(self):
        """ 重载魔术方法，返回图像数量 """
        return len(self.imgs_path)

