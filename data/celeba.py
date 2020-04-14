from .base_dataset import BaseDataset
import os
import random
import numpy as np


class CelebADataset(BaseDataset):
    """  CelebA Dataset 供训练使用"""
    def __init__(self, opt):
        super(CelebADataset, self).__init__(opt)

    def get_aus_by_path(self, img_path):  # 通过路径(basename)获取AUs
        assert os.path.isfile(img_path), "Cannot find image file: %s" % img_path
        # basename中包含扩展名，这里splitext[0]是为了去掉拓展名
        img_id = str(os.path.splitext(os.path.basename(img_path))[0])
        return self.aus_dict[img_id] / 5.0  # [0,5] norm to [0,1]

    def make_dataset(self):  # 返回一个包含所有文件路径的列表
        imgs_path = []
        assert os.path.isfile(self.imgs_name_file), "%s does not exist." % self.imgs_name_file  # 检查图片id文件是否存在

        with open(self.imgs_name_file, 'r') as f:
            lines = f.readlines()
            imgs_path = [os.path.join(self.imgs_dir, line.strip()) for line in lines]  # 图片完全路径列表
            imgs_path = sorted(imgs_path)
        return imgs_path

    def __getitem__(self, index):  # 重载魔术方法，返回一个封装好的训练条目
        img_path = self.imgs_path[index]

        src_img = self.get_img_by_path(img_path)  # 加载一张源图片
        src_img_tensor = self.img2tensor(src_img)
        src_aus = self.get_aus_by_path(img_path)  # 加载源图片对应的AUs

        tar_img_path = random.choice(self.imgs_path)  # 随机选择一个图片作为目标图片
        tar_img = self.get_img_by_path(tar_img_path)  # 加载目标图片
        tar_img_tensor = self.img2tensor(tar_img)
        tar_aus = self.get_aus_by_path(tar_img_path)  # 加载目标图片对应的AUs

        if self.is_train and self.opt.aus_noise:
            tar_aus = tar_aus + np.random.uniform(-0.1, 0.1, tar_aus.shape)  # 添加噪声

        # 封装并返回，用于调试和测试
        data_dict = {'src_img': src_img_tensor, 'src_aus': src_aus, 'src_path': img_path, 'tar_img': tar_img_tensor,
                     'tar_aus': tar_aus, 'tar_path': tar_img_path}
        return data_dict
