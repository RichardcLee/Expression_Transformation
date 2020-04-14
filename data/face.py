from .base_dataset import BaseDataset
import os
import random
import numpy as np


class FaceDataset(BaseDataset):
    """  face Dataset """
    def __init__(self, opt):
        super(FaceDataset, self).__init__(opt)
        if self.opt.test_mode == "single_target":  # 单张目标图片模式
            self.mode = 0
            if self.opt.single_target_img == "none":
                raise Exception("You should give a path to figure out single target image!")

        elif self.opt.test_mode == "random_target":  # 随机目标图片模式
            self.mode = 1

        elif self.opt.test_mode == "pair_target":  # <源图片, 目标图片> 成对模式
            self.mode = 2
            # check here
            pass  # todo

        else:
            raise NotImplementedError("No such test mode: %s. There are 3 mode:"
                                      " [single_target|random_target|pair_target]" % self.test_mode)

    def get_aus_by_path(self, img_path):  # 通过路径(basename)获取AUs
        assert os.path.isfile(img_path), "Cannot find image file: %s" % img_path
        img_id = str(os.path.splitext(os.path.basename(img_path))[0])
        return self.aus_dict[img_id] / 5.0

    def make_dataset(self):  # 返回一个包含所有文件路径的列表
        imgs_path = []
        assert os.path.isfile(self.imgs_name_file), "%s does not exist." % self.imgs_name_file  # id csv

        with open(self.imgs_name_file, 'r') as f:
            lines = f.readlines()
            imgs_path = [os.path.join(self.imgs_dir, line.strip()) for line in lines]  # 图片完全路径列表
            imgs_path = sorted(imgs_path)
        return imgs_path

    def __getitem__(self, index):
        img_path = self.imgs_path[index]

        src_img = self.get_img_by_path(img_path)  # 加载一张源图片
        src_img_tensor = self.img2tensor(src_img)
        src_aus = self.get_aus_by_path(img_path)  # 加载源图片对应的AUs

        if self.mode == 0:  # 单张目标图片测试模式
            tar_img_path = self.opt.single_target_img  # 指定唯一的目标图片
            tar_img = self.get_img_by_path(tar_img_path)
            tar_img_tensor = self.img2tensor(tar_img)
            tar_aus = self.get_aus_by_path(tar_img_path)  # 加载目标图片对应的AUs，不要忘记计算该图片的AUs!!!

        elif self.mode == 1:  # 随机目标图片测试模式
            tar_img_path = random.choice(self.imgs_path)  # 在测试集中随机选择一个图片作为目标图片
            tar_img = self.get_img_by_path(tar_img_path)  # 加载目标图片
            tar_img_tensor = self.img2tensor(tar_img)
            tar_aus = self.get_aus_by_path(tar_img_path)  # 加载目标图片对应的AUs

        elif self.mode == 2:  # 配对测试模式，暂不实现
            pass  # todo

        else:
            pass

        if self.is_train and self.opt.aus_noise:
            tar_aus = tar_aus + np.random.uniform(-0.1, 0.1, tar_aus.shape)  # 添加噪声

        # 封装并返回，用于调试和测试
        data_dict = {'src_img': src_img_tensor, 'src_aus': src_aus, 'src_path': img_path, 'tar_img': tar_img_tensor,
                     'tar_aus': tar_aus, 'tar_path': tar_img_path}
        return data_dict
