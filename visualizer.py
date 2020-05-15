import numpy as np
import torch
from PIL import Image
import os
from matplotlib import pyplot as plt
import re


class Visualizer(object):
    def __init__(self, opt):
        super(Visualizer, self).__init__()
        self.initialize(opt)

    def initialize(self, opt):
        self.opt = opt
        self.losses = {
            "dis_fake": [],  # WGAN-GP对抗损失第二项，值越大越好（正值）
            "dis_real": [],  # WGAN-GP对抗损失第一项，值越小越好（负值）
            "dis_real_aus": [],  # 条件表情损失第二项
            "gen_rec": [],  # 循环一致性损失
            'dis': [],  # 生成器损失
            'gen': [],  # 判别器损失
            "total": []
        }
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    # 绘制当前损失波动图、可视化当前模型训练中间图
    def plot(self, plot_dict):  # 绘制方法
        img_dict = plot_dict['img']
        # 可视化训练过程中的效果
        for name, img in img_dict.items():
            tmp = self.numpy2im(img.cpu().detach().float().numpy()[0])  # 注意一个batch n张图，这里只选择一张即可
            path = os.path.join(plot_dict['visual_path'], name+'.jpg')
            tmp.save(path)
            tmp.close()
        # 可视化损失波动
        self._plot_loss(plot_dict['visual_path'])

    def _plot_loss(self, visual_path):
        plt.figure(dpi=120)
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.4, hspace=0.45)  # 调整子图间距
        xy = ["321", "322", "323", "324", '325', '326']
        widths = [0.09, 0.09, 0.10, 0.15, 0.09, 0.09]
        names = ['adversarial loss 2', 'adversarial loss 1', 'condition loss', 'cycle consistency loss',
                  'dis loss', 'gen loss', 'total loss']
        idx = 0
        step = [i for i in range(len(self.losses["dis_fake"]))]
        fontsize = 10
        for name, val in self.losses.items():
            if idx == 6:
                continue
            plt.subplot(xy[idx])
            plt.title(names[idx], fontsize=fontsize + 2)
            plt.plot(step[::], val[::], linewidth=widths[idx], color='k')  # label=labels[idx]
            # plt.xlabel("step", fontsize=fontsize - 1)
            # plt.ylabel("loss value", fontsize=fontsize - 1)
            # 设置刻度字体大小
            plt.xticks(fontsize=fontsize - 1)
            plt.yticks(fontsize=fontsize - 1)
            idx += 1
        plt.savefig(os.path.join(visual_path, 'losses.jpg'))
        plt.close()

        fontsize = 20
        plt.figure(dpi=80)
        plt.title(names[-1], fontsize=fontsize+6)
        plt.plot(step[::], self.losses['total'][::], linewidth=0.2, color='k')
        # plt.xlabel("step", fontsize=fontsize - 6)
        # plt.ylabel("loss value", fontsize=fontsize - 6)
        # 设置刻度字体大小
        plt.xticks(fontsize=fontsize - 6)
        plt.yticks(fontsize=fontsize - 1)
        plt.savefig(os.path.join(visual_path, "total_loss.jpg"))
        plt.close()

    def print_losses_info(self, info_dict):  # 打印loss -> cmd || log
        msg = '[{}][Epoch: {:0>3}/{:0>3}; Images: {:0>4}/{:0>4}; Time: {:.3f}s/Batch({}); LR: {:.7f}] '.format(
                self.opt.name, 
                info_dict['epoch'],
                info_dict['epoch_len'], 
                info_dict['epoch_steps'], 
                info_dict['epoch_steps_len'], 
                info_dict['step_time'], 
                self.opt.batch_size, 
                info_dict['cur_lr'])
        # {}是格式化，:0>3，是用0占位(:)，右对齐(>)，宽度为3

        for k, v in info_dict['losses'].items():  # 不同loss分开显示
            msg += '| {}: {:.4f} '.format(k, v)
            self.losses[k].append(v)  # 记录下所有损失
        self.losses['total'].append(10*self.losses["gen_rec"][-1] + self.losses["dis_fake"][-1]+self.losses["dis_real"][-1] + 160*self.losses["dis_real_aus"][-1])
        msg += '|'
        print(msg)

        with open(info_dict['log_path'], 'a+') as f:
            f.write(msg + '\n')

    # utils
    def tensor2im(self, input_image, imtype=np.uint8):  # tensor->numpy->image
        if isinstance(input_image, torch.Tensor):
            image_tensor = input_image.data    # 用data不会计算梯度
        else:  # 非tensor，直接返回
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()
        im = self.numpy2im(image_numpy, imtype).resize((80, 80), Image.ANTIALIAS)  # AntiAlias抗锯齿, 图片大小80*80

        return np.array(im)
        
    def numpy2im(self, image_numpy, imtype=np.uint8):
        if image_numpy.shape[0] == 1:
            image_numpy = np.tile(image_numpy, (3, 1, 1))  
            # 如果是灰度图，就把灰度图变成三通道的灰色图
            # tile是在某个维度上重复的意思
        
        # image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0  # 输入应该在 [0, 1]
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) / 2. + 0.5) * 255.0  # 把像素转到[0,255]，输入应该在[-1,1]
        image_numpy = image_numpy.astype(imtype)
        return Image.fromarray(image_numpy)
