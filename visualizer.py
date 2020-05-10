import numpy as np
import torch
from PIL import Image


class Visualizer(object):
    def __init__(self, opt):
        super(Visualizer, self).__init__()
        self.initialize(opt)

    def initialize(self, opt):
        self.opt = opt
        self.display = self.opt.display  # 是否开启可视化

        if self.display:  # 表示启用
            pass  # todo

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
