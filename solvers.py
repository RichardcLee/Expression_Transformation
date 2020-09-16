from data import create_dataloader
from model import create_model
from visualizer import Visualizer
import copy
import time
import os
import torch
import numpy as np
from PIL import Image


def create_solver(opt):
    return Solver(opt)  # 返回一个用配置文件初始化的求解器


class Solver(object):
    def __init__(self, opt):
        super(Solver, self).__init__()
        self.initialize(opt)

    def initialize(self, opt):  # 初始化
        self.opt = opt  # 传入参数配置
        self.visual = Visualizer(opt)  # 根据参数配置创建可视化模块

    def run_solver(self):  # 根据模式来运行网络
        if self.opt.mode == "train":
            self.train_networks()
        else:
            self.test_networks(self.opt)

    def train_networks(self):  # 训练网络
        self.init_train_setting()  # 初始化训练配置

        for epoch in range(self.opt.epoch_count, self.epoch_len + 1):  # 从epoch_count开始迭代，直到epoch_len
            self.train_epoch(epoch)  # 每次训练以epoch为单位
            self.cur_lr = self.train_model.update_learning_rate()  # 更新并获取当前学习率

            if epoch % self.opt.save_epoch_freq == 0:  # 根据设定的频率保存checkpoints
                self.train_model.save_ckpt(epoch)

        self.train_model.save_ckpt(self.epoch_len)  # 保存最后一次训练的结果

    def init_train_setting(self):  # 初始化训练配置
        self.train_dataset = create_dataloader(self.opt)  # 根据配置创建数据加载器
        self.train_model = create_model(self.opt)  # 根据配置创建模型
        self.train_total_steps = 0  # 记录当前训练次数
        self.epoch_len = self.opt.niter + self.opt.niter_decay  # 总epoch是学习率不变的和学习率线性衰减部分之和
        self.cur_lr = self.opt.lr  # 当前学习率

    def train_epoch(self, epoch):  # 一个epoch的训练流程
        epoch_start_time = time.time()  # 开始时间
        epoch_steps = 0  # 当前epoch的step数，img==step->batch->epoch==iteration
        last_print_step_t = time.time()  # 初始化上次打印到控制台的时间

        for idx, batch in enumerate(self.train_dataset):
            self.train_total_steps += self.opt.batch_size
            epoch_steps += self.opt.batch_size
            self.train_model.feed_batch(batch)  # forward
            self.train_model.optimize_paras(train_gen=(idx % self.opt.train_gen_iter == 0))  # backward
            # 每隔self.opt.train_gen_iter个batch，训练一次生成器

            if self.train_total_steps % self.opt.print_losses_freq == 0:  # 每隔一定step打印一次当前损失
                cur_losses = self.train_model.get_latest_losses()  # 获得当前损失（一个所有损失组成的字典）
                avg_step_t = (time.time() - last_print_step_t) / self.opt.print_losses_freq  # 计算每个step（图片）需要平均处理时间
                last_print_step_t = time.time()  # 更新一下上次打印到控制台的时间

                # 打印损失
                info_dict = {
                        'epoch': epoch,
                         'epoch_len': self.epoch_len,
                         'epoch_steps': epoch_steps,
                         'epoch_steps_len': len(self.train_dataset),
                         'step_time': avg_step_t,
                         'cur_lr': self.cur_lr,
                         'log_path': os.path.join(self.opt.ckpt_dir, self.opt.log_file),
                         'losses': cur_losses
                    }
                self.visual.print_losses_info(info_dict)

            # 可视化
            if self.opt.display and self.train_total_steps % self.opt.display_freq == 0:
                cur_visuals = self.train_model.get_latest_visuals()  # 获取可视化对象（字典）
                visual_path = os.path.join('.', 'visualization')  # 可视化保存路径
                # cur_losses = self.train_model.get_latest_losses()  # 获得当前损失（一个所有损失组成的字典）
                avg_step_t = (time.time() - last_print_step_t) / self.opt.print_losses_freq  # 计算每个step（图片）需要平均处理时间
                last_print_step_t = time.time()  # 更新一下上次打印到控制台的时间
                # 绘制信息
                plot_dict = {''
                             'epoch': epoch,
                             'epoch_len': self.epoch_len,
                             'epoch_steps': epoch_steps,
                             'epoch_steps_len': len(self.train_dataset),
                             'step_time': avg_step_t,
                             'cur_lr': self.cur_lr,
                             'visual_path': visual_path,
                             'img': cur_visuals,
                             'log_path': os.path.join(self.opt.ckpt_dir, self.opt.log_file)
                }
                self.visual.plot(plot_dict)  # 绘制

    def test_networks(self, opt):  # 测试网络
        self.init_test_setting(opt)
        self.test_ops()

    def init_test_setting(self, opt):  # 初始化测试配置
        self.test_dataset = create_dataloader(opt)  # 创建数据加载器
        self.test_model = create_model(opt)  # 创建网络模型

    def test_ops(self):  # 模型测试
        for batch_idx, batch in enumerate(self.test_dataset):
            with torch.no_grad():  # 测试模型时不计算梯度
                faces_list = [batch['src_img'].float().numpy()]  # [[ tensor->numpy+float ]] 添加真实源图像
                paths_list = [batch['src_path'], batch['tar_path']]  # [[图片源路径], [图片目标途径]] 一一对应

                # interpolate several times（插值 平滑）
                for idx in range(self.opt.interpolate_len):
                    # alpha是AU的激活度，表征源图像向目标图像转换的度
                    # 例如：interpolate_len=5, idx=2, α=[0.2, 0.4, 0.6, 0.8, 1.0]
                    cur_alpha = (idx + 1.0) / float(self.opt.interpolate_len)
                    # 通过下式控制变换程度
                    # targetAUs = α·targetAUs + (1-α)·sourceAUs
                    cur_tar_aus = cur_alpha * batch['tar_aus'] + (1 - cur_alpha) * batch['src_aus']
                    test_batch = {'src_img': batch['src_img'], 'tar_aus': cur_tar_aus,
                                  'src_aus': batch['src_aus'], 'tar_img': batch['tar_img']}

                    self.test_model.feed_batch(test_batch)
                    self.test_model.forward()  # 前向传播

                    cur_gen_faces = self.test_model.fake_img.cpu().float().numpy()  # tensor->numpy+float, gpu->cpu
                    faces_list.append(cur_gen_faces)  # 保存生成器生成的虚假图像

                faces_list.append(batch['tar_img'].float().numpy())  # 添加真实目标图像
                # faces_list==[[src],[fake1],[fake2],...,[fake(len)],[tar]]

            self.test_save_imgs(faces_list, paths_list)  # 保存每个batch的所有图像

    def test_save_imgs(self, faces_list, paths_list):  # 保存图像
        for idx in range(len(paths_list[0])):
            # 这里basename包括扩展名，splitext分隔出文件名
            src_name = os.path.splitext(os.path.basename(paths_list[0][idx]))[0]
            tar_name = os.path.splitext(os.path.basename(paths_list[1][idx]))[0]

            if self.opt.save_test_gif:  # 输出动图
                import imageio
                imgs_numpy_list = []

                for face_idx in range(len(faces_list) - 1):  
                    # -1 means to remove target image
                    cur_numpy = np.array(self.visual.numpy2im(faces_list[face_idx][idx]))  # 转化为numpy
                    imgs_numpy_list.extend([cur_numpy for _ in range(3)])  # 为了延时，复制三份

                saved_path = os.path.join(self.opt.results, "%s_%s.gif" % (src_name, tar_name))  # opt.results/srcName_tarName.gif
                imageio.mimsave(saved_path, imgs_numpy_list)

            else:
                # 拼接源图片、不同α对应的生成图片、目标表情图片
                concate_img = np.array(self.visual.numpy2im(faces_list[0][idx]))
                for face_idx in range(1, len(faces_list)):
                    concate_img = np.concatenate((concate_img, np.array(self.visual.numpy2im(faces_list[face_idx][idx]))), axis=1)
                concate_img = Image.fromarray(concate_img)

                saved_path = os.path.join(self.opt.results, "src_%s_tar_%s.jpg" % (src_name, tar_name))
                concate_img.save(saved_path)

                # 单独保存每个α对应的图片
                if self.opt.save_all_alpha_image:
                    for face_idx in range(1, len(faces_list)-1):
                        path = self.opt.results + "/scale_fake_face/alpha_%d/" % face_idx
                        if not os.path.exists(path):
                            os.makedirs(path)

                        img = Image.fromarray(np.array(self.visual.numpy2im(faces_list[face_idx][idx])))
                        img.save(path + src_name + ".jpg")

            print("[Success] Saved images to %s" % saved_path)
