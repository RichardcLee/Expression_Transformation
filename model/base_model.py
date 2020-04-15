import torch
import os
from collections import OrderedDict
import random
from . import model_utils


class BaseModel:
    """定义模型基类"""
    def __init__(self):
        super(BaseModel, self).__init__()
        self.name = "Base"

    def _initialize(self, opt):
        self.opt = opt
        self.gpu_ids = self.opt.gpu_ids
        self.device = torch.device('cuda:%d' % self.gpu_ids[0] if self.gpu_ids else 'cpu')
        self.is_train = (self.opt.mode == "train")
        # inherit to define network model 
        self.models_name = []  # 用于存放模型名

    def setup(self):
        """ 公操作，子类调用 """
        print("%s with Model [%s]" % (self.opt.mode.capitalize(), self.name))

        if self.is_train:  # 训练模式
            self.set_train()

            # 定义损失函数，放到GPU上计算
            self.criterionGAN = model_utils.GANLoss(gan_type=self.opt.gan_type).to(self.device)
            self.criterionL1 = torch.nn.L1Loss().to(self.device)
            self.criterionMSE = torch.nn.MSELoss().to(self.device)
            self.criterionTV = model_utils.TVLoss().to(self.device)

            # 并行计算
            torch.nn.DataParallel(self.criterionGAN, self.gpu_ids)
            torch.nn.DataParallel(self.criterionL1, self.gpu_ids)
            torch.nn.DataParallel(self.criterionMSE, self.gpu_ids)
            torch.nn.DataParallel(self.criterionTV, self.gpu_ids)

            # inherit to set up train/val/test status
            self.losses_name = []
            self.optims = []  # 优化器列表
            self.schedulers = []  # 学习率策略列表
        else:  # 测试模式
            self.set_eval()

    def set_eval(self):
        print("Set model to Test state.")
        for name in self.models_name:
            if isinstance(name, str):
                net = getattr(self, 'net_' + name)  # 获得网络层
                if not self.opt.no_test_eval:
                    # 就是测试的时候使用评估模型（评估模型最主要是BN使用全样本，Dropout失效）
                    net.eval()
                    print("Set net_%s to EVAL." % name)
                else:  # 否则使用训练模型
                    net.train()
        self.is_train = False

    def set_train(self):
        print("Set model to Train state.")
        for name in self.models_name:
            if isinstance(name, str):
                net = getattr(self, 'net_' + name)
                net.train()  # 所有模型设置为训练状态
                print("Set net_%s to TRAIN." % name)
        self.is_train = True

    def set_requires_grad(self, parameters, requires_grad=False):
        # 注意这里默认是False
        if not isinstance(parameters, list):
            parameters = [parameters]  # 包装一下（对传入的单个参数）
        for param in parameters:
            if param is not None:
                param.requires_grad = requires_grad

    def get_latest_visuals(self, visuals_name):  # 返回最新的需要可视化的内容
        # visuals_name is a list of name to visualize
        visual_ret = OrderedDict()  # 学过汇编的都知道，ret is return
        for name in visuals_name:
            if isinstance(name, str) and hasattr(self, name):
                visual_ret[name] = getattr(self, name)
        return visual_ret

    def get_latest_losses(self, losses_name):  # 返回最新的损失
        errors_ret = OrderedDict()
        for name in losses_name:
            if isinstance(name, str):
                cur_loss = float(getattr(self, 'loss_' + name))
                errors_ret[name] = cur_loss
        return errors_ret

    def feed_batch(self, batch):  # 重载
        pass 

    def forward(self):  # 前向传播，重载
        pass

    def optimize_paras(self):  # 重载
        pass

    def update_learning_rate(self):
        """更新所有模型的学习率，并返回第一个优化器(Generator)的学习率，将被继承"""
        for scheduler in self.schedulers:
            scheduler.step()  # 应用学习率衰减策略
        lr = self.optims[0].param_groups[0]['lr']
        return lr

    def save_ckpt(self, epoch, models_name):
        '''
        保存检查点
        :param epoch: int
        :param models_name: list, a list of model's name want to save
        '''
        for name in models_name:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (epoch, name)
                save_path = os.path.join(self.opt.ckpt_dir, save_filename)
                net = getattr(self, 'net_' + name)

                # save cpu params, so that it can be used in other GPU settings
                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    torch.save(net.module.cpu().state_dict(), save_path)
                    # 使用了gpu的情况下，保存checkpoints先转移到cpu，提高通用性
                    net.to(self.gpu_ids[0])
                    net = torch.nn.DataParallel(net, self.gpu_ids)
                else:
                    torch.save(net.cpu().state_dict(), save_path)

    def load_ckpt(self, epoch, models_name):
        '''
        加载检查点
        :param epoch: int
        :param models_name: list, a list of model's name want to load
        '''
        for name in models_name:
            if isinstance(name, str):
                load_filename = '%s_net_%s.pth' % (epoch, name)
                load_path = os.path.join(self.opt.ckpt_dir, load_filename)
                assert os.path.isfile(load_path), "File '%s' does not exist." % load_path
                
                pretrained_state_dict = torch.load(load_path, map_location=str(self.device))
                # 数据在不同设备上处理是不一样的要映射一下

                if hasattr(pretrained_state_dict, '_metadata'):
                    del pretrained_state_dict._metadata

                net = getattr(self, 'net_' + name)
                if isinstance(net, torch.nn.DataParallel):  # 注意CPU和GPU上模型的略微差别
                    net = net.module

                # load only existing keys
                pretrained_dict = {k: v for k, v in pretrained_state_dict.items() if k in net.state_dict()}
                net.load_state_dict(pretrained_dict)
                print("[Info] Successfully load trained weights for %s_net_%s." % (epoch,name))

    def clean_ckpt(self, epoch, models_name):
        '''
        删除某一个epoch的checkpoints文件
        :param epoch: int
        :param models_name: list, a list of model's name want to delete
        :return:
        '''
        for name in models_name:
            if isinstance(name, str):
                load_filename = '%s_net_%s.pth' % (epoch, name)
                load_path = os.path.join(self.opt.ckpt_dir, load_filename)
                if os.path.isfile(load_path):
                    os.remove(load_path)

    def gradient_penalty(self, input_img, generate_img):
        """
        施加梯度惩罚
        :math   x: inter_img, y: inter_img_prob
        :math   L2_norm((dy/dx) - 1)**2
        :param input_img: input original/real image
        :param generate_img: generate fake image
        :return:
        """
        alpha = torch.rand(input_img.size(0), 1, 1, 1).to(self.device)
        inter_img = (alpha * input_img.data + (1 - alpha) * generate_img.data).requires_grad_(True)
        inter_img_prob, _ = self.net_dis(inter_img)  # 返回probability和AU向量

        dydx = torch.autograd.grad(outputs=inter_img_prob,
                                   inputs=inter_img,
                                   grad_outputs=torch.ones(inter_img_prob.size()).to(self.device),
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)  # 拉成一维
        dydx_l2norm = torch.sqrt(torch.sum(dydx ** 2, dim=1))  # L2范数
        return torch.mean((dydx_l2norm - 1) ** 2)
