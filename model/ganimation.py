import torch
from .base_model import BaseModel
from . import model_utils


class GANimationModel(BaseModel):
    """ GANimation Model"""
    def __init__(self):
        super(GANimationModel, self).__init__()
        self.name = "GANimation"

    def initialize(self, opt):
        super(GANimationModel, self).initialize(opt)  # 调用基类初始化公操作

        self.net_gen = model_utils.define_splitG(self.opt.img_nc,  # 通道数，默认3
                                                 self.opt.aus_nc,  # AU向量维数，默认17
                                                 self.opt.ngf,  # Generator第一层feature map数目
                                                 use_dropout=self.opt.use_dropout,  # dropoout
                                                 norm=self.opt.norm,  # normalization 正则化
                                                 init_type=self.opt.init_type,  # weights initialization 权重初始化
                                                 init_gain=self.opt.init_gain,  # scaling factor
                                                 gpu_ids=self.gpu_ids)
        self.models_name.append('gen')  # 维护model记录
        
        if self.is_train:  # 判别器仅用于训练
            self.net_dis = model_utils.define_splitD(self.opt.img_nc,
                                                     self.opt.aus_nc, 
                                                     self.opt.final_size,  # 预处理后图片的大小
                                                     self.opt.ndf,  # 判别器第一层特征图数
                                                     norm=self.opt.norm, 
                                                     init_type=self.opt.init_type, 
                                                     init_gain=self.opt.init_gain, 
                                                     gpu_ids=self.gpu_ids)
            self.models_name.append('dis')  # 维护model记录

        if self.opt.load_epoch > 0:
            self.load_ckpt(self.opt.load_epoch)  # 加载指定epoch的模型

    def setup(self):
        """ 训练模式下，配置Adam优化器和学习率衰减策略 """
        super(GANimationModel, self).setup()  # 调用基类setup
        if self.is_train:
            # setup optimizer
            self.optim_gen = torch.optim.Adam(self.net_gen.parameters(),
                                              lr=self.opt.lr,  
                                              betas=(self.opt.beta1, 0.999))  # Generator
            self.optims.append(self.optim_gen)

            self.optim_dis = torch.optim.Adam(self.net_dis.parameters(), 
                                              lr=self.opt.lr, 
                                              betas=(self.opt.beta1, 0.999))  # Discriminator
            self.optims.append(self.optim_dis)

            # setup schedulers
            # 每个optimizer有一个scheduler，以分别控制学习率的衰减
            self.schedulers = [model_utils.get_scheduler(optim, self.opt) for optim in self.optims]

    def feed_batch(self, batch):  # 一个batch移动到GPU
        self.src_img = batch['src_img'].to(self.device)
        self.tar_aus = batch['tar_aus'].type(torch.FloatTensor).to(self.device)
        if self.is_train:
            self.src_aus = batch['src_aus'].type(torch.FloatTensor).to(self.device)
            self.tar_img = batch['tar_img'].to(self.device)

    def forward(self):
        """ 前向传播：生成虚假图片<=>重建源图片 """
        # 生成过程：fake image
        # color_mask是色彩掩膜，aus_mask是注意力掩膜
        self.color_mask ,self.aus_mask, self.embed = self.net_gen(self.src_img, self.tar_aus)   # 源图片 + 目标AUs => 两个掩膜
        self.fake_img = self.aus_mask * self.src_img + (1 - self.aus_mask) * self.color_mask  # 源图像+两个掩膜 => fake image

        # 重建过程： rec real image
        if self.is_train:
            self.rec_color_mask, self.rec_aus_mask, self.rec_embed = self.net_gen(self.fake_img, self.src_aus)  # 虚假图像+源AUs => 两个掩膜
            self.rec_real_img = self.rec_aus_mask * self.fake_img + (1 - self.rec_aus_mask) * self.rec_color_mask  # 虚假图像+两个掩膜 => rec real image

    def backward_dis(self):
        """判别器反向传播"""
        # 源图片
        pred_real, self.pred_real_aus = self.net_dis(self.src_img)
        self.loss_dis_real = self.criterionGAN(pred_real, True)   # WGAN-GP对抗损失第一项
        self.loss_dis_real_aus = self.criterionMSE(self.pred_real_aus, self.src_aus)  # 条件表情损失第二项

        # 虚假图片
        pred_fake, _ = self.net_dis(self.fake_img.detach())  #  detach to stop backward to generator
        self.loss_dis_fake = self.criterionGAN(pred_fake, False)   # WGAN-GP对抗损失第二项

        # 生成器损失：按预设的权重线性叠加以上损失得到
        self.loss_dis = self.opt.lambda_dis * (self.loss_dis_fake + self.loss_dis_real) \
                        + self.opt.lambda_aus * self.loss_dis_real_aus

        if self.opt.gan_type == 'wgan-gp':  # WGAN-GP对抗损失第三项，施加梯度惩罚，L2范数，
            self.loss_dis_gp = self.gradient_penalty(self.src_img, self.fake_img)
            self.loss_dis = self.loss_dis + self.opt.lambda_wgan_gp * self.loss_dis_gp
        
        # backward 判别器
        self.loss_dis.backward()

    def backward_gen(self):
        """生成器反向传播"""
        # 从源图片生成符合目标表情的虚假图像，生成器需要尽可能骗过判别器
        pred_fake, self.pred_fake_aus = self.net_dis(self.fake_img)
        self.loss_gen_GAN = self.criterionGAN(pred_fake, True)   # GAN LOSS Mean
        # pred值是图片为真的概率，True or False表示计算相对于True或者False的损失
        self.loss_gen_fake_aus = self.criterionMSE(self.pred_fake_aus, self.tar_aus)  # 条件表情损失第一项

        # 从符合目标表情的虚假图片重建原始表情的源图片, 保持身份等关键属性，也即循环一致性损失
        self.loss_gen_rec = self.criterionL1(self.rec_real_img, self.src_img)  # 循环一致性损失

        # 判别器损失：线性叠加以上损失项
        self.loss_gen = self.opt.lambda_dis * self.loss_gen_GAN \
                        + self.opt.lambda_aus * self.loss_gen_fake_aus \
                        + self.opt.lambda_rec * self.loss_gen_rec

        # backward 生成器
        self.loss_gen.backward()

    def optimize_paras(self, train_gen):
        self.forward()
        # 更新判别器
        self.set_requires_grad(self.net_dis, True)
        self.optim_dis.zero_grad()  # clear
        self.backward_dis()  # backward时开始计算grad
        self.optim_dis.step()

        if train_gen:  # 每隔几次判别器参数更新，才更新一次生成器参数
            self.set_requires_grad(self.net_dis, False)
            self.optim_gen.zero_grad()
            self.backward_gen()
            self.optim_gen.step()

    def save_ckpt(self, epoch):  # 保存特定epoch的网络状态
        save_models_name = ['gen', 'dis']
        return super(GANimationModel, self).save_ckpt(epoch, save_models_name)

    def load_ckpt(self, epoch):  # 加载特定epoch的网络状态
        load_models_name = ['gen']
        if self.is_train:  # 仅训练模式加载判别器
            load_models_name.extend(['dis'])
        return super(GANimationModel, self).load_ckpt(epoch, load_models_name)

    def clean_ckpt(self, epoch):
        load_models_name = ['gen', 'dis']
        return super(GANimationModel, self).clean_ckpt(epoch, load_models_name)

    def get_latest_losses(self):  # 返回最新的损失
        get_losses_name = ['dis_fake', 'dis_real', 'dis_real_aus', 'gen_rec', 'dis', 'gen']
        return super(GANimationModel, self).get_latest_losses(get_losses_name)

    def get_latest_visuals(self):  # 返回最新的可视化对象
        visuals_name = ['src_img', 'tar_img', 'color_mask', 'aus_mask', 'fake_img']
        if self.is_train:
            visuals_name.extend(['rec_color_mask', 'rec_aus_mask', 'rec_real_img'])
        return super(GANimationModel, self).get_latest_visuals(visuals_name)
