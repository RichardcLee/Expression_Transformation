import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
from collections import OrderedDict


'''
Helper functions for model
Borrow tons of code from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
'''


def get_norm_layer(norm_type='instance'):  # 将所有normalization layer的公共参数置为默写参数，简化调用
    """Return a normalization layer
    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none
    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':  # Batch Normalization
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':  # Instance Normalization
        # change default flag, make sure instance norm behave as the same in both train and eval
        # https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/issues/395
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):  # 根据配置来获取学习率衰减策略
    """  adjust the learning rate based on the number of epochs.
        https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
    """
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', gain=0.02):
    """ initialize every layer's weights in the net
        :param net: net model
        :param init_type: method of net weights initialization
        :param gain:
        :return: None
    """
    def init_func(m):
        classname = m.__class__.__name__
        # 卷积层和线性层权重初始化
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:  # 偏差初始化
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # 正则化层权值初始化
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """initialize net
    :param net: net model
    :param init_type: method of net weights initialization
    :param init_gain:
    :param gpu_ids: list, available gpu list
    :return: net model after initialization
    """
    if len(gpu_ids) > 0:  # 若有GPU，则将网络转移到GPU上训练
        # print("gpu_ids,", gpu_ids)
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # 并行训练
    init_weights(net, init_type, gain=init_gain)
    return net


def define_G(input_nc, output_nc, ngf, which_model_netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """ define Generator
     :param input_nc: int, channel of input image
     :param output_nc: int, channel of output image
     :param ngf: int, feature amount of Generator input layer(first layer)
     :param which_model_netG: str, name of Generator
     :param norm: str, name of normalization layer
     :param use_dropout: bool, use dropout or not
     :param init_type: str, method of net weights initialization
     :param init_gain:
     :param gpu_ids: list, available gpu list
     :return: net after processing
     """
    netG = None
    norm_layer = get_norm_layer(norm_type=norm)

    if which_model_netG == 'resnet_9blocks':  # 残差网络
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif which_model_netG == 'resnet_6blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    elif which_model_netG == 'unet_128':  #
        netG = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif which_model_netG == 'unet_256':
        netG = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)
    return init_net(netG, init_type, init_gain, gpu_ids)


def define_D(input_nc, ndf, which_model_netD, n_layers_D=3, norm='batch', use_sigmoid=False,
             init_type='normal', init_gain=0.02, gpu_ids=[]):
    """define Discriminator
        :param input_nc: int, channel of input image
        :param ndf: int, feature amount of Discrimination input layer(first layer)
        :param which_model_netD: str, name of Discriminator model
        :param n_layers_D:
        :param norm: str, name of normalization layer
        :param use_sigmoid:  bool, use or not use sigmoid
        :param init_type: str, method of net weights initialization
        :param init_gain:
        :param gpu_ids: list, available gpu list
        :return: net after processing
    """
    netD = None
    norm_layer = get_norm_layer(norm_type=norm)

    if which_model_netD == 'basic':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif which_model_netD == 'n_layers':  # 马尔可夫判别器
        netD = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif which_model_netD == 'pixel':
        netD = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' %
                                  which_model_netD)
    return init_net(netD, init_type, init_gain, gpu_ids)


##############################################################################
# Classes
##############################################################################


class GANLoss(nn.Module):
    """
        Defines the GAN loss which uses either LSGAN or the regular GAN.
        When LSGAN is used, it is basically same as MSELoss,
        but it abstracts away the need to create the target label tensor
        that has the same size as the input
    """
    def __init__(self, gan_type='wgan-gp', target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_type = gan_type
        if self.gan_type == 'wgan-gp':
            self.loss = lambda x, y: -torch.mean(x) if y else torch.mean(x)
        elif self.gan_type == 'lsgan':  # 损失敏感GAN，使用均方误差
            self.loss = nn.MSELoss()
        elif self.gan_type == 'gan':  # 一般GAN，使用二元交叉熵
            self.loss = nn.BCELoss()
        else:
            raise NotImplementedError('GAN loss type [%s] is not found' % gan_type)

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        if self.gan_type == 'wgan-gp':
            target_tensor = target_is_real
        else:
            target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


class ResnetGenerator(nn.Module):
    """
        Defines the generator that consists of Resnet blocks between a few
        downsampling/upsampling operations.
        Code and idea originally from Justin Johnson's architecture.
        https://github.com/jcjohnson/fast-neural-style/
    """
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),  # 类似于一种 镜像填充
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        # 下采样
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        # 上采样
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':  # 默认方式，零值填充
            p = 1  # 四周各补一行
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]  # 3*3, Ic=Oc，basic block?
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]  # 3*3

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetGenerator, self).__init__()

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input):
        return self.model(input)


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        '''
        :param outer_nc: int, 输出通道数
        :param inner_nc:
        :param input_nc:  int, 输入通道数
        :param submodule: 中间子模块
        :param outermost: 结构一
        :param innermost: 结构二
        :param norm_layer: 正则化层
        :param use_dropout: 是否使用dropout
        '''
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):  # 马尔可夫判别器
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


class PixelDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        if use_sigmoid:
            self.net.append(nn.Sigmoid())

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        return self.net(input)


##############################################################################
# Basic network model 
##############################################################################
# 返回已初始化的模型
def define_splitG(img_nc, aus_nc, ngf, use_dropout=False, norm='instance', init_type='normal', init_gain=0.02, gpu_ids=[]):
    norm_layer = get_norm_layer(norm_type=norm)
    net_img_au = SplitGenerator(img_nc, aus_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    return init_net(net_img_au, init_type, init_gain, gpu_ids)


def define_splitD(input_nc, aus_nc, image_size, ndf, norm='instance', init_type='normal', init_gain=0.02, gpu_ids=[]):
    norm_layer = get_norm_layer(norm_type=norm)
    net_dis_aus = SplitDiscriminator(input_nc, aus_nc, image_size, ndf, n_layers=6, norm_layer=norm_layer)
    return init_net(net_dis_aus, init_type, init_gain, gpu_ids)


class SplitGenerator(nn.Module):
    def __init__(self, img_nc, aus_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='zero'):
        assert(n_blocks >= 0)
        super(SplitGenerator, self).__init__()
        self.input_nc = img_nc + aus_nc  # 默认 3 + 17 = 20
        self.ngf = ngf  # 默认64
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.Conv2d(self.input_nc, ngf, kernel_size=7, stride=1, padding=3, 
                           bias=use_bias),  # 卷积核7*7 步幅1 填充3，输入特征数（通道数）Ic:20，输出特征数Oc:64
                 norm_layer(ngf),
                 nn.ReLU(True)]  # ReLU(inplace=True) 对Conv2d中传递下来的tensor直接进行修改，这样能够节省运算内存，不用多存储其他变量

        n_downsampling = 2
        for i in range(n_downsampling):  # 下采样
            mult = 2**i  # 2^0=1 2^1=2
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, \
                                kernel_size=4, stride=2, padding=1, \
                                bias=use_bias),  # Ic:64 Oc:128  Ic2:128 Oc2:256
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2**n_downsampling  # 2^2=4
        for i in range(n_blocks):  # 默认6个残差块，默认填充方式为zero, Ic:256 Oc:256
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # 上采样
            mult = 2**(n_downsampling - i)  # 2^2=4, 2^1=2
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=4, stride=2, padding=1,
                                         bias=use_bias),  # 4*4, Ic:256 Oc:128, Ic2:128 Oc2:64
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]

        self.model = nn.Sequential(*model)
        # color mask generator top
        color_top = []
        color_top += [nn.Conv2d(ngf, img_nc, kernel_size=7, stride=1, padding=3, bias=False),
                        nn.Tanh()]
        self.color_top = nn.Sequential(*color_top)
        # AUs mask generator top 
        au_top = []
        au_top += [nn.Conv2d(ngf, 1, kernel_size=7, stride=1, padding=3, bias=False),
                    nn.Sigmoid()]
        self.au_top = nn.Sequential(*au_top)

        # from torchsummary import summary
        # summary(self.model.to("cuda"), (20, 148, 148))
        # summary(self.color_top.to("cuda"), (64, 148, 148))
        # summary(self.au_top.to("cuda"), (64, 148, 148))
        # assert False

    def forward(self, img, au):
        # replicate AUs vector to match image shap and concate to construct input 
        sparse_au = au.unsqueeze(2).unsqueeze(3)  # 在当前第二维增加一个维度，然后又在当前第三维增加一个维度
        # 把AUs扩展成[batch, N, H, W]，其中N为AUs向量长度（默认17），H和W分别为图像高和宽，batch为batch size
        sparse_au = sparse_au.expand(sparse_au.size(0), sparse_au.size(1), img.size(2), img.size(3))
        # 把输入的原图像和目标AUs拼接到一起 [batch, C+N, H, W]，其中C为图片通道数（默认3）
        self.input_img_au = torch.cat([img, sparse_au], dim=1)

        embed_features = self.model(self.input_img_au)  # 模型产出内嵌特征图

        return self.color_top(embed_features), self.au_top(embed_features), embed_features  # 返回色彩掩模、注意力掩膜和内嵌特征图

# 测试，取消summary注释
# SplitGenerator(3, 17)


class SplitDiscriminator(nn.Module):
    def __init__(self, input_nc, aus_nc, image_size=128, ndf=64, n_layers=6, norm_layer=nn.BatchNorm2d):
        super(SplitDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),  # 4*4 s2p1, Ic:3, Oc:64
            nn.LeakyReLU(0.01, True)
        ]

        cur_dim = ndf  # 64
        for n in range(1, n_layers):  # 添加n_layers-1层，默认n_layers=6
            # Ic:64 Oc:128, Ic2:128 Oc2:256, Ic3:256 Oc3:512, Ic4:512 Oc4:1024, Ic5:1024 Oc5:2048
            sequence += [
                nn.Conv2d(cur_dim, 2 * cur_dim,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                nn.LeakyReLU(0.01, True)
            ]
            cur_dim = 2 * cur_dim

        self.model = nn.Sequential(*sequence)
        # patch discriminator top , patchGAN
        dis_top = []
        self.dis_top = nn.Conv2d(cur_dim, 1, kernel_size=kw-1, stride=1, padding=padw, bias=False)
        # AUs classifier top，Aus分类器
        k_size = int(image_size / (2 ** n_layers))  # 默认128/(2^6)=2
        self.aus_top = nn.Conv2d(cur_dim, aus_nc, kernel_size=k_size, stride=1, bias=False)

        # from torchsummary import summary
        # summary(self.model.to("cuda"), (3, 148, 148))
        # summary(self.dis_top.to("cuda"), (2048, 2, 2))
        # summary(self.aus_top.to("cuda"), (2048, 2, 2))
        # assert False

    def forward(self, img):
        embed_features = self.model(img)
        pred_map = self.dis_top(embed_features)   # patch
        pred_aus = self.aus_top(embed_features)   # AUs分类
        return pred_map.squeeze(), pred_aus.squeeze()


# 测试，取消summary注释
# SplitDiscriminator(3, 17)


# https://github.com/jxgu1016/Total_Variation_Loss.pytorch/blob/master/TVLoss.py
class TVLoss(nn.Module):  # 总变分损失
    def __init__(self, TVLoss_weight=1):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]




