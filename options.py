import argparse
import torch
import os
from datetime import datetime
import time
import torch 
import random
import numpy as np 
import sys


class Options(object):
    def __init__(self):
        super(Options, self).__init__()
        
    def _initialize(self):   # 初始化
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        # 模式：训练 || 测试
        parser.add_argument('--mode', type=str, default='train', help='Mode of code. [train|test]')
        # 使用的模型：GANimation || Star-GAN
        parser.add_argument('--model', type=str, default='ganimation', help='[ganimation|stargan], see model.__init__ from more details.')
        # 选择随机数种子，提高复现性
        parser.add_argument('--lucky_seed', type=int, default=0, help='seed for random initialize, 0 to use current time.')
        # 是否开启数据可视化
        parser.add_argument('--display',  action='store_true', help='open data visualization.todo')
        # 测试结果保存路径
        parser.add_argument('--results', type=str, default="results", help='save test results to this path.')
        # 测试模式下插值次数
        parser.add_argument('--interpolate_len', type=int, default=5, help='interpolate length for test.')
        parser.add_argument('--no_test_eval', action='store_true', help='do not use eval mode during test time.')
        # 是否单独保存每一个α对应的生成图像（α与变换程度正相关，α∈[0,1]）
        parser.add_argument('--save_all_alpha_image', action='store_true', help='Save all generated images corresponding to different α values')
        # 保存动态图（默认静态拼接图片）
        parser.add_argument('--save_test_gif', action='store_true', help='save gif images instead of the concatenation of static images.')
        # 数据集路径
        parser.add_argument('--data_root', required=True, help='paths to data set.')
        # 数据集中图片的路径
        parser.add_argument('--imgs_dir', type=str, default="imgs", help='path to image')
        # 测试模式
        parser.add_argument('--test_mode', type=str, default="random_target", help='test mode: [single_target|random_target|pair_target]')
        # 配对测试模式,目标图片的路径 todo
        parser.add_argument('--target_imgs_dir', type=str, default="imgs", help='path to target image')
        # 单张目标图片测试模式，需要给出该目标图片的路径
        parser.add_argument('--single_target_img', type=str, default="none", help='path to single target image')

        # 对数据集中图片提取的AU向量（17维）的字典序列化之后的pickle文件。内容是一个字典，key是图片名，value是AU向量。
        parser.add_argument('--aus_pkl', type=str, default="aus_openface.pkl", help='AUs pickle dictionary.')
        # 该csv中需包含数据集中用于训练的图片的id（即文件名）
        parser.add_argument('--train_csv', type=str, default="train_ids.csv", help='train images paths')
        # 该csv中需包含数据集中用于测试的图片的id（即文件名）
        parser.add_argument('--test_csv', type=str, default="test_ids.csv", help='test images paths')
        # 批大小
        parser.add_argument('--batch_size', type=int, default=25, help='input batch size.')
        # 设置后，将不对数据进行shuffle
        parser.add_argument('--serial_batches', action='store_true', help='if specified, input images in order.')
        # 线程数
        parser.add_argument('--n_threads', type=int, default=6, help='number of workers to load data.')
        # 最多使用的图片数
        parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='maximum number of samples.')
        # 图像预处理（图像增强），在windows下设置none会出错，这是pickle在windows下对默认none时使用lambda函数的不支持造成的
        parser.add_argument('--resize_or_crop', type=str, default='none', help='Preprocessing image, [resize_and_crop|crop|none]')
        # 预处理时，调整图片尺寸到size
        parser.add_argument('--load_size', type=int, default=148, help='scale image to this size.')
        # 最终图片大小
        parser.add_argument('--final_size', type=int, default=128, help='crop image to this size.')
        # 预处理不进行图像翻转
        parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip image.')
        # 有关目标AU向量的噪声
        parser.add_argument('--aus_noise', action='store_true', help='if specified, add noise to target AUs.')
        # gpu列表，默认使用gpu0，可以使用多个，设置-1使用cpu
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids, eg. 0,1,2; -1 for cpu.')
        # 保存checkpoints的目录(记录了每个epoch后的模型参数)
        parser.add_argument('--ckpt_dir', type=str, default='./ckpts', help='directory to save check points.')
        # 训练从哪个epoch开始
        parser.add_argument('--load_epoch', type=int, default=0, help='load epoch; 0: do not load')
        # 损失日志的路径
        parser.add_argument('--log_file', type=str, default="logs.txt", help='log loss')
        # 配置文件的路径
        parser.add_argument('--opt_file', type=str, default="opt.txt", help='options file')

        # train options 
        # 图片通道数，默认3通道
        parser.add_argument('--img_nc', type=int, default=3, help='image number of channel')
        # AU向量的维度，默认17维
        parser.add_argument('--aus_nc', type=int, default=17, help='aus number of channel')
        # 生成网络中第一层的特征数（网络中倍增）
        parser.add_argument('--ngf', type=int, default=64, help='ngf')
        # 判别网络中第一层的特征数（网络中倍增）
        parser.add_argument('--ndf', type=int, default=64, help='ndf')
        # 使用dropout
        parser.add_argument('--use_dropout', action='store_true', help='if specified, use dropout.')

        # 指定GAN对抗损失
        parser.add_argument('--gan_type', type=str, default='wgan-gp', help='GAN loss [wgan-gp|lsgan|gan]')
        # 指定神经网络参数初始化策略
        parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal|xavier|kaiming|orthogonal]')
        # 初始化使用的gain比例因子
        parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
        # 指定正则化方式
        parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization [batch|instance|none]')
        # 配置Adma优化器的动量
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        # 初始学习率
        parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam')
        # 学习率变换策略
        parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy: lambda|step|plateau|cosine')
        # 学习率衰减间隔
        parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
        # 设置开始时是第几个epoch，便于保存chekcpoint和进行tune
        parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')

        # niter + niter_decay = 总epoch（包括预训练模型的epoch）
        # 保持初始学习率的迭代次数，应等于预训练模型epoch
        parser.add_argument('--niter', type=int, default=20, help='# of iter at starting learning rate')
        # 线性下降学习率到零的次数
        parser.add_argument('--niter_decay', type=int, default=10, help='# of iter to linearly decay learning rate to zero')

        # 各种损失项的权重
        parser.add_argument('--lambda_dis', type=float, default=1.0, help='discriminator weight in loss')
        parser.add_argument('--lambda_aus', type=float, default=160.0, help='AUs weight in loss')
        parser.add_argument('--lambda_rec', type=float, default=10.0, help='reconstruct loss weight')
        parser.add_argument('--lambda_mask', type=float, default=0, help='mse loss weight')
        parser.add_argument('--lambda_tv', type=float, default=0, help='total variation loss weight')
        parser.add_argument('--lambda_wgan_gp', type=float, default=10., help='wgan gradient penalty weight')

        # 每多少次迭代训练一次生成器
        parser.add_argument('--train_gen_iter', type=int, default=5, help='train G every n interations.')
        # 打印损失的频率
        parser.add_argument('--print_losses_freq', type=int, default=100, help='print log every print_freq step.')
        # 每多少个epoch保存一次checkpoints
        parser.add_argument('--save_epoch_freq', type=int, default=2, help='save checkpoint every save_epoch_freq epoch.')

        return parser

    def parse(self):    # 解析参数
        parser = self._initialize()  # 初始化默认值和参数项
        parser.set_defaults(name=datetime.now().strftime("%y%m%d_%H%M%S"))  # 设置name参数，初值为当前时间
        opt = parser.parse_args()   # 命令行参数
        dataset_name = os.path.basename(opt.data_root.strip('/'))

        if opt.mode == 'train' and opt.load_epoch == 0:  # 创建检查点目录，用于保存之后的ckp
            # e.g.（ckpts\celebA\ganimation\200316_161852）
            opt.ckpt_dir = os.path.join(opt.ckpt_dir, dataset_name, opt.model, opt.name)

            if not os.path.exists(opt.ckpt_dir):
                os.makedirs(opt.ckpt_dir)

        # if test, disable visdom, update results path
        if opt.mode == "test":  # 测试环节需进行如下特殊处理：
            opt.display = False   # 关闭可视化
            # 修改results路径，e.g. results\celebA_ganimation_30
            opt.results = os.path.join(opt.results, "%s_%s_%s" % (dataset_name, opt.model, opt.load_epoch))

            if not os.path.exists(opt.results):
                os.makedirs(opt.results)

        # 配置gpu
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            cur_id = int(str_id)
            if cur_id >= 0:
                opt.gpu_ids.append(cur_id)

        if len(opt.gpu_ids) > 0:  # 有多个GPU默认只使用第一个
            torch.cuda.set_device(opt.gpu_ids[0])

        if opt.lucky_seed == 0:  # 如果没有设置随机种子，则用系统时间作为种子
            opt.lucky_seed = int(time.time())

        random.seed(a=opt.lucky_seed)
        np.random.seed(seed=opt.lucky_seed)
        torch.manual_seed(opt.lucky_seed)

        if len(opt.gpu_ids) > 0:  # 使用cudnn神经网络加速库,并设置一些奇怪的东西和随机数种子
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.cuda.manual_seed(opt.lucky_seed)
            torch.cuda.manual_seed_all(opt.lucky_seed)

        # 记录每次运行模型时配置的参数值
        script_dir = opt.ckpt_dir
        with open(os.path.join(script_dir, "run_script.sh"), 'a+') as f:
            f.write("[%5s][%s]python %s\n" % (opt.mode, opt.name, ' '.join(sys.argv)))

        # 打印和保存配置信息
        msg = ''
        msg += '------------------- [%5s][%s]Options --------------------\n' % (opt.mode, opt.name)
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default_v = parser.get_default(k)
            if v != default_v:
                comment = '\t[default: %s]' % str(default_v)
            msg += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        msg += '--------------------- [%5s][%s]End ----------------------\n' % (opt.mode, opt.name)
        print(msg)
        with open(os.path.join(script_dir, "opt.txt"), 'a+') as f:
            f.write(msg + '\n')
        return opt
