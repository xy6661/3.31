import torch
import torch.nn as nn
import functools
from torch.nn import init
from torch.optim import lr_scheduler
import torch.nn.functional as F

#---多尺度相关融合所需包
from .fusion import CGAFusion

def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            # lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            lr_l = 0.3 ** max(0, epoch + opt.epoch_count - opt.n_epochs)
            return lr_l
        """epoch是当前训练轮数，opt.epoch_count=1，opt.n_epochs=2，opt.n_epochs_decay=3
            此处是建立每个epoch对应的学习率衰减系数的关系函数，具体的学习了初始值是由优化器optimizer决定的"""

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find(
                'BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=()):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def mean_variance_norm(feat):
    size = feat.size()
    mean, std = calc_mean_std(feat)
    normalized_feat = (feat - mean.expand(size)) / std.expand(size)
    return normalized_feat

def gram_matrix(feat):
    (b, c, h, w) = feat.size()
    feat = feat.view(b, c, h * w)
    feat_t = feat.transpose(1, 2)
    gram = torch.bmm(feat, feat_t) / (c * h * w)
    return gram


class AdaAttN(nn.Module):

    def __init__(self, in_planes, max_sample=256 * 256, key_planes=None):
        super(AdaAttN, self).__init__()
        if key_planes is None:
            key_planes = in_planes
            #由于设置了opt.shallow_layer为假，则key_planes始终等于in_planes
        self.v = nn.Conv2d(key_planes, key_planes, (1, 1))
        self.v_style_pooling = nn.Conv2d(key_planes, 1, (1, 1))
        self.k = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.sm = nn.Softmax(dim=-1)
        #self.max_sample = max_sample
        #阶段二：自适应指导生成所有层---
        #这个小型MLP从局部内容查询中生成FiLM参数(gamma,beta)。
        #它通过1x1卷积被高效地实现。
        self.mlp_guidance1 = nn.Sequential(
            nn.Conv2d(key_planes, key_planes, kernel_size=1),
            nn.ReLU(inplace=True),
            #输出通道数为value_channels * 2，分别对应gamma和beta
            nn.Conv2d(key_planes, in_planes , kernel_size=1)
        )
        self.mlp_guidance2 = nn.Sequential(
            nn.Conv2d(key_planes, key_planes, kernel_size=1),
            nn.ReLU(inplace=True),
            # 输出通道数为value_channels * 2，分别对应gamma和beta
            nn.Conv2d(key_planes, in_planes , kernel_size=1)
        )

        self.Q_gudide_conv=nn.Conv2d(key_planes, key_planes, (1, 1))

    def forward(self, content, style,seed=None):
        # 阶段一：计算全局风格向量
        V = self.v(style)
        b, c, h_s, w_s = V.size()   # 使用 h_s, w_s 表示 style 的高和宽

        key_pooling_style = self.v_style_pooling(mean_variance_norm(style)).view(b, 1, h_s*w_s)#b,1,h*w
        style_weights=self.sm(key_pooling_style)

        # b,c,w*h @ b,hw,1 -> b,c,1
        global_style_vector = torch.bmm(V.view(b, c, w_s * h_s), style_weights.permute(0, 2, 1))

        # 阶段二：自适应指导生成
        # 从 content feature 中生成 Q_guide
        Q_guide = self.Q_gudide_conv(mean_variance_norm(content))
        b, _, h_c, w_c = Q_guide.size() # 使用 h_c, w_c 表示 content 的高和宽

        # ==================== 代码修改部分开始 ====================

        # 1. (新) 使用 global_style_vector 生成 FiLM 参数
        #    需要将 global_style_vector 从 (b, c, 1) 扩展为 (b, c, 1, 1) 以适应 Conv2d
        gamma = self.mlp_guidance1(global_style_vector.unsqueeze(-1))
        beta = self.mlp_guidance2(global_style_vector.unsqueeze(-1))
        # gamma, beta = torch.chunk(film_params, 2, dim=1) # gamma 和 beta 的形状都是 (b, c, 1, 1)

        # 2. (新) 使用 gamma 和 beta 去调制 Q_guide
        #    利用广播机制 (broadcasting)，(b,c,1,1) 会自动扩展以匹配 (b,c,h_c,w_c) 的维度
        locala_guidance = gamma * Q_guide + beta

        # ==================== 代码修改部分结束 ====================

        # 阶段三: 计算注意力并融合特征
        locala_guidance_norm = mean_variance_norm(locala_guidance).view(b, -1, h_c * w_c)
        Q1 = Q_guide.view(b, -1, h_c * w_c)
        # locala_guidance 和 Q_guide 维度相同，可以直接 reshape
        Q2 = locala_guidance.view(b, -1, h_c * w_c)
        Q3=Q1.permute(0, 2, 1)#因为前面已经对Q_guide进行了Normal
        Q4=locala_guidance_norm.permute(0, 2, 1)
        Q = Q3 + Q4

        K = self.k(mean_variance_norm(style)).view(b, -1, w_s * h_s).contiguous()
        v_v = V.view(b, -1, w_s * h_s).permute(0, 2, 1)

        energy = torch.bmm(Q, K)
        S = self.sm(energy)
        out_flat = torch.bmm(S, v_v)

        # 将结果重塑为图像格式
        # 输出的尺寸应该与 content/Q_guide 一致
        out = out_flat.permute(0, 2, 1).contiguous().view(b, -1, h_c, w_c)

        style_features = (content) + out+locala_guidance
        return style_features




class Transformer(nn.Module):

    def __init__(self, in_planes, key_planes=None, shallow_layer=False):
        super(Transformer, self).__init__()
        self.attn_adain_4_1 = AdaAttN(in_planes=in_planes, key_planes=key_planes)
        self.attn_adain_5_1 = AdaAttN(in_planes=in_planes,
                                        key_planes=key_planes + 512 if shallow_layer else key_planes)
        self.upsample5_1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.merge_conv_pad = nn.ReflectionPad2d((1, 1, 1, 1))
        self.merge_conv = nn.Conv2d(in_planes, in_planes, (3, 3))

        #-------新增：初始化CGAFusion模块-------
        self.cga_fusion=CGAFusion(dim=in_planes)
        #------------------------------------

    def forward(self, content4_1, style4_1, content5_1, style5_1, seed=None):
        Stylized_feat4=self.attn_adain_4_1(content4_1, style4_1, seed=seed)
        Stylized_feat5=self.attn_adain_5_1(content5_1, style5_1, seed=seed)
        Stylized_feat5_upsampled=self.upsample5_1(Stylized_feat5)
        fused_features=self.cga_fusion(Stylized_feat4,Stylized_feat5_upsampled)
        return self.merge_conv(self.merge_conv_pad(fused_features))
        # return self.merge_conv(self.merge_conv_pad(
        #     self.attn_adain_4_1(content4_1, style4_1,seed=seed) +
        #     self.upsample5_1(self.attn_adain_5_1(content5_1, style5_1,  seed=seed))))



class Decoder(nn.Module):

    def __init__(self, skip_connection_3=False):
        super(Decoder, self).__init__()
        self.decoder_layer_1 = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 256, (3, 3)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest')
        )
        self.decoder_layer_2 = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256 + 256 if skip_connection_3 else 256, 256, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 128, (3, 3)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 128, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 64, (3, 3)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 3, (3, 3))
        )

    def forward(self, cs, c_adain_3_feat=None):
        cs = self.decoder_layer_1(cs)
        if c_adain_3_feat is None:
            cs = self.decoder_layer_2(cs)
        else:
            cs = self.decoder_layer_2(torch.cat((cs, c_adain_3_feat), dim=1))
        return cs

