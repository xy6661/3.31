import torch
import torch.nn as nn
import itertools
from .base_model import BaseModel
from . import networks
from . import loss
import torch.nn.functional as F



class AdaAttNModel(BaseModel):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--image_encoder_path', required=True, help='path to pretrained image encoder')
        parser.add_argument('--skip_connection_3', action='store_true',
                            help='if specified, add skip connection on ReLU-3')
        parser.add_argument('--shallow_layer', action='store_true',
                            help='if specified, also use features of shallow layers')
        if is_train:
            parser.add_argument('--lambda_content', type=float, default=3., help='weight for L2 content loss')
            parser.add_argument('--lambda_global', type=float, default=10., help='weight for L2 style loss')
            parser.add_argument('--lambda_local', type=float, default=0.,
                                help='weight for attention weighted style loss')
            parser.add_argument('--lambda_id1', type=float, default=1., help='weight for identity loss 1')#为身份损失增加，参数暂时未曾调整
            parser.add_argument('--lambda_id2', type=float, default=1., help='weight for identity loss 2')

            #增加对比损失命令行选项
            parser.add_argument('--lambda_contrast_c', type=float, default=0.3, help='内容对比损失的权重')
            parser.add_argument('--lambda_contrast_s', type=float, default=0.3, help='风格对比损失的权重')

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        image_encoder = nn.Sequential(
            nn.Conv2d(3, 3, (1, 1)),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(3, 64, (3, 3)),
            nn.ReLU(),  # relu1-1
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),  # relu1-2
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 128, (3, 3)),
            nn.ReLU(),  # relu2-1
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 128, (3, 3)),
            nn.ReLU(),  # relu2-2
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 256, (3, 3)),
            nn.ReLU(),  # relu3-1
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),  # relu3-2
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),  # relu3-3
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),  # relu3-4
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 512, (3, 3)),
            nn.ReLU(),  # relu4-1, this is the last layer used
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu4-2
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu4-3
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu4-4
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu5-1
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu5-2
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu5-3
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU()  # relu5-4
        )
        image_encoder.load_state_dict(torch.load(opt.image_encoder_path))
        enc_layers = list(image_encoder.children())
        enc_1 = nn.DataParallel(nn.Sequential(*enc_layers[:4]).to(opt.gpu_ids[0]), opt.gpu_ids)
        enc_2 = nn.DataParallel(nn.Sequential(*enc_layers[4:11]).to(opt.gpu_ids[0]), opt.gpu_ids)
        enc_3 = nn.DataParallel(nn.Sequential(*enc_layers[11:18]).to(opt.gpu_ids[0]), opt.gpu_ids)
        enc_4 = nn.DataParallel(nn.Sequential(*enc_layers[18:31]).to(opt.gpu_ids[0]), opt.gpu_ids)
        enc_5 = nn.DataParallel(nn.Sequential(*enc_layers[31:44]).to(opt.gpu_ids[0]), opt.gpu_ids)
        self.image_encoder_layers = [enc_1, enc_2, enc_3, enc_4, enc_5]
        for layer in self.image_encoder_layers:
            for param in layer.parameters():
                param.requires_grad = False
        self.visual_names = ['c', 'cs', 's']#用于指定在可视化或者保存结果时需要展示的变量名
        self.model_names = ['decoder', 'transformer','projector_c','projector_s']
        parameters = []
        self.max_sample = 64 * 64
        if opt.skip_connection_3:     #第三层在我的代码中是不需要的
            adaattn_3 = networks.AdaAttN(in_planes=256, key_planes=256 + 128 + 64 if opt.shallow_layer else 256,
                                              max_sample=self.max_sample)
            self.net_adaattn_3 = networks.init_net(adaattn_3, opt.init_type, opt.init_gain, opt.gpu_ids)
            self.model_names.append('adaattn_3')
            parameters.append(self.net_adaattn_3.parameters())
        if opt.shallow_layer:#本份代码全部不启用opt.shallow_layer
            channels = 512 + 256 + 128 + 64
        else:
            channels = 512
        transformer = networks.Transformer(
            in_planes=512, key_planes=channels, shallow_layer=opt.shallow_layer)
        decoder = networks.Decoder(opt.skip_connection_3)
        self.net_decoder = networks.init_net(decoder, opt.init_type, opt.init_gain, opt.gpu_ids)
        self.net_transformer = networks.init_net(transformer, opt.init_type, opt.init_gain, opt.gpu_ids)
        parameters.append(self.net_decoder.parameters())
        parameters.append(self.net_transformer.parameters())

#此处实例化投影器和损失函数===========================================================
        projector_c = loss.FeatureProjector(input_dim=256)  # relu3_1 的通道数是 256
        projector_s = loss.GramProjector(input_dim=256*256)
        self.net_projector_c = networks.init_net(projector_c, opt.init_type, opt.init_gain, opt.gpu_ids)
        self.net_projector_s = networks.init_net(projector_s, opt.init_type, opt.init_gain, opt.gpu_ids)
        # 将投影器的参数加入到优化器中
        parameters.append(self.net_projector_c.parameters())
        parameters.append(self.net_projector_s.parameters())


        self.c = None
        self.cs = None
        self.s = None
        self.s_feats = None
        self.c_feats = None
        self.seed = 6666
        if self.isTrain:
            self.loss_names = ['content', 'global','id1','id2', 'contrast_c', 'contrast_s']#在loss_names中添加身份损失
            self.criterionMSE = torch.nn.MSELoss().to(self.device)
            # 实例化对比损失函数 (它没有可训练参数)
            self.criterionContentContrastive = loss.ContentContrastiveLoss().to(self.device)
            self.criterionStyleContrastive = loss.StyleContractiveLoss().to(self.device)

            self.optimizer_g = torch.optim.Adam(itertools.chain(*parameters), lr=opt.lr)
            self.optimizers.append(self.optimizer_g)
            self.loss_global = torch.tensor(0., device=self.device)
            self.loss_local = torch.tensor(0., device=self.device)
            self.loss_content = torch.tensor(0., device=self.device)

            self.loss_id1 = torch.tensor(0., device=self.device) #用于内容身份损失
            self.loss_id2 = torch.tensor(0., device=self.device) #用于风格身份损失#在此处不声明也没有关系，这几个损失的初始值后面也是会提到的

    def set_input(self, input_dict):
        """将内容图像和风格图像在字典中取出来，并且还进行配对"""
        self.c = input_dict['c'].to(self.device)
        self.s = input_dict['s'].to(self.device)
        self.image_paths = input_dict['name']

    def encode_with_intermediate(self, input_img):
        results = [input_img]
        for i in range(5):
            func = self.image_encoder_layers[i]
            results.append(func(results[-1]))
        return results[1:]

    @staticmethod
    def get_key(feats, last_layer_idx, need_shallow=True):
        if need_shallow and last_layer_idx > 0:
            results = []
            _, _, h, w = feats[last_layer_idx].shape
            for i in range(last_layer_idx):
                results.append(networks.mean_variance_norm(nn.functional.interpolate(feats[i], (h, w))))
            results.append(networks.mean_variance_norm(feats[last_layer_idx]))
            return torch.cat(results, dim=1)
        else:
            return networks.mean_variance_norm(feats[last_layer_idx])

    def forward(self):
        self.c_feats = self.encode_with_intermediate(self.c)
        self.s_feats = self.encode_with_intermediate(self.s)
        if self.opt.skip_connection_3:
            c_adain_feat_3 = self.net_adaattn_3(self.c_feats[2], self.s_feats[2], self.get_key(self.c_feats, 2, self.opt.shallow_layer),
                                                   self.get_key(self.s_feats, 2, self.opt.shallow_layer), self.seed)
        else:
            c_adain_feat_3 = None
        cs = self.net_transformer(self.c_feats[3], self.s_feats[3], self.c_feats[4], self.s_feats[4], self.seed)
        self.cs = self.net_decoder(cs, c_adain_feat_3)

        if self.isTrain:
            # 身份损失相关参数==========================================================================================================================================
            cc = self.net_transformer(self.c_feats[3], self.c_feats[3], self.c_feats[4], self.c_feats[4], self.seed)
            self.cc = self.net_decoder(cc)

            ss = self.net_transformer(self.s_feats[3], self.s_feats[3], self.s_feats[4], self.s_feats[4], self.seed)
            self.ss = self.net_decoder(ss)
        # 身份损失相关参数============================结束==============================================================================================================

 # 对比损失相关参数==========================================================================================================================================
            # 为了获得 I_ccs (内容正样本)，我们用风格化的图像 cs 来对内容图像 c 进行再次风格化。
            self.cs_feats = self.encode_with_intermediate(self.cs)
            # transformer 的输入是 relu4_1 和 relu5_1 特征
            cs_c_transformer_output = self.net_transformer(self.cs_feats[3], self.c_feats[3], self.cs_feats[4],
                                                           self.c_feats[4], self.seed)
            self.I_csc = self.net_decoder(cs_c_transformer_output)

            # 为了获得 I_scs (风格正样本)，我们用风格化的图像 cs 来对风格图像 s 进行风格化。
            s_cs_transformer_output = self.net_transformer(self.s_feats[3], self.cs_feats[3], self.s_feats[4],
                                                           self.cs_feats[4], self.seed)
            self.I_scs = self.net_decoder(s_cs_transformer_output)  # 风格正样本图像

    # 对比损失相关参数============================结束==============================================================================================================

    def compute_Contentcontrastive_losses(self):
        """计算内容和风格的对比损失"""
        # --- 7. 提取特征并进行投影 ---
        # 对比损失使用 relu3_1 特征，它在特征列表中的索引是 2
        phi3_c = self.c_feats[2]
        phi3_csc = self.encode_with_intermediate(self.I_csc)[2]  # 内容正样本的特征

        # 使用内容投影器
        z_c = self.net_projector_c(phi3_c)  # 查询 (Query)
        z_csc = self.net_projector_c(phi3_csc)  # 正样本 (Positive Key)

        # ---  使用损失标准计算损失 ---
        # 内容对比损失：锚点是原始内容图像 c
        self.loss_contrast_c = self.criterionContentContrastive(z_c, z_csc)

    # def compute_Stylecontrastive_losses(self):
    #     """计算内容和风格的对比损失"""
    #     # --- 7. 提取特征并进行投影 ---
    #     # 对比损失使用 relu3_1 特征，它在特征列表中的索引是 2
    #     phi3_s = self.s_feats[2]
    #     phi3_scs = self.encode_with_intermediate(self.I_scs)[2]
    #     z_s = self.net_projector_s(phi3_s)
    #     z_scs = self.net_projector_s(phi3_scs)
    #     self.loss_contrast_s = self.criterionContrastive(z_s, z_scs)

    def compute_Stylecontrastive_losses(self):
        """计算基于投影Gram的风格对比损失"""
        phi3_s = self.s_feats[2]
        phi3_scs = self.encode_with_intermediate(self.I_scs)[2]

        # --- 将 VGG 特征和 projector 网络一起传入损失函数 ---
        self.loss_contrast_s = self.criterionStyleContrastive(phi3_s, phi3_scs, self.net_projector_s)


    def compute_content_loss(self, stylized_feats):
        self.loss_content = torch.tensor(0., device=self.device)
        if self.opt.lambda_content > 0:
            for i in range(1, 5):
                self.loss_content += self.criterionMSE(networks.mean_variance_norm(stylized_feats[i]),
                                                       networks.mean_variance_norm(self.c_feats[i]))

    def compute_style_loss(self, stylized_feats):
        self.loss_global = torch.tensor(0., device=self.device)         #这行代码中还是命名为Self.global，虽然是表示风格损失，但是若是想清楚一点表达，最好还是改好
        if self.opt.lambda_global > 0:
            for i in range(1, 5):
                s_feats_mean, s_feats_std = networks.calc_mean_std(self.s_feats[i])
                stylized_feats_mean, stylized_feats_std = networks.calc_mean_std(stylized_feats[i])
                self.loss_global += self.criterionMSE(
                    stylized_feats_mean, s_feats_mean) + self.criterionMSE(stylized_feats_std, s_feats_std)

    def computer_lossid1(self, content_content,style_style):
        self.loss_id1=torch.tensor(0., device=self.device)
        self.loss_id1=self.criterionMSE(content_content, self.c)+self.criterionMSE(style_style, self.s)
    #=======================================================================================================================================================
    #=====================上面是计算loss_id1函数，下面是计算loss_id2函数==========================================================================================
    def computer_lossid2(self, content_content_feats,style_style_feats):
        self.loss_id2=torch.tensor(0., device=self.device)
        for i in range(1, 5):
            self.loss_id2 += self.criterionMSE(content_content_feats[i],self.c_feats[i])
            self.loss_id2 += self.criterionMSE(style_style_feats[i],self.s_feats[i])

    #=========================================================================================================================================================










    """
    # def compute_style_loss(self, stylized_feats):
    #     self.loss_global = torch.tensor(0., device=self.device)
    #     if self.opt.lambda_global > 0:
    #         for i in range(1, 5):
    #             s_feats_mean, s_feats_std = networks.calc_mean_std(self.s_feats[i])
    #             stylized_feats_mean, stylized_feats_std = networks.calc_mean_std(stylized_feats[i])
    #             self.loss_global += self.criterionMSE(
    #                 stylized_feats_mean, s_feats_mean) + self.criterionMSE(stylized_feats_std, s_feats_std)
    #     self.loss_local = torch.tensor(0., device=self.device)
    #     if self.opt.lambda_local > 0:
    #         for i in range(1, 5):
    #             c_key = self.get_key(self.c_feats, i, self.opt.shallow_layer)
    #             s_key = self.get_key(self.s_feats, i, self.opt.shallow_layer)
    #             s_value = self.s_feats[i]
    #             b, _, h_s, w_s = s_key.size()
    #             s_key = s_key.view(b, -1, h_s * w_s).contiguous()
    #             if h_s * w_s > self.max_sample:
    #                 torch.manual_seed(self.seed)
    #                 index = torch.randperm(h_s * w_s).to(self.device)[:self.max_sample]
    #                 s_key = s_key[:, :, index]
    #                 style_flat = s_value.view(b, -1, h_s * w_s)[:, :, index].transpose(1, 2).contiguous()
    #             else:
    #                 style_flat = s_value.view(b, -1, h_s * w_s).transpose(1, 2).contiguous()
    #             b, _, h_c, w_c = c_key.size()
    #             c_key = c_key.view(b, -1, h_c * w_c).permute(0, 2, 1).contiguous()
    #             attn = torch.bmm(c_key, s_key)
    #             # S: b, n_c, n_s
    #             attn = torch.softmax(attn, dim=-1)
    #             # mean: b, n_c, c
    #             mean = torch.bmm(attn, style_flat)
    #             # std: b, n_c, c
    #             std = torch.sqrt(torch.relu(torch.bmm(attn, style_flat ** 2) - mean ** 2))
    #             # mean, std: b, c, h, w
    #             mean = mean.view(b, h_c, w_c, -1).permute(0, 3, 1, 2).contiguous()
    #             std = std.view(b, h_c, w_c, -1).permute(0, 3, 1, 2).contiguous()
    #             self.loss_local += self.criterionMSE(stylized_feats[i], std * networks.mean_variance_norm(self.c_feats[i]) + mean)
    """
    def compute_losses(self):
        stylized_feats = self.encode_with_intermediate(self.cs)
        content_content=self.cc#loss_id1是直接用decoder生成的这个东西进行计算的
        style_style=self.ss
        content_content_feats = self.encode_with_intermediate(self.cc)#用于计算loss_id2,,，loss_id2需要过一下VGG
        style_style_feats = self.encode_with_intermediate(self.ss)#用于计算loss_id2
        self.compute_content_loss(stylized_feats)
        self.compute_style_loss(stylized_feats)

        self.computer_lossid1(content_content,style_style)#计算loss_id1
        self.computer_lossid2(content_content_feats,style_style_feats)#计算loss_id2

        self.compute_Contentcontrastive_losses()
        self.compute_Stylecontrastive_losses()

        self.loss_content = self.loss_content *self.opt.lambda_content
        self.loss_local = self.loss_local * self.opt.lambda_local
        self.loss_global = self.loss_global * self.opt.lambda_global

        self.loss_id1=self.loss_id1 * self.opt.lambda_id1
        self.loss_id2=self.loss_id2 * self.opt.lambda_id2

        self.loss_contrast_c = self.loss_contrast_c * self.opt.lambda_contrast_c
        self.loss_contrast_s = self.loss_contrast_s * self.opt.lambda_contrast_s

        
    def optimize_parameters(self):
        self.seed = int(torch.randint(10000000, (1,))[0])
        self.forward()
        self.optimizer_g.zero_grad()
        self.compute_losses()
        loss = self.loss_content + self.loss_global + self.loss_local+self.loss_id1+self.loss_id2+ self.loss_contrast_c + self.loss_contrast_s
        loss.backward()
        self.optimizer_g.step()

