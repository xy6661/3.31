import os
import time
import torch
import torch.nn as nn
from thop import profile, clever_format

from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html


# ==========================================
# 新增: 包装 BaseModel 并自动剥离 DataParallel，完美适配 thop 计算
# ==========================================
class ModelWrapper(nn.Module):
    def __init__(self, basemodel):
        super(ModelWrapper, self).__init__()
        self.basemodel = basemodel

        # 【修复点 1】: 遍历所有的网络模块，如果是 DataParallel，则剥离它。
        # 否则 thop 注入的 CPU 计数 Buffer 会和 DataParallel 的设备检查冲突
        for name in basemodel.model_names:
            if isinstance(name, str):
                net = getattr(basemodel, 'net_' + name)
                if isinstance(net, nn.DataParallel):
                    net = net.module
                    setattr(basemodel, 'net_' + name, net)  # 将纯净的网络覆写回 BaseModel
                self.add_module('net_' + name, net)

        # 【修复点 2】: 单独处理 VGG 的图像编码器层，它们在原代码里也是 DataParallel
        if hasattr(basemodel, 'image_encoder_layers'):
            new_layers = []
            for i, layer in enumerate(basemodel.image_encoder_layers):
                if isinstance(layer, nn.DataParallel):
                    layer = layer.module
                new_layers.append(layer)
                self.add_module(f'image_encoder_layer_{i}', layer)
            basemodel.image_encoder_layers = new_layers

    def forward(self, c, s):
        # 模拟模型内部的特征提取和前向传播
        self.basemodel.c = c
        self.basemodel.s = s
        self.basemodel.forward()
        return self.basemodel.cs


if __name__ == '__main__':
    opt = TestOptions().parse()  # 获取测试参数

    # ==========================================
    # 核心功能 1: 随意控制图像尺寸是 256 还是 512
    # ==========================================
    # ★ 你可以在这里直接修改分辨率大小，框架会自动处理
    # 直接读取命令行的 crop_size 设置作为基准分辨率
    TEST_IMAGE_SIZE = opt.crop_size

    # 确保 load_size 和 crop_size 保持一致，避免不必要的缩放误差
    opt.load_size = TEST_IMAGE_SIZE

    # 针对测试环境锁死的一些关键参数
    opt.num_threads = 0
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True
    opt.display_id = -1

    # 初始化数据集和模型
    dataset = create_dataset(opt)
    model = create_model(opt)
    # ---------- ✨ 新增的动态拦截代码 ✨ ----------
    # 为了避免测试阶段加载不存在的损失投影器权重，使得模型纯净且 thop 统计精准，在此强行剔除
    if hasattr(model, 'model_names'):
        if 'projector_c' in model.model_names:
            model.model_names.remove('projector_c')
        if 'projector_s' in model.model_names:
            model.model_names.remove('projector_s')
    # ---------------------------------------------
    model.setup(opt)

    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))
    if opt.load_iter > 0:
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))

    if opt.eval:
        model.eval()

    # ==========================================
    # 核心功能 2 & 3: 使用 thop 计算 Params 和 FLOPs
    # ==========================================
    print(f"\n---> 正在初始化 thop 计算 (分辨率: {TEST_IMAGE_SIZE}x{TEST_IMAGE_SIZE}) ...")

    # 将模型包装并将其所有组件强制推送到正确的设备 (GPU) 上
    wrapper = ModelWrapper(model).to(model.device)
    wrapper.eval()

    # 生成用于 thop 计算的 dummy 假数据
    dummy_c = torch.randn(1, 3, TEST_IMAGE_SIZE, TEST_IMAGE_SIZE).to(model.device)
    dummy_s = torch.randn(1, 3, TEST_IMAGE_SIZE, TEST_IMAGE_SIZE).to(model.device)

    with torch.no_grad():
        # profile 计算 MACs (乘加操作次数) 和 Params
        macs, params = profile(wrapper, inputs=(dummy_c, dummy_s), verbose=False)
        flops = macs * 2  # 工业界标准：FLOPs 通常是 MACs 的两倍
        flops_str, params_str = clever_format([flops, params], "%.3f")

    # ==========================================
    # 核心功能 4: 精确计算平均推理时间
    # ==========================================
    total_time = 0.0
    count = 0

    # GPU 预热，消除显存初始分配时的耗时波动，提升时间统计准确性
    if torch.cuda.is_available():
        print("---> 正在预热 GPU ...")
        with torch.no_grad():
            for _ in range(3):
                wrapper(dummy_c, dummy_s)
        torch.cuda.synchronize()

    print("---> 开始执行推理 ...\n")
    for i, data in enumerate(dataset):
        if i >= opt.num_test:
            break

        model.set_input(data)

        # 记录开始时间 (使用 synchronize 确保 GPU 任务完全同步)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.time()

        # 运行推理 (此时已经没有 DataParallel 壳子了，会稍快一点)
        model.test()

        # 记录结束时间
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.time() - start_time

        total_time += elapsed
        count += 1

        visuals = model.get_current_visuals()
        img_path = model.get_image_paths()

        if i % 5 == 0:
            print('Processing (%04d/%d) | Time: %.4f s | Path: %s' % (i, len(dataset), elapsed,
                                                                      os.path.basename(img_path[0])))

        save_images(webpage, visuals, img_path, width=opt.display_winsize)

    webpage.save()

    avg_time = total_time / count if count > 0 else 0.0

    # ==========================================
    # 核心功能 6: 打印具体读取的 .pth 文件及其参数量拆解
    # ==========================================
    print("\n" + "=" * 55)
    print(" " * 10 + "Detailed Parameters per .pth File")
    print("=" * 55)

    total_calculated_params = 0

    # 1. 遍历并打印 model_names 中定义的所有子网络 (.pth) 的参数量
    for name in model.model_names:
        if isinstance(name, str):
            net = getattr(model, 'net_' + name)
            # 如果外面套了 DataParallel，需要剥离以便准确统计
            if isinstance(net, torch.nn.DataParallel):
                net = net.module

            # 计算当前子网络的参数量
            num_params = sum(p.numel() for p in net.parameters())
            total_calculated_params += num_params

            # 还原框架底层读取该网络时使用的 .pth 文件名
            load_suffix = 'iter_%d' % opt.load_iter if opt.load_iter > 0 else opt.epoch
            pth_filename = f"{load_suffix}_net_{name}.pth"

            print(f" File: {pth_filename:<25} | Params: {num_params / 1e6:8.3f} M")

    # 2. 单独打印预训练的图像编码器 (VGG) 的参数量
    if hasattr(model, 'image_encoder_layers'):
        vgg_params = 0
        for layer in model.image_encoder_layers:
            if isinstance(layer, torch.nn.DataParallel):
                layer = layer.module
            vgg_params += sum(p.numel() for p in layer.parameters())
        total_calculated_params += vgg_params

        # 获取编码器的具体文件名
        vgg_path = os.path.basename(opt.image_encoder_path) if hasattr(opt, 'image_encoder_path') else "vgg_encoder.pth"
        print(f" File: {vgg_path:<25} | Params: {vgg_params / 1e6:8.3f} M")

    print("-" * 55)
    print(f" Check Total Params : {total_calculated_params / 1e6:8.3f} M")
    print("=" * 55 + "\n")

    # ==========================================
    # 核心功能 5: 输出总结图表 (格式化输出)
    # ==========================================
    print("\n" + "=" * 55)
    print(" " * 14 + "Model Performance Summary")
    print("=" * 55)
    print(f" Image Resolution :  {TEST_IMAGE_SIZE} x {TEST_IMAGE_SIZE}")
    print(f" Total Parameters :  {params_str}")
    print(f" Average FLOPs    :  {flops_str}")
    print(f" Average Time     :  {avg_time:.4f} s / image")
    print(f" Total Tested     :  {count} images")
    print("=" * 55 + "\n")

"""
python test_P_F.py \
--content_path /root/autodl-tmp/test05/content/ \
--style_path /root/autodl-tmp/test05/style/  \
--name AdaAttN_test \
--model adaattn \
--dataset_mode unaligned \
--load_size 512 \
--crop_size 512 \
--image_encoder_path /root/autodl-tmp/P_AND_F/AesFA-main/vgg_normalised.pth \
--gpu_ids 0 \
--skip_connection_3 \
--shallow_layer

python test_P_F.py \
--content_path /mnt/harddisk2/Zhangmengge/codespace/Paper_two/dataset/contents/ \
--style_path /mnt/harddisk2/Zhangmengge/codespace/Paper_two/dataset/style/ \
--results_dir ./out512 \
--name AdaAttN_test \
--model adaattn \
--dataset_mode unaligned \
--load_size 512 \
--crop_size 512 \
--image_encoder_path /mnt/harddisk2/Zhangmengge/codespace/MyAttn/MyAtt_GSA_contentEnhance_XS/models/vgg_normalised.pth \
--gpu_ids 0"""