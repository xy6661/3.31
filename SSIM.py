import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import math  # 新增：导入math库用于高斯函数计算

# --- 1. 配置区域 ---

# 请将这里的路径修改为您电脑上的实际路径
CONTENT_DIR = r"/mnt/harddisk2/Zhangmengge/codespace/test/test04/content/"
STYLE_DIR = r"/mnt/harddisk2/Zhangmengge/codespace/test/test04/style/"
STYLIZED_DIR = r"/mnt/harddisk2/Zhangmengge/codespace/Paper_two/norm_3mlpContGramT/results-loss12mlp0503/AdaAttN_test/test_latest/images/"

# VGG预训练权重文件的路径 (请确保此文件存在)
VGG_ENCODER_PATH = '/mnt/harddisk2/Zhangmengge/codespace/MyAttn/MyAtt_GSA_contentEnhance_XS/models/vgg_normalised.pth'

# 图像处理的尺寸
IMAGE_SIZE = 512

# --- 2. VGG 编码器和辅助函数的定义 ---

# 定义设备 (优先使用GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 定义VGG编码器网络结构
image_encoder = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(), nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(64, 64, (3, 3)), nn.ReLU(),  # relu1
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(64, 128, (3, 3)), nn.ReLU(), nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)), nn.ReLU(),  # relu2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(128, 256, (3, 3)), nn.ReLU(), nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)), nn.ReLU(), nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 256, (3, 3)), nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 256, (3, 3)), nn.ReLU(),  # relu3
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 512, (3, 3)), nn.ReLU(), nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)), nn.ReLU(), nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(512, 512, (3, 3)), nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(512, 512, (3, 3)), nn.ReLU(),  # relu4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(512, 512, (3, 3)), nn.ReLU(), nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)), nn.ReLU(), nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(512, 512, (3, 3)), nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(512, 512, (3, 3)), nn.ReLU()  # relu5
)

# 分割模型层
enc_layers = list(image_encoder.children())
enc_1 = nn.Sequential(*enc_layers[:4])
enc_2 = nn.Sequential(*enc_layers[4:11])
enc_3 = nn.Sequential(*enc_layers[11:18])
enc_4 = nn.Sequential(*enc_layers[18:31])
enc_5 = nn.Sequential(*enc_layers[31:44])
image_encoder_layers = [enc_1, enc_2, enc_3, enc_4, enc_5]
for layer in image_encoder_layers:
    layer.to(device)
    layer.eval()
    for param in layer.parameters():
        param.requires_grad = False


def encode_with_intermediate(input_img):
    results = [input_img]
    for i in range(len(image_encoder_layers)):
        func = image_encoder_layers[i]
        results.append(func(results[-1]))
    return results[1:]


def load_and_preprocess_image(image_path, image_size):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0).to(device)


def calc_mean_std(feat, eps=1e-5):
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


def calculate_content_loss(stylized_feats, content_feats):
    assert (stylized_feats[-1].requires_grad is False)
    assert (content_feats[-1].requires_grad is False)
    # 提醒：这里的内容损失计算方式并非最经典的方式（经典方式不带归一化），但我们遵从原代码逻辑
    loss_c = nn.MSELoss()(mean_variance_norm(stylized_feats[-1]), mean_variance_norm(content_feats[-1])) + \
             (nn.MSELoss()(mean_variance_norm(stylized_feats[-2]), mean_variance_norm(content_feats[-2])))

    return loss_c


def calculate_style_loss(stylized_feats, style_feats):
    loss_s = 0.0
    for i in range(len(stylized_feats)):
        stylized_mean, stylized_std = calc_mean_std(stylized_feats[i])
        style_mean, style_std = calc_mean_std(style_feats[i])
        loss_s += nn.MSELoss()(stylized_mean, style_mean) + \
                  nn.MSELoss()(stylized_std, style_std)
    return loss_s


def rgb_to_grayscale(tensor):
    return 0.299 * tensor[:, 0:1, :, :] + 0.587 * tensor[:, 1:2, :, :] + 0.114 * tensor[:, 2:3, :, :]


# =========================================================================================
# --- 中文标注区域：新增函数 ---
# 新增原因：根据SSIM原始论文，计算局部统计量时需要使用高斯加权窗口，而非简单的均值窗口。
# 这个函数就是用来创建论文中指定的 "11x11 circular-symmetric Gaussian" 窗口。
def create_gaussian_window(window_size, sigma):
    # 生成一维高斯分布
    gauss = torch.Tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    gauss = gauss / gauss.sum()  # 归一化

    # 通过一维高斯创建二维高斯核
    gauss_2d = gauss.unsqueeze(1) * gauss.unsqueeze(0)

    # 将核的形状调整为 [1, 1, window_size, window_size] 以便用于卷积
    window = gauss_2d.expand(1, 1, window_size, window_size)
    return window


# =========================================================================================


# =========================================================================================
# --- 中文标注区域：修改函数 ---
# 修改原因：将原始的均值滤波窗口替换为更严谨的高斯加权窗口，以完全符合经典SSIM论文的实现。
def calculate_ssim(img1, img2, window_size=11, size_average=True):
    img1 = rgb_to_grayscale(img1)
    img2 = rgb_to_grayscale(img2)
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # --- 这里是核心修改点 ---
    # 原始代码（均值滤波窗口），现已注释掉:
    # window = torch.ones(1, 1, window_size, window_size, device=img1.device) / (window_size * window_size)

    # 修改后的代码（高斯加权窗口）:
    # 根据论文 Page 7, Section C, "In this paper, we use an 11 × 11 circular-symmetric Gaussian weighting function...
    # ...with standard deviation of 1.5 samples..."
    # 我们调用新函数来创建这个高斯窗口
    window = create_gaussian_window(window_size, 1.5).to(img1.device)
    # -------------------------

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=1)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=1)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=1) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=1) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=1) - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


# =========================================================================================


# --- 3. 主执行逻辑 ---

def main():
    print(f"Using device: {device}")

    try:
        print(f"Loading VGG encoder weights from '{VGG_ENCODER_PATH}'...")
        image_encoder.load_state_dict(torch.load(VGG_ENCODER_PATH, map_location=device))
        print("VGG encoder loaded successfully.")
    except FileNotFoundError:
        print(f"错误: 找不到VGG权重文件 '{VGG_ENCODER_PATH}'。")
        return

    try:
        content_files = os.listdir(CONTENT_DIR)
        style_files = os.listdir(STYLE_DIR)
        stylized_files = os.listdir(STYLIZED_DIR)
    except FileNotFoundError as e:
        print(f"错误：找不到路径 {e.filename}。")
        return

    stylized_filenames_set = set(stylized_files)

    # 初始化用于累加和计数的变量
    total_content_loss, total_style_loss, total_ssim = 0.0, 0.0, 0.0
    match_count = 0

    # 初始化用于分类汇总的字典
    per_content_stats = {}
    per_style_stats = {}  # 新增：用于按风格图统计的字典

    print("\n开始匹配并计算损失...")
    xu = 1

    for content_filename in content_files:
        content_num = os.path.splitext(content_filename)[0]

        for style_filename in style_files:
            style_num = os.path.splitext(style_filename)[0]

            target_filename = f"{style_num}_{content_num}_cs.png"
            if xu == 1:
                print(target_filename)
                xu = xu + 1
            if target_filename in stylized_filenames_set:
                match_count += 1

                # 初始化字典条目 (如果第一次遇到)
                if content_filename not in per_content_stats:
                    per_content_stats[content_filename] = {'content_loss': 0.0, 'style_loss': 0.0, 'ssim': 0.0,
                                                           'count': 0}
                if style_filename not in per_style_stats:
                    per_style_stats[style_filename] = {'content_loss': 0.0, 'style_loss': 0.0, 'ssim': 0.0, 'count': 0}

                # 增加计数
                per_content_stats[content_filename]['count'] += 1
                per_style_stats[style_filename]['count'] += 1

                content_path = os.path.join(CONTENT_DIR, content_filename)
                style_path = os.path.join(STYLE_DIR, style_filename)
                stylized_path = os.path.join(STYLIZED_DIR, target_filename)

                print("-" * 70)
                print(f"匹配 #{match_count}: C='{content_filename}', S='{style_filename}'")

                try:
                    with torch.no_grad():
                        content_tensor = load_and_preprocess_image(content_path, IMAGE_SIZE)
                        style_tensor = load_and_preprocess_image(style_path, IMAGE_SIZE)
                        stylized_tensor = load_and_preprocess_image(stylized_path, IMAGE_SIZE)

                        content_features = encode_with_intermediate(content_tensor)
                        style_features = encode_with_intermediate(style_tensor)
                        stylized_features = encode_with_intermediate(stylized_tensor)

                        content_loss = calculate_content_loss(stylized_features, content_features)
                        style_loss = calculate_style_loss(stylized_features, style_features)
                        ssim_score = calculate_ssim(stylized_tensor, content_tensor)

                    print(f"  > 内容损失 (Content Loss): {content_loss.item():.4f}")
                    print(f"  > 风格损失 (Style Loss):   {style_loss.item():.4f}")
                    print(f"  > SSIM:                    {ssim_score.item():.4f}")

                    # 累加到总统计变量
                    total_content_loss += content_loss.item()
                    total_style_loss += style_loss.item()
                    total_ssim += ssim_score.item()

                    # 累加到按内容图统计的字典
                    per_content_stats[content_filename]['content_loss'] += content_loss.item()
                    per_content_stats[content_filename]['style_loss'] += style_loss.item()
                    per_content_stats[content_filename]['ssim'] += ssim_score.item()

                    # 累加到按风格图统计的字典
                    per_style_stats[style_filename]['content_loss'] += content_loss.item()
                    per_style_stats[style_filename]['style_loss'] += style_loss.item()
                    per_style_stats[style_filename]['ssim'] += ssim_score.item()

                except Exception as e:
                    print(f"\n处理图片时发生错误: {e}")

    print("-" * 70)

    # --- 打印每张内容图的平均统计 ---
    print("\n--- 每张内容图的平均指标总结 ---")
    for content_filename, stats in per_content_stats.items():
        count = stats['count']
        if count > 0:
            avg_c_loss = stats['content_loss'] / count
            avg_s_loss = stats['style_loss'] / count
            avg_ssim = stats['ssim'] / count
            print(f"\n内容图: {content_filename} (与 {count} 张风格图匹配)")
            print(f"  > 平均内容损失: {avg_c_loss:.4f}")
            print(f"  > 平均风格损失: {avg_s_loss:.4f}")
            print(f"  > 平均 SSIM:    {avg_ssim:.4f}")

    print("\n" + "-" * 70)  # 分隔符

    # --- 新增：打印每张风格图的平均统计 ---
    print("\n--- 每张风格图的平均指标总结 ---")
    for style_filename, stats in per_style_stats.items():
        count = stats['count']
        if count > 0:
            avg_c_loss = stats['content_loss'] / count
            avg_s_loss = stats['style_loss'] / count
            avg_ssim = stats['ssim'] / count
            print(f"\n风格图: {style_filename} (与 {count} 张内容图匹配)")
            print(f"  > 平均内容损失: {avg_c_loss:.4f}")
            print(f"  > 平均风格损失: {avg_s_loss:.4f}")
            print(f"  > 平均 SSIM:    {avg_ssim:.4f}")

    if match_count > 0:
        # 计算并打印最终总平均值
        final_avg_content_loss = total_content_loss / match_count
        final_avg_style_loss = total_style_loss / match_count
        final_avg_ssim = total_ssim / match_count
        print("\n" + "=" * 30 + " 最终总结 " + "=" * 30)
        print(f"总共找到并处理了 {match_count} 个匹配项。")
        print("\n--- 所有匹配项的总体平均指标 ---")
        print(f"总体平均内容损失: {final_avg_content_loss:.4f}")
        print(f"总体平均风格损失: {final_avg_style_loss:.4f}")
        print(f"总体平均 SSIM:    {final_avg_ssim:.4f}")
    else:
        print("\n处理完成。未找到任何匹配项。")


if __name__ == '__main__':
    main()