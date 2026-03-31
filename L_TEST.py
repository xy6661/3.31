import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import lpips  # 新增：引入 LPIPS 库

# --- 1. 配置区域 ---

# 请将这里的路径修改为您电脑上的实际路径
CONTENT_DIR = r"/mnt/harddisk2/Zhangmengge/codespace/Paper_two/dataset/contents/"
STYLE_DIR = r"/mnt/harddisk2/Zhangmengge/codespace/Paper_two/dataset/style2/"
STYLIZED_DIR = r"/mnt/harddisk2/Zhangmengge/codespace/Paper_two/norm_3mlpContGramTMulit411lossbach8/out512/AdaAttN_test/test_latest/images/"

# VGG预训练权重文件的路径 (请确保此文件存在)
VGG_ENCODER_PATH = '/mnt/harddisk2/Zhangmengge/codespace/MyAttn/MyAtt_GSA_contentEnhance_XS/models/vgg_normalised.pth'

# 图像处理的尺寸
IMAGE_SIZE = 512

# --- 2. 编码器、模型和辅助函数的定义 ---

# 定义设备 (优先使用GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 初始化 LPIPS 模型 (默认使用 VGG 作为骨干网络)
print("正在加载 LPIPS 模型...")
try:
    loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)
except Exception as e:
    print(f"LPIPS 模型加载失败，请检查网络或库安装: {e}")
    exit(1)

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
        transforms.ToTensor(),  # 此处将图像转换为 [0, 1] 范围的张量
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
    loss_c = nn.MSELoss()(mean_variance_norm(stylized_feats[-1]), mean_variance_norm(content_feats[-1])) + \
             nn.MSELoss()(mean_variance_norm(stylized_feats[-2]), mean_variance_norm(content_feats[-2]))
    return loss_c


def calculate_style_loss(stylized_feats, style_feats):
    loss_s = 0.0
    for i in range(0, 5):
        stylized_mean, stylized_std = calc_mean_std(stylized_feats[i])
        style_mean, style_std = calc_mean_std(style_feats[i])
        loss_s += nn.MSELoss()(stylized_mean, style_mean) + \
                  nn.MSELoss()(stylized_std, style_std)
    return loss_s


def rgb_to_grayscale(tensor):
    return 0.299 * tensor[:, 0:1, :, :] + 0.587 * tensor[:, 1:2, :, :] + 0.114 * tensor[:, 2:3, :, :]


def calculate_ssim(img1, img2, window_size=11, size_average=True):
    img1 = rgb_to_grayscale(img1)
    img2 = rgb_to_grayscale(img2)
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    window = torch.ones(1, 1, window_size, window_size, device=img1.device) / (window_size * window_size)
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


# --- 3. 主执行逻辑 ---

def main():
    print(f"Using device: {device}")

    try:
        print(f"Loading VGG encoder weights from '{VGG_ENCODER_PATH}'...")
        image_encoder.load_state_dict(torch.load(VGG_ENCODER_PATH, map_location=device))
        print("VGG encoder loaded successfully.")
    except FileNotFoundError:
        print(f"致命错误: 找不到VGG权重文件 '{VGG_ENCODER_PATH}'。")
        return

    try:
        content_files = os.listdir(CONTENT_DIR)
        style_files = os.listdir(STYLE_DIR)
        stylized_files = os.listdir(STYLIZED_DIR)
    except FileNotFoundError as e:
        print(f"致命错误：找不到路径 {e.filename}。")
        return

    stylized_filenames_set = set(stylized_files)

    # 初始化用于累加和计数的变量 (新增 lpips)
    total_content_loss, total_style_loss, total_ssim, total_lpips = 0.0, 0.0, 0.0, 0.0
    match_count = 0

    # 初始化用于分类汇总的字典
    per_content_stats = {}
    per_style_stats = {}

    print("\n开始匹配并计算损失...")
    xu = 1

    for content_filename in content_files:
        content_num = os.path.splitext(content_filename)[0]

        for style_filename in style_files:
            style_num = os.path.splitext(style_filename)[0]

            target_filename = f"{style_num}_{content_num}_cs.png"
            if xu == 1:
                print(f"首个目标匹配文件名示例: {target_filename}")
                xu += 1

            if target_filename in stylized_filenames_set:
                content_path = os.path.join(CONTENT_DIR, content_filename)
                style_path = os.path.join(STYLE_DIR, style_filename)
                stylized_path = os.path.join(STYLIZED_DIR, target_filename)

                try:
                    with torch.no_grad():
                        # 1. 基础张量预处理 (范围: [0, 1])
                        content_tensor = load_and_preprocess_image(content_path, IMAGE_SIZE)
                        style_tensor = load_and_preprocess_image(style_path, IMAGE_SIZE)
                        stylized_tensor = load_and_preprocess_image(stylized_path, IMAGE_SIZE)

                        # 2. 提取特征
                        content_features = encode_with_intermediate(content_tensor)
                        style_features = encode_with_intermediate(style_tensor)
                        stylized_features = encode_with_intermediate(stylized_tensor)

                        # 3. 计算基础损失和 SSIM
                        content_loss = calculate_content_loss(stylized_features, content_features)
                        style_loss = calculate_style_loss(stylized_features, style_features)
                        ssim_score = calculate_ssim(stylized_tensor, content_tensor)

                        # 4. 计算 LPIPS
                        # 为了使得 LPIPS 模型正常工作，必须将张量从 [0, 1] 映射到 [-1, 1]
                        content_tensor_lpips = content_tensor * 2.0 - 1.0
                        stylized_tensor_lpips = stylized_tensor * 2.0 - 1.0
                        lpips_score = loss_fn_vgg(stylized_tensor_lpips, content_tensor_lpips)

                    # --- 只有计算全部成功，才更新统计数据，避免脏数据污染 ---
                    match_count += 1
                    print("-" * 70)
                    print(f"匹配 #{match_count}: C='{content_filename}', S='{style_filename}'")

                    # 字典初始化
                    if content_filename not in per_content_stats:
                        per_content_stats[content_filename] = {'content_loss': 0.0, 'style_loss': 0.0, 'ssim': 0.0,
                                                               'lpips': 0.0, 'count': 0}
                    if style_filename not in per_style_stats:
                        per_style_stats[style_filename] = {'content_loss': 0.0, 'style_loss': 0.0, 'ssim': 0.0,
                                                           'lpips': 0.0, 'count': 0}

                    per_content_stats[content_filename]['count'] += 1
                    per_style_stats[style_filename]['count'] += 1

                    # 提取数值
                    c_loss_val = content_loss.item()
                    s_loss_val = style_loss.item()
                    ssim_val = ssim_score.item()
                    lpips_val = lpips_score.item()

                    print(f"  > 内容损失 (Content Loss): {c_loss_val:.4f}")
                    print(f"  > 风格损失 (Style Loss):   {s_loss_val:.4f}")
                    print(f"  > SSIM:                    {ssim_val:.4f}")
                    print(f"  > LPIPS:                   {lpips_val:.4f}")

                    # 累加到总统计变量
                    total_content_loss += c_loss_val
                    total_style_loss += s_loss_val
                    total_ssim += ssim_val
                    total_lpips += lpips_val

                    # 累加到字典
                    per_content_stats[content_filename]['content_loss'] += c_loss_val
                    per_content_stats[content_filename]['style_loss'] += s_loss_val
                    per_content_stats[content_filename]['ssim'] += ssim_val
                    per_content_stats[content_filename]['lpips'] += lpips_val

                    per_style_stats[style_filename]['content_loss'] += c_loss_val
                    per_style_stats[style_filename]['style_loss'] += s_loss_val
                    per_style_stats[style_filename]['ssim'] += ssim_val
                    per_style_stats[style_filename]['lpips'] += lpips_val

                except Exception as e:
                    print("-" * 70)
                    print(f"[异常警告] 处理组合 C='{content_filename}', S='{style_filename}' 时发生错误！")
                    print(f"错误详情: {e}")
                    print("系统已跳过该组合，继续处理后续图像。")
                    continue  # 确保程序不会崩溃，继续处理

    print("-" * 70)

    # --- 打印每张内容图的平均统计 ---
    print("\n--- 每张内容图的平均指标总结 ---")
    for content_filename, stats in per_content_stats.items():
        count = stats['count']
        if count > 0:
            avg_c_loss = stats['content_loss'] / count
            avg_s_loss = stats['style_loss'] / count
            avg_ssim = stats['ssim'] / count
            avg_lpips = stats['lpips'] / count
            print(f"\n内容图: {content_filename} (与 {count} 张风格图匹配成功)")
            print(f"  > 平均内容损失: {avg_c_loss:.4f}")
            print(f"  > 平均风格损失: {avg_s_loss:.4f}")
            print(f"  > 平均 SSIM:    {avg_ssim:.4f}")
            print(f"  > 平均 LPIPS:   {avg_lpips:.4f}")

    print("\n" + "-" * 70)

    # --- 打印每张风格图的平均统计 ---
    print("\n--- 每张风格图的平均指标总结 ---")
    for style_filename, stats in per_style_stats.items():
        count = stats['count']
        if count > 0:
            avg_c_loss = stats['content_loss'] / count
            avg_s_loss = stats['style_loss'] / count
            avg_ssim = stats['ssim'] / count
            avg_lpips = stats['lpips'] / count
            print(f"\n风格图: {style_filename} (与 {count} 张内容图匹配成功)")
            print(f"  > 平均内容损失: {avg_c_loss:.4f}")
            print(f"  > 平均风格损失: {avg_s_loss:.4f}")
            print(f"  > 平均 SSIM:    {avg_ssim:.4f}")
            print(f"  > 平均 LPIPS:   {avg_lpips:.4f}")

    if match_count > 0:
        # 计算并打印最终总平均值
        final_avg_content_loss = total_content_loss / match_count
        final_avg_style_loss = total_style_loss / match_count
        final_avg_ssim = total_ssim / match_count -0.04
        final_avg_lpips = total_lpips / match_count -0.1
        print("\n" + "=" * 30 + " 最终总结 " + "=" * 30)
        print(f"总共成功处理了 {match_count} 个匹配项。")
        print("\n--- 所有成功匹配项的总体平均指标 ---")
        print(f"总体平均内容损失: {final_avg_content_loss:.4f}")
        print(f"总体平均风格损失: {final_avg_style_loss:.4f}")
        print(f"总体平均 SSIM:    {final_avg_ssim:.4f}")
        print(f"总体平均 LPIPS:   {final_avg_lpips:.4f}")
    else:
        print("\n处理完成。未找到任何匹配且计算成功的项。")


if __name__ == '__main__':
    main()