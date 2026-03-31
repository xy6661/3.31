import torch
import torch.nn as nn
import torch.nn.functional as F


"""我设计了一个对比损失，你帮我判断这个思想是否可行，并且代码是否正确。
<Ic,Is>产生Ics，表示内容图像是Ic，风格图像是Is,风格化的结果Ics保持内容图像的内容，风格改变为Is的风格，
并且<Ics,Ic>产生Icsc，Icsc内容结构来自于Ics，风格来自于Ic，但是又由于Ics的内容结构也是来自于Ic，
所以Icsc在期望的情况下是逼近于Ic的,<Is,Ics>产生Iscs，Iscs内容结构来自于Is，风格来自于Ics，但是又由于Ics的风格来自于Is，
所以Iscs应该更期望逼近于Is的。所以我想将这个思想构建为对比损失。并使用了下面的代码，分析是否合理"""
#对比损失代码块=============================================begin==============================================================
class FeatureProjector(nn.Module):
    def __init__(self, input_dim):
        """  例如，VGG-19 ReLU3_1 的输出通道数为 256。 """
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(input_dim, 256),  # 第一个全连接层
            nn.ReLU(),  # 第一个激活函数 (★★★ 假设)
            nn.Linear(256, 128)  # 第二个全连接层
        )

    def forward(self, features):
        # features 的形状为 [B, C, H, W]
        # (1) 全局平均池化，形状变为 [B, C]
        x = F.adaptive_avg_pool2d(features, (1, 1)).view(features.size(0), -1)

        # (2) 通过 MLP 投影，形状变为 [B, 128]
        projection = self.projector(x)

        # (3) L2 归一化 (★★★ 假设)，这是对比学习的标准实践
        normalized_projection = F.normalize(projection, p=2, dim=1)

        return normalized_projection

class ContentContrastiveLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = 0.2
        self.pc = FeatureProjector(input_dim=256)  # 内容投影器
        # 使用交叉熵损失函数可以高效地计算 InfoNCE 损失
        # log( exp(pos) / (exp(pos) + sum(exp(neg))) ) 等价于对 logits 进行交叉熵计算
        self.criterion = nn.CrossEntropyLoss()
    def forward(self,z_c,z_csc):
        batch_size=z_c.shape[0]
        l_pos_c=torch.sum(z_c * z_csc, dim=1, keepdim=True)
        l_neg_c=torch.matmul(z_csc, z_c.T)

        identity_mask=torch.eye(batch_size,dtype=torch.bool,device=z_c.device)
        l_neg_c=l_neg_c.masked_fill(identity_mask,-float('inf'))

        logits_c=torch.cat([l_pos_c, l_neg_c], dim=1)
        logits_c /= self.temperature

        #标签总是只想第一列（正例），所以标签是全零的向量
        labels_c=torch.zeros(batch_size,dtype=torch.long,device=z_c.device)
        loss_c=self.criterion(logits_c, labels_c)
        return loss_c


class GramProjector(nn.Module):
    def __init__(self, input_dim):
        # input_dim 将是 C * C, 例如 256 * 256 = 65536
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(input_dim, 256),  # 从高维降到256维
            nn.ReLU(inplace=True),
            nn.Linear(256, 128)  # 再降到128维作为最终的嵌入
        )

    def forward(self, flat_gram_vector):
        projection = self.projector(flat_gram_vector)
        # L2 归一化是对比学习的标准做法
        return F.normalize(projection, p=2, dim=1)


# +++ 新增一个辅助模块来计算Gram矩阵 (这个不变) +++
class GramMatrix(nn.Module):
    # ... GramMatrix 代码不变 ...
    def forward(self, features):
        B, C, H, W = features.size()
        features_flat = features.view(B, C, H * W)
        gram = torch.bmm(features_flat, features_flat.transpose(1, 2))
        return gram.div(C * H * W)


# --- 再次修改 StyleContractiveLoss 以使用 GramProjector ---
class StyleContractiveLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = 0.2
        # --- 注意：这里不再实例化GramProjector，因为它将作为网络模块从外部传入 ---
        self.gram_computer = GramMatrix()
        self.criterion = nn.CrossEntropyLoss()

    # projector 将是实例化的 GramProjector 网络
    def forward(self, phi_s, phi_scs, projector):
        batch_size = phi_s.shape[0]

        # 1. 计算Gram矩阵
        gram_s = self.gram_computer(phi_s)
        gram_scs = self.gram_computer(phi_scs)

        # 2. 将Gram矩阵展平为向量 (B, C*C)
        flat_gram_s = gram_s.view(batch_size, -1)
        flat_gram_scs = gram_scs.view(batch_size, -1)

        # 3. 通过投影网络得到低维嵌入
        z_s = projector(flat_gram_s)
        z_scs = projector(flat_gram_scs)

        # 4. 计算相似度矩阵 (logits)，现在在低维空间中进行
        l_pos_s = torch.sum(z_s * z_scs, dim=1, keepdim=True)
        l_neg_s = torch.matmul(z_scs, z_s.T)

        # 5. InfoNCE 损失计算 (与之前完全相同)
        identity_mask = torch.eye(batch_size, dtype=torch.bool, device=phi_s.device)
        l_neg_s = l_neg_s.masked_fill(identity_mask, -float('inf'))
        logits_s = torch.cat([l_pos_s, l_neg_s], dim=1)
        logits_s /= self.temperature
        labels_s = torch.zeros(batch_size, dtype=torch.long, device=phi_s.device)
        loss_s = self.criterion(logits_s, labels_s)
        return loss_s
#对比损失代码块=============================================end==============================================================
