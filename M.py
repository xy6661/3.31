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
        b, c, h_s, w_s = V.size()

        key_pooling_style = self.v_style_pooling(mean_variance_norm(style)).view(b, 1, h_s*w_s)#b,1,h*w
        style_weights=self.sm(key_pooling_style)

        global_style_vector = torch.bmm(V.view(b, c, w_s * h_s), style_weights.permute(0, 2, 1))

        Q_guide = self.Q_gudide_conv(mean_variance_norm(content))
        b, _, h_c, w_c = Q_guide.size() # 使用 h_c, w_c 表示 content 的高和宽

        gamma = self.mlp_guidance1(global_style_vector.unsqueeze(-1))
        beta = self.mlp_guidance2(global_style_vector.unsqueeze(-1))
        locala_guidance = gamma * Q_guide + beta


        Q1 = Q_guide.view(b, -1, h_c * w_c).permute(0, 2, 1)
        # locala_guidance 和 Q_guide 维度相同，可以直接 reshape
        Q2 = locala_guidance.view(b, -1, h_c * w_c).permute(0, 2, 1)
        Q = Q1 + Q2

        K = self.k(style).view(b, -1, w_s * h_s).contiguous()
        v_v = V.view(b, -1, w_s * h_s).permute(0, 2, 1)

        energy = torch.bmm(Q, K)
        S = self.sm(energy)
        out_flat = torch.bmm(S, v_v)

        out = out_flat.permute(0, 2, 1).contiguous().view(b, -1, h_c, w_c)

        style_features = mean_variance_norm(content) + out
        return style_features