import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

class GlobalContextBlock(nn.Module):
    """全局上下文语义块"""
    def __init__(self, in_channels, num_heads=3):
        super().__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.W_theta = nn.Linear(in_channels, in_channels, bias=False)
        self.W_phi   = nn.Linear(in_channels, in_channels, bias=False)
        self.W_heads = nn.ModuleList([
            nn.Linear(in_channels, in_channels, bias=False)
            for _ in range(num_heads)
        ])
        self.relu = nn.ReLU()

    def forward(self, X):  # X: [N, C]
        theta_X = self.W_theta(X)  # [N, C]
        phi_X   = self.W_phi(X)    # [N, C]
        e = torch.matmul(theta_X, phi_X.transpose(0,1))  # [N, N]
        A = F.softmax(e, dim=1)   # row-wise
        out = 0
        for Wk in self.W_heads:
            Xk = Wk(X)
            out = out + self.relu(A @ Xk)
        return out

class LocalTopologyBlockWithAngle(nn.Module):
    """局部拓扑关系块，加入极角差编码"""
    def __init__(self, in_channels, num_heads=3, leaky_slope=0.2):
        super().__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        # 消息维度 2C + 2
        self.W_gamma = nn.Linear(2*in_channels+2, 1, bias=False)
        self.W_heads = nn.ModuleList([
            nn.Linear(in_channels, in_channels, bias=False)
            for _ in range(num_heads)
        ])
        self.leaky = nn.LeakyReLU(leaky_slope)
        self.relu = nn.ReLU()

    def forward(self, X, phis):
        return self._ltb_forward(X, phis)

    def _ltb_forward(self, X, phis):
        N, C = X.shape
        # 使用广播优化拼接
        Xu = X.view(N, 1, C).expand(-1, N, -1)
        Xv = X.view(1, N, C).expand(N, -1, -1)

        delta = phis.view(N, 1) - phis.view(1, N)
        cos_d = torch.cos(delta).unsqueeze(-1)
        sin_d = torch.sin(delta).unsqueeze(-1)

        # 拼接时确保与模型参数类型一致
        pairs = torch.cat([Xu, Xv, cos_d, sin_d], dim=-1).to(self.W_gamma.weight.dtype)

        # 计算关联分
        e = self.leaky(self.W_gamma(pairs)).squeeze(-1)  # [N, N]
        A = F.softmax(e, dim=0)  # column-wise
        out = 0
        for Wk in self.W_heads:
            Xk = Wk(X)  # [N, C]
            out = out + self.relu(A @ Xk)
        return out

    # def forward(self, X, phis):
        # N, C = X.shape
        # # 使用广播优化拼接
        # Xu = X.view(N, 1, C).expand(-1, N, -1)
        # Xv = X.view(1, N, C).expand(N, -1, -1)
        #
        # delta = phis.view(N, 1) - phis.view(1, N)
        # cos_d = torch.cos(delta).unsqueeze(-1)
        # sin_d = torch.sin(delta).unsqueeze(-1)
        #
        # # 拼接时确保与模型参数类型一致
        # pairs = torch.cat([Xu, Xv, cos_d, sin_d], dim=-1).to(self.W_gamma.weight.dtype)
        #
        # # 计算关联分
        # e = self.leaky(self.W_gamma(pairs)).squeeze(-1)  # [N, N]
        # A = F.softmax(e, dim=0)  # column-wise
        # out = 0
        # for Wk in self.W_heads:
        #     Xk = Wk(X)  # [N, C]
        #     out = out + self.relu(A @ Xk)
        # return out

def make_bilinear_weights(stride, channels):
    """
    为 depthwise ConvTranspose2d 生成一个等价于双线性插值的权重。
    由于在 ConvTranspose2d 中我们用 groups=channels，每个通道单独插值，
    所以权重张量形状应为 [channels, 1, k, k] 而不是 [channels, channels, k, k]。
    """
    # 参考：https://stackoverflow.com/questions/59128322/how-to-convert-bilinear-up-sampling-to-transposed-convolution
    factor = (2 * stride - stride % 2) / (2.0 * stride)
    og = torch.arange(stride).float()
    filt = (1 - torch.abs(og / factor - (stride - 1) / (2 * factor)))  # [stride]
    bilinear_kernel = filt.unsqueeze(0) * filt.unsqueeze(1)  # [stride, stride]

    # 构造一个 shape=[channels, 1, stride, stride] 的张量
    weight = torch.zeros(channels, 1, stride, stride)
    for i in range(channels):
        weight[i, 0, :, :] = bilinear_kernel

    return weight  # [channels, 1, k, k]

class UpsampleWithTransposedConv(nn.Module):
    """
    用 depthwise ConvTranspose2d 实现等价双线性插值，上采样比例 = stride。
    权重初始化为双线性核，并设置 requires_grad=False，保证完全固定不变。
    """
    def __init__(self, channels, stride=3):
        super().__init__()
        self.stride = stride
        # groups=channels: depthwise 转置卷积：每个通道单独插值
        self.deconv = nn.ConvTranspose2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=stride,
            stride=stride,
            padding=0,
            bias=False,
            groups=channels
        )
        # 生成并复制固定的双线性核
        weight = make_bilinear_weights(stride, channels)  # [channels, 1, k, k]
        self.deconv.weight.data.copy_(weight)

        # 不让这层的权重参与梯度更新
        for param in self.deconv.parameters():
            param.requires_grad = False

    def forward(self, x):
        """
        x: [B, C, H', W']
        返回: [B, C, H'*stride, W'*stride]
        """
        return self.deconv(x)

class ARGRModule(nn.Module):
    """自适应关系图推理模块，自动计算 phis"""
    def __init__(self, in_channels, num_heads=3):
        super().__init__()
        self.gcb = GlobalContextBlock(in_channels, num_heads)
        self.ltb = LocalTopologyBlockWithAngle(in_channels, num_heads)
        self.fuse = nn.Conv2d(in_channels*3, in_channels, kernel_size=1)
        self.downsample = nn.MaxPool2d(2, 2)  # 下采样
        self.upsample = UpsampleWithTransposedConv(channels=in_channels, stride=2)

    def forward(self, feat_map):
        B, C, H, W = feat_map.shape

        # 0. 确保输入类型与模型参数一致（如 float16）
        if feat_map.dtype != self.fuse.weight.dtype:
            feat_map = feat_map.to(self.fuse.weight.dtype)

        # 1. 下采样输入特征图（降低 N = H * W）
        feat_down = self.downsample(feat_map)  # [B, C, H', W']
        H_down, W_down = feat_down.shape[2:]

        # 2. 基于下采样后的特征图计算极角 phis
        device = feat_down.device
        dtype = feat_down.dtype
        ys = torch.arange(H_down, device=device).view(H_down, 1).to(dtype)  # [H_down, 1]
        xs = torch.arange(W_down, device=device).view(1, W_down).to(dtype)  # [1, W_down]
        cy, cx = (H_down - 1) / 2.0, (W_down - 1) / 2.0
        phi = torch.atan2(ys - cy, xs - cx)
        phis = phi.reshape(-1)

        # 3. 转换为节点形式
        nodes = feat_down.permute(0, 2, 3, 1).reshape(B, -1, C)  # [B, N', C]

        # 5. 循环里对每个样本单独 checkpoint
        outs = []
        for b in range(B):
            Xb = nodes[b]  # [N', C]

            # 5.1 纯函数包装
            def run_gcb(x):
                return self.gcb(x)

            def run_ltb(x, p):
                return self.ltb(x, p)

            # 5.2 分别做一次重算
            g_out_b = checkpoint(run_gcb, Xb)  # [N', C]
            l_out_b = checkpoint(run_ltb, Xb, phis)  # [N', C]

            fused_b = torch.cat([Xb, g_out_b, l_out_b], dim=-1)  # [N', 3C]
            outs.append(fused_b)

        # 6. 拼回特征图 [B, 3C, H', W']
        fused = torch.stack(outs, dim=0) \
            .reshape(B, H_down, W_down, 3 * C) \
            .permute(0, 3, 1, 2)

        # 7. 1×1 融合回 [B, C, H', W']
        out_down = self.fuse(fused)

        # 8. 上采样到 [B, C, H, W]
        out_up = self.upsample(out_down)
        if out_up.shape != feat_map.shape:
            out_up = F.interpolate(out_up, size=(H, W), mode='nearest')

        return out_up



# IoU 损失保持不变
# def iou_loss(pred_boxes, gt_boxes, eps=1e-6):
#     x1 = torch.max(pred_boxes[:,0], gt_boxes[:,0])
#     y1 = torch.max(pred_boxes[:,1], gt_boxes[:,1])
#     x2 = torch.min(pred_boxes[:,2], gt_boxes[:,2])
#     y2 = torch.min(pred_boxes[:,3], gt_boxes[:,3])
#     inter = (x2-x1).clamp(0) * (y2-y1).clamp(0)
#     area_p = (pred_boxes[:,2]-pred_boxes[:,0])*(pred_boxes[:,3]-pred_boxes[:,1])
#     area_g = (gt_boxes[:,2]-gt_boxes[:,0])*(gt_boxes[:,3]-gt_boxes[:,1])
#     union = area_p + area_g - inter + eps
#     iou = inter/union
#     return torch.mean(1 - iou**2)
