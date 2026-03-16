import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchinfo import summary

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def fft2freq(d1, d2, use_rfft=False):
    '''生成2D频率坐标，并且根据其到零频率分量点(0,0)的距离进行排序，这有助于对频率分量进行分组
    Args:
        d1: height, 第一个维度的大小
        d2: width, 第二个维度的大小
        use_rfft: 是否用实数fft
    Returns:
        freq_hw: shape=[N, 2], 2D频率坐标(u,v)，以供从卷积核里取频率分量
        sorted_coords: shape=[N, 2], 经过排序之后的2D频率坐标(u,v)，照着取就能取到低频——>高频分量
    '''
    # 创建行和列的频率分量
    freq_h = torch.fft.fftfreq(d1) # freq_h: [d1]
    if use_rfft:
        freq_w = torch.fft.rfftfreq(d2) # freq_w: [d2//2+1]
    else:
        freq_w = torch.fft.fftfreq(d2) # freq_w: [d2]

    # 创建频率坐标的2D网格
    freq_hw = torch.stack(torch.meshgrid(freq_h, freq_w, indexing='ij'), dim=-1) # freq_hw: [d1, d2//2+1, 2]
    # 在频率空间中计算L2范数（即到零频率分量(0,0)的距离）
    dist = torch.norm(freq_hw, dim=-1) # dist: [d1, d2//2+1]

    # 对距离进行排序并获取原始索引
    # sorted_dist: 1D张量，根据dist中l2范数的值，由低到高排序
    # indices: 1D张量，排序之前dist的索引
    sorted_dist, indices = torch.sort(dist.view(-1))

    # 获取排序之后，距离所对应的2D坐标
    if use_rfft:
        d2 = d2 // 2 + 1
    sorted_coords = torch.stack([indices // d2, indices % d2], dim=-1)

    return sorted_coords, freq_hw


class StarReLU(nn.Module):
    """
    StarReLU: s * relu(x) ** 2 + b
    """

    def __init__(self, scale_value=1.0, bias_value=0.0,
                 scale_learnable=True, bias_learnable=True,
                 mode=None, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.relu = nn.ReLU(inplace=inplace)
        self.scale = nn.Parameter(scale_value * torch.ones(1),
                                  requires_grad=scale_learnable)
        self.bias = nn.Parameter(bias_value * torch.ones(1),
                                 requires_grad=bias_learnable)

    def forward(self, x):
        return self.scale * self.relu(x) ** 2 + self.bias

class KernelAttention(nn.Module):
    def __init__(
            self,
            in_channels,
            #out_channels,
            reduction,
            num_experts,
            act_type,
            min_channel=16,
    ):
        super().__init__()
        attention_channel = max(int(in_channels * reduction), min_channel)
        self.fc = nn.Conv2d(in_channels, attention_channel, 1, bias=False)
        self.relu = StarReLU()
        self.kernel_fc = nn.Conv2d(attention_channel, num_experts, 1, bias=True)
        self.temperature = math.sqrt(num_experts)
        self.act_type = act_type
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        if hasattr(self, 'kernel_fc') and isinstance(self.kernel_fc, nn.Conv2d):
            nn.init.normal_(self.kernel_fc.weight, std=1e-6)

    def forward(self, x):
        x = F.adaptive_avg_pool2d(x, 1)  # gap_x: [b,c,1,1]
        x = self.fc(x) # x: [b, attention_channel, 1, 1]
        x = self.relu(x)
        kernel_attn = self.kernel_fc(x) # x: [b, num_experts, 1, 1]
        if self.act_type == 'softmax':
            kernel_attn = F.softmax(kernel_attn / self.temperature, dim=1)
        elif self.act_type == 'sigmoid':
            kernel_attn = torch.sigmoid(kernel_attn / self.temperature * 2 / kernel_attn.size(1))
        return kernel_attn

class FPC(nn.Conv2d):
    def __init__(self,
                 *args,
                 msfa_size,
                 num_experts,
                 convert_param=True,
                 reduction=0.0625,
                 act_type='sigmoid',
                 **kwargs,
                 ):
        super().__init__(*args, **kwargs)
        self.msfa_size = msfa_size
        self.num_experts = num_experts
        self.alpha = min(self.out_channels, self.kernel_size[0], self.kernel_size[1]) // 2 * self.num_experts
        self.kernel_attn = KernelAttention(self.in_channels, reduction, num_experts, act_type)
        self.convert2dftweight(convert_param)

    def convert2dftweight(self, convert_param):
        d1, d2, k1, k2 = self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1]
        freq_indices, _ = fft2freq(d1 * d2, k1 * k2, use_rfft=True) # freq_indices: [d1*d2*k1*k2//2+1, 2]
        weight = self.weight.permute(0, 2, 1, 3).reshape(d1 * d2, k1 * k2)
        weight_rfft = torch.fft.rfft2(weight, dim=(0, 1))
        weight_rfft = (torch.stack([weight_rfft.real, weight_rfft.imag], dim=-1)
                        / (min(self.out_channels, self.kernel_size[0], self.kernel_size[1]) // 2)) # weight_rfft: [d1*k1, d2*k2//2+1, 2]

        if convert_param:
            self.dft_weight = nn.Parameter(weight_rfft, requires_grad=True) # self.dft_weight: [d1*k1, d2*k2//2+1, 2]
            del self.weight

        indices = freq_indices.reshape(self.num_experts, -1, 2) # self.indices: [num_experts, -1, 2]
        self.register_buffer('indices', indices, persistent=True)

    def forward(self, x):
        batch_size, inc, height, width = x.shape
        kernel_attention = self.kernel_attn(x).squeeze(-1) # [b, num_experts, 1]
        DFT_map = torch.zeros((batch_size, self.out_channels * self.in_channels, self.kernel_size[0] * self.kernel_size[1] // 2 + 1, 2), device=x.device)
        if hasattr(self, 'dft_weight'):
            dft_weight = self.dft_weight
        w = dft_weight[self.indices[:, :, 0], self.indices[:, :, 1]][None] # w: [1, num_experts, -1, 2]
        DFT_map[:, self.indices[:, :, 0], self.indices[:, :, 1]] += torch.stack([w[..., 0] * kernel_attention, w[..., 1] * kernel_attention], dim=-1)
        # 2D irfft把动态卷积参数转换回空间域
        adaptive_weight = torch.fft.irfft2(torch.view_as_complex(DFT_map), s=(self.out_channels*self.in_channels, self.kernel_size[0]*self.kernel_size[1]),
                                           dim=(1, 2)).reshape(batch_size, 1,
                                                               self.out_channels, self.in_channels,
                                                               self.kernel_size[0], self.kernel_size[1])

        # 进行卷积
        adaptive_weight = torch.sum(adaptive_weight, dim=1) # adaptive_weight: [b, out_c, in_c, k, k]
        adaptive_weight = adaptive_weight.view(-1, self.in_channels // self.groups, self.kernel_size[0], self.kernel_size[1])
        x = x.reshape(1, -1, height, width)
        output = F.conv2d(x, weight=adaptive_weight, bias=None, stride=self.stride, padding=self.padding,
                          dilation=self.dilation, groups=self.groups * batch_size)
        output = output.view(batch_size, -1, output.size(-2), output.size(-1))
        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1, 1)
        return output

if __name__ == '__main__':
    msfa_size = 4
    model = FPC(msfa_size=4, in_channels=1, out_channels=16, kernel_size=5, num_experts=8, padding=2, bias=True).cuda()
    raw = torch.randn(20,1,480,512).cuda()
    summary(model, input_data=raw)