import torch.nn as nn
import torch
import numpy as np
from torchinfo import summary

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Pos2Weight(nn.Module):
    def __init__(self, msfa_size, kernel_size, meta_channel=128):
        super(Pos2Weight, self).__init__()
        self.msfa_size = msfa_size
        self.kernel_size = kernel_size
        self.inC = 1
        self.outC = msfa_size**2

        self.meta_block = nn.Sequential(
            nn.Linear(2, meta_channel),
            nn.ReLU(inplace=True),
            nn.Linear(meta_channel, kernel_size * kernel_size * self.inC * self.outC)
        )

    def forward(self, x):
        output = self.meta_block(x)
        return output

class MosaicConvModule(nn.Module):
    def __init__(self, msfa_size, kernel_size):
        super(MosaicConvModule, self).__init__()
        self.msfa_size = msfa_size
        self.kernel_size = kernel_size
        self.P2W = Pos2Weight(msfa_size=msfa_size, kernel_size=kernel_size)

    def input_matrix_wpn(self, msfa_size=4):
        h_offset_coord = torch.zeros(msfa_size, msfa_size, 1)  # height坐标
        w_offset_coord = torch.zeros(msfa_size, msfa_size, 1)  # width坐标
        for i in range(0, msfa_size):
            h_offset_coord[i, :, 0] = (i + 1) / msfa_size
            w_offset_coord[:, i, 0] = (i + 1) / msfa_size

        pos_mat = torch.cat((h_offset_coord, w_offset_coord), 2)
        pos_mat = pos_mat.contiguous().view(1, -1, 2)
        return pos_mat # [b, msfa_size**2, 2]

    def forward(self, input_raw):
        '''
        Args:
            input_raw: torch.Tensor, [b,1,h,w]
        Returns:
            Raw_conv_buff: torch.Tensor, [b,c,h,w]
        '''
        B, C, H, W = input_raw.shape
        pos_mat = self.input_matrix_wpn(self.msfa_size).to(device) # pos_mat: [1,16,2]， 生成msfa**2个独特坐标

        # 一个坐标确定一个卷积核，一个msfa周期包含一个共计16种卷积核的filter：local_weight: [16,400], 400 = in_c * out_c * kernel_h * kernel_w (1*16*5*5)
        local_weight = self.P2W(pos_mat.view(pos_mat.size(1), -1))
        local_weight = local_weight.view(self.msfa_size, self.msfa_size,
                                         self.msfa_size ** 2 * self.kernel_size * self.kernel_size)  # local_weight: [4,4,400]
        #local_weight1 = local_weight.clone()  # local_weights1: [4,4,400]
        cols = nn.functional.unfold(input_raw, self.kernel_size,
                                    padding=(self.kernel_size - 1) // 2)  # cols: [1,25,480*512],把input_raw切成供卷积使用的多个窗口
        cols = cols.contiguous().view(cols.size(0), 1,
                                      cols.size(1), cols.size(2), 1).permute(0, 1, 3, 4, 2).contiguous()  # cols: [1,25,480*512] -> [1,1,480*512,1,25]

        h_pattern_n = 1
        # This h_pattern_n can divide H / msfa_size as a int
        local_weight = local_weight.repeat(h_pattern_n, int(W / self.msfa_size), 1)  # local_weight1: [4,4,400]->[4,512,400], repeat使得dim=1那一维度的数据重复int(W / self.msfa_size)次
        local_weight = local_weight.view(h_pattern_n * self.msfa_size * W,
                                           self.msfa_size ** 2 * self.kernel_size * self.kernel_size)  # local_weight1: [2048,400]
        local_weight = local_weight.contiguous().view(1, h_pattern_n * self.msfa_size * W, -1,
                                                        self.msfa_size ** 2)  # local_weight1: [2048,400]->[1,2048,25,16]
        # 这种特殊卷积的要点是：每个filter对应一个pos_mat上的坐标，而pos_mat的坐标呈现周期性(每4*4为一个循环)，这就使得卷积的权重就呈现这种周期性
        # 这部分正式开始做卷积运算，实现手段是nn.functional.unfold和matmul
        # 这个循环其实是逐行处理（H维，480/4=120）
        for i in range(0, int(H / self.msfa_size / h_pattern_n)):
            # 取cols的一部分，cols_buff: [1,1,480*512,1,25]->[1,2048,1,25]
            cols_buff = cols[:, 0, i * self.msfa_size * h_pattern_n * W: (i + 1) * self.msfa_size * h_pattern_n * W, :, :]
            if i == 0:
                Raw_conv_buff = torch.matmul(cols_buff,
                                             local_weight)  # cols_buff:[1,2048,1,25] * local_weight1:[1,2048,25,16] = Raw_conv_buff:[1,2048,1,16]
            else:
                Raw_conv_buff = torch.cat([Raw_conv_buff, torch.matmul(cols_buff, local_weight)],
                                          dim=-3)  # Raw_conv_buff: [1,cat,1,16]

        Raw_conv_buff = torch.unsqueeze(Raw_conv_buff, 0)  # Raw_conv_buff:[1,480*512,1,16]->[1,1,480*512,1,16]
        Raw_conv_buff = Raw_conv_buff.permute(0, 1, 4, 2, 3)  # Raw_conv_buff:[1,1,480*512,1,16]->[1,1,16,480*512,1]
        Raw_conv_buff = Raw_conv_buff.contiguous().view(B, 1, 1, self.msfa_size ** 2, H, W)  # Raw_conv_buff: [1,1,16,480*512,1]->[1,1,1,16,480,512]
        Raw_conv_buff = Raw_conv_buff.contiguous().view(B, self.msfa_size ** 2, H, W)  # Raw_conv_buff: [1,1,1,16,480,512]->[1,16,480,512]
        return Raw_conv_buff

if __name__ == '__main__':
    model = MosaicConvModule(msfa_size=4, kernel_size=5).cuda()
    raw = torch.randn(1,1,480,512).cuda()
    summary(model, input_data=raw)