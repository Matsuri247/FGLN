import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import math
from MosaicConvModule import MosaicConvModule
from MyAttention import MACA
from MySIConvModule import FPC

from utils import My_summary

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class ACBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, dilation=1, groups=1, bias=False, padding_mode='zeros',
                 deploy=False):
        super().__init__()
        self.deploy = deploy
        if deploy:
            self.fused_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                                        dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode)
        else:
            self.square_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(kernel_size, kernel_size),
                                         stride=stride, padding=padding, groups=groups, bias=bias, padding_mode=padding_mode)
            # common use case: k=3, p=1 or k=5, p=2
            if padding - kernel_size // 2 >= 0:
                self.crop = 0
                # align the sliding windows
                # padding=[rows, cols]
                hor_padding = [padding - kernel_size // 2, padding]
                ver_padding = [padding, padding - kernel_size // 2]
            else:
                #   A negative "padding" (padding - kernel_size//2 < 0, which is not a common use case) is cropping.
                #   Since nn.Conv2d does not support negative padding, we implement it manually
                self.crop = kernel_size // 2 - padding
                hor_padding = [0, padding]
                ver_padding = [padding, 0]

            self.hor_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, kernel_size),
                                      stride=stride, padding=hor_padding, dilation=dilation, groups=groups, bias=bias,
                                      padding_mode=padding_mode)
            self.ver_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(kernel_size, 1),
                                      stride=stride, padding=ver_padding, dilation=dilation, groups=groups, bias=bias,
                                      padding_mode=padding_mode)

    def _add_to_square_kernel(self, square_kernel, asym_kernel):
        '''叠加方形kernel和非方形kernel: 在方形kernel的对应位置上叠加
            kernel shape = [out_c, in_c, k, k]
            为了让非方形kernel(小核)的中心落在方形kernel(大核)的中心，放小核的起始位置应当等于两个中心索引之差
            实质上就是找这个起始位置对应大核中哪个索引
        '''
        asym_h = asym_kernel.size(2)
        asym_w = asym_kernel.size(3)
        square_h = square_kernel.size(2)
        square_w = square_kernel.size(3)
        square_kernel[:, :, square_h//2-asym_h//2:square_h//2-asym_h//2+asym_h,
        square_w//2-asym_w//2:square_w//2-asym_w//2+asym_w] += asym_kernel

    def get_equivalent_kernel_bias(self):
        # 合成kernel
        square_k = self.square_conv.weight.clone() # [out_c, in_c, k, k]
        hor_k = self.hor_conv.weight.clone()
        ver_k = self.ver_conv.weight.clone()
        self._add_to_square_kernel(square_k, hor_k)
        self._add_to_square_kernel(square_k, ver_k)

        # 合成bias
        square_b = self.square_conv.bias if self.square_conv.bias is not None else torch.zeros(square_k.size(0))
        hor_b = self.hor_conv.bias if self.hor_conv.bias is not None else torch.zeros(hor_k.size(0))
        ver_b = self.ver_conv.bias if self.ver_conv.bias is not None else torch.zeros(ver_k.size(0))
        square_b = square_b + hor_b + ver_b

        return square_k, square_b

    def switch_to_deploy(self):
        deploy_k, deploy_b = self.get_equivalent_kernel_bias()
        self.deploy = True
        if self.square_conv.bias is not None:
            bias = True
        else:
            bias = False
        self.fused_conv = nn.Conv2d(in_channels=self.square_conv.in_channels, out_channels=self.square_conv.out_channels,
                                    kernel_size=self.square_conv.kernel_size, stride=self.square_conv.stride,
                                    padding=self.square_conv.padding, dilation=self.square_conv.dilation, groups=self.square_conv.groups,
                                    bias=bias, padding_mode=self.square_conv.padding_mode)
        self.__delattr__('square_conv')
        self.__delattr__('hor_conv')
        self.__delattr__('ver_conv')
        self.fused_conv.weight.data = deploy_k
        if self.fused_conv.bias is not None:
            self.fused_conv.bias.data = deploy_b

    def forward(self, x):
        if self.deploy:
            return self.fused_conv(x)
        else:
            squared_output = self.square_conv(x)
            if self.crop > 0:
                ver_input = x[:, :, :, self.crop:-self.crop]
                hor_input = x[:, :, self.crop:-self.crop, :]
            else:
                ver_input = x
                hor_input = x
            ver_output = self.ver_conv(ver_input)
            hor_output = self.hor_conv(hor_input)
            out = squared_output + ver_output + hor_output
            return out

class FT_init(nn.Module):
    def __init__(self, msfa_size=4):
        super().__init__()
        self.msfa_size = msfa_size

    def forward(self, x):
        '''
        Args:
            x: [b,1,h,w], mosaic raw image
        Returns:
            out: [b,c,h,w], c=msfa_size**2 , full resolution spectral image
        '''
        # split along channel
        sub_images = F.pixel_unshuffle(x, downscale_factor=self.msfa_size) # sub_images: [b,1,h,w] -> [b,msfa_size**2,h//msfa_size, w//msfa_size]
        # 2D DFT extracts magnitude and phase spectrum of each sub_image
        mag_list = [] # length = msfa**2
        pha_list = [] # length = msfa**2
        for channel in range(sub_images.shape[1]):
            fft_result = torch.fft.fft2(sub_images[:, channel, :, :].unsqueeze(1)) # fft_result: [b,1,h//msfa_size, w//msfa_size]
            magnitude = torch.abs(fft_result) # magnitude: [b,1,h//msfa_size, w//msfa_size]
            phase = torch.angle(fft_result) # phase: [b,1,h//msfa_size, w//msfa_size]
            mag_list.append(magnitude)
            pha_list.append(phase)
        # FT-init
        init_list = []
        X_list = []
        for i in range(len(mag_list)):
            # x_fake = A_i + P_j, j=0,...,msfa_size**2
            init_list.clear() # be sure to clear init_list
            for j in range(len(pha_list)):
                fft_recon = mag_list[i] * torch.exp(1j * pha_list[j]) # reconstruct fft representation, fft_recon: [b,1,h//msfa_size, w//msfa_size]
                x_fake = torch.fft.ifft2(fft_recon).real # x_fake: [b,1,h//msfa_size, w//msfa_size]
                init_list.append(x_fake)
            # merge
            sub_images_new = torch.cat(init_list, dim=1) # fake_image: [b, msfa_size**2, h//msfa_size, w//msfa_size]
            X = F.pixel_shuffle(sub_images_new, upscale_factor=self.msfa_size) # X: [b, msfa_size**2, h//msfa_size, w//msfa_size] -> [b,1,h,w]
            X_list.append(X)

        # get full FT-init result
        full_X = torch.concat(X_list, dim=1) # full_X: [b,c,h,w]
        return full_X

class SplittingBIConv(nn.Module):
    def __init__(self, msfa_size):
        super().__init__()
        self.msfa_size = msfa_size
        self.conv = nn.Conv2d(in_channels=msfa_size**2, out_channels=msfa_size**2, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, sparse_raw):
        '''
        Args:
            sparse_raw: [b,c,h,w]
        Returns:
            out: [b,c,h,w]
        '''
        b, c, h, w = sparse_raw.shape
        intepolated = F.interpolate(sparse_raw, size=(h, w), mode='bicubic', align_corners=True)
        out = self.conv(intepolated)
        return out

class PxTConv(nn.Module):
    def __init__(self, msfa_size):
        super().__init__()
        self.msfa_size = msfa_size
        self.conv = nn.ConvTranspose2d(in_channels=msfa_size**2, out_channels=msfa_size**2, kernel_size=msfa_size, stride=msfa_size, bias=False)

    def forward(self, raw):
        '''
        Args:
            raw: [b,1,h,w]
        Returns:
            out: [b,c,h,w]
        '''
        b, c, h, w = raw.shape
        mosaic_reshape = F.pixel_unshuffle(raw, downscale_factor=self.msfa_size)  # [b,1,h,w] -> [b,c,h//msfa_size, w//msfa_size]
        out = self.conv(mosaic_reshape) # [b,c,h//msfa_size, w//msfa_size] -> [b,c,h,w]
        return out


class _Conv_LSA_Block_msfasize(nn.Module):
    def __init__(self, msfa_size):
        super(_Conv_LSA_Block_msfasize, self).__init__()
        self.ma = MACA(msfa_size, 64, k_size=5, reduction=16)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.cov_block = nn.Sequential(
            ACBlock(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            ACBlock(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            ACBlock(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
        )

    def forward(self, x):
        residual = x
        output = self.cov_block(x)
        output = self.ma(output)
        output += residual
        output = self.relu(output)
        return output


class FGLN(nn.Module):
    def __init__(self, msfa_size, SI_type='FPC', num_blocks=2):
        super(FGLN, self).__init__()
        self.msfa_size = msfa_size
        self.SI_type = SI_type

        # WB
        self.WB_Conv = nn.Conv2d(in_channels=msfa_size ** 2, out_channels=msfa_size ** 2, kernel_size=2 * msfa_size - 1,
                                 stride=1, padding=msfa_size - 1, bias=False, groups=msfa_size ** 2)
        WB_weight = self.get_WB_filter_msfa()
        c1, c2, h, w = self.WB_Conv.weight.data.size() # [out_channel, in_channel, kernel_size_h, kernel_size_w]
        WB_weight = WB_weight.view(1,1,h,w).repeat(c1,c2,1,1)
        self.WB_Conv.weight = nn.Parameter(WB_weight, requires_grad=False)
        # Spectral image initialization
        mcm_ksize = msfa_size + 1
        if msfa_size == 4:
            mcm_ksize = msfa_size + 1
        elif msfa_size == 5:
            mcm_ksize = msfa_size + 2

        self.SI = None
        if SI_type == 'MCM':
            self.SI = MosaicConvModule(msfa_size=msfa_size, kernel_size=mcm_ksize)
        elif SI_type == 'BI':
            # Splitting + BI + Conv
            self.SI = SplittingBIConv(msfa_size=msfa_size)
        elif SI_type == 'TConv':
            # PixelUnshuffle + TransposedConv
            self.SI = PxTConv(msfa_size=msfa_size)
        elif SI_type == 'Conv':
            # vanilla conv
            self.SI = nn.Conv2d(in_channels=1, out_channels=msfa_size**2, kernel_size=3, stride=1, padding=1, bias=False)
        elif SI_type == 'FT-init':
            self.SI = FT_init(msfa_size=msfa_size)
        elif SI_type == 'FPC':
            self.SI = FPC(msfa_size=msfa_size, in_channels=1, out_channels=16, kernel_size=5, num_experts=8, padding=2, bias=True)

        self.CA = MACA(msfa_size=msfa_size, channel=msfa_size ** 2, k_size=5, reduction=4)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.conv_input = nn.Conv2d(in_channels=msfa_size**2, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv_output = nn.Conv2d(in_channels=64, out_channels=msfa_size**2, kernel_size=3, stride=1, padding=1, bias=True)

        # blocks
        blocks = [_Conv_LSA_Block_msfasize(msfa_size=msfa_size) for _ in range(num_blocks)]
        self.blocks = nn.Sequential(*blocks)


    def get_WB_filter_msfa(self):
        """make a 2D weight bilinear kernel suitable for WB_Conv"""
        size = 2 * self.msfa_size - 1
        ligne = []
        colonne = []
        for i in range(size):
            if (i + 1) <= np.floor(math.sqrt(self.msfa_size ** 2)):
                ligne.append(i + 1)
                colonne.append(i + 1)
            else:
                ligne.append(ligne[i - 1] - 1.0)
                colonne.append(colonne[i - 1] - 1.0)
        BilinearFilter = np.zeros(size * size)
        for i in range(size):
            for j in range(size):
                BilinearFilter[(j + i * size)] = (ligne[i] * colonne[j] / (self.msfa_size ** 2))
        filter0 = np.reshape(BilinearFilter, (size, size))
        return torch.from_numpy(filter0).float()


    def forward(self, raw, sparse_raw):
        '''
        :param raw: [b,1,h,w]
        :param sparse_raw: [b,c,h,w]
        :return out: [b,c,h,w]
        '''

        # b, c, h_inp, w_inp = sparse_raw.shape
        # hb, wb = 16, 16
        # pad_h = (hb - h_inp % hb) % hb
        # pad_w = (wb - w_inp % wb) % wb
        # x_in_raw = F.pad(raw, [0, pad_w, 0, pad_h], mode='reflect')
        # x_in_sparse = F.pad(sparse_raw, [0, pad_w, 0, pad_h], mode='reflect')
        b, c, h, w = sparse_raw.shape
        x_in_sparse = sparse_raw
        x_in_raw = raw

        # WB Subbranch
        WB_x = self.WB_Conv(x_in_sparse) # WB_x: [b,c,h,w]

        # Initialization
        if self.SI_type == 'HardSplitting':
            h = x_in_sparse # h: [b,c,h,w]
        elif self.SI_type == 'BI':
            h = self.SI(x_in_sparse)
        else:
            h = self.SI(x_in_raw) # h: [b,1,h,w]->[b,c,h,w]

        h = self.CA(h)  # [b,c,h,w]
        h = self.lrelu(h) # [b,c,h,w]

        h = self.conv_input(h) # [c,64,h,w]

        # Main branch
        h = self.blocks(h) # h: [b,64,h,w], feature refinement
        h = self.conv_output(h) # h: [b,64,h,w] -> [b,c,h,w]

        # out
        out = torch.add(h, WB_x)
        return out # h: [b,c,h,w]

        #return out[:, :, :h_inp, :w_inp]  # 多的部分不要

if __name__ == '__main__':
    torch.manual_seed(0)

    model = FGLN(msfa_size=4, SI_type='FPC', num_blocks=2).cuda()
    raw = torch.randn(1,1,480,512).cuda()
    sparse_raw = torch.randn(1,16,480,512).cuda()
    input_data = (raw, sparse_raw)
    print('=================Before inference=====================')
    My_summary(model, input_data=input_data)
    out = model(raw, sparse_raw)

    for m in model.modules():
        if hasattr(m, 'switch_to_deploy'):
            m.switch_to_deploy()

    print('=================inference=====================')
    My_summary(model, input_data=input_data)
    deployout = model(raw, sparse_raw)

    print('difference between the outputs of the training-time and converted model is ')
    print(((deployout - out) ** 2).sum())