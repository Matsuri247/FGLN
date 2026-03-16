import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

class MACA(nn.Module):
    def __init__(self, msfa_size, channel, k_size=5, reduction=16):
        super().__init__()
        self.msfa_size = msfa_size
        self.channel = channel
        self.k_size = k_size

        self.avg_pool1 = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=k_size, padding=(k_size-1) // 2, bias=False)
        self.act1 = nn.Sigmoid()

        self.avg_pool2 = nn.AdaptiveAvgPool2d(1)
        self.fc2 = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        '''
        Args:
            x: [b,c,h,w]
        Returns:
            out: [b,c,h,w]
        '''
        batch, C, H, W = x.size()
        x_in = x.view(-1, 1, H, W) # x_in: [b, c, h, w]->[b*c, 1, h, w]
        px = F.pixel_unshuffle(x_in, downscale_factor=self.msfa_size) # px: [b*c, 1, h, w] -> [b*c, msfa_size**2, h//msfa_size, w//msfa_size]
        n, c, _, _ = px.size()
        py = self.avg_pool1(px).squeeze(-1).transpose(-2, -1) # py: [b*c, msfa_size**2, 1, 1] -> [b*c, msfa_size**2, 1] -> [b*c, 1, msfa_size**2]
        py = self.conv1(py).transpose(-2, -1).unsqueeze(-1) # py: [b*c, 1, msfa_size**2] -> [b*c, msfa_size**2, 1] -> [b*c, msfa_size**2, 1, 1]
        py = self.act1(py) # py: [b*c, msfa_size**2, 1, 1]
        py = py.expand_as(px) # py: [b*c, msfa_size**2, h//msfa_size, w//msfa_size] # TODO:核心操作
        py = F.pixel_shuffle(py, upscale_factor=self.msfa_size)  # py: [b*c, msfa_size**2, h//msfa_size, w//msfa_size] -> [b*c, 1, h, w] # TODO:核心操作
        pout = x_in * py # pout: [b*c, 1, h, w]
        pout = pout.view(batch, C, H, W) # pout: [b,c,h,w]

        y = self.avg_pool2(x).view(batch, C) # y: [b, c]
        y = self.fc2(y).view(batch, C, 1, 1) # y: [b, c, 1, 1]
        out = pout * y.expand_as(pout) # out: [b, c, h, w]
        return out



if __name__ == '__main__':
    b, c, h, w = 1,64,480,512
    inp=torch.randn(b,c,h,w)
    attn = MACA(msfa_size=4, channel=c, k_size=5, reduction=16)
    summary(attn, input_data=inp)