from __future__ import division
import torch
import torch.nn as nn
import logging
import numpy as np
import os
from ssim_torch import ssim
import hdf5storage
from torchinfo import summary
from fvcore.nn import FlopCountAnalysis
from thop import profile

def save_matv73(mat_name, var_name, var):
    hdf5storage.savemat(mat_name, {var_name: var}, format='7.3', store_python_metadata=True)

def reconstruct(method, model, input, msfa_size, crop_size, stride):
    '''
    :param method: str, name of model
    :param model:
    :param input: tuple (input_raw, input_sparse_raw), input_raw: torch.tensor, [b,1,h,w], input_sparse_raw: torch.tensor, [b,c,h,w]
    :param msfa_size: int, default 4
    :param crop_size: int
    :param stride: int
    :return: output: output with complete size, torch.tensor, [b,c,h,w]
    '''

    # same size as output
    out_size = torch.zeros_like(input[0]).repeat(1,msfa_size**2,1,1)
    abundance_matrix = torch.zeros_like(out_size).cuda()
    index_matrix = torch.zeros_like(out_size).cuda()

    h_idx = []
    for j in range(0, input[0].shape[2]-crop_size+1, stride):
        h_idx.append(j)
    h_idx.append(input[0].shape[2]-crop_size)

    w_idx = []
    for j in range(0, input[0].shape[3]-crop_size+1, stride):
        w_idx.append(j)
    w_idx.append(input[0].shape[3]-crop_size)

    # patch-wise reconstruction to avoid out of memory
    for h in h_idx:
        for w in w_idx:
            patch_input0 = input[0][:, :, h:h+crop_size, w:w+crop_size]
            patch_input1 = input[1][:, :, h:h+crop_size, w:w+crop_size]
            with torch.no_grad():
                # model output
                if method == 'MSFN_dual_PSA_each_PeriodGate_fullspa_shifted_cross_3branches':
                    _, patch_output = model(patch_input0, patch_input1)  # [b,c,h,w]
                    patch_output = torch.clamp(patch_output, 0, 1)

                # from patch to whole img
                abundance_matrix[:, :, h:h+crop_size, w:w+crop_size] = patch_output + abundance_matrix[:, :, h:h+crop_size, w:w+crop_size]
                # deal with overlapping
                index_matrix[:, :, h:h+crop_size, w:w+crop_size] = 1 + index_matrix[:, :, h:h+crop_size, w:w+crop_size]
    output = abundance_matrix / index_matrix
    return output

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def initialize_logger(file_dir):
    logger = logging.getLogger()
    fhandler = logging.FileHandler(filename=file_dir, mode='a')
    formatter = logging.Formatter('%(asctime)s - %(message)s', "%Y-%m-%d %H:%M:%S")
    fhandler.setFormatter(formatter)
    logger.addHandler(fhandler)
    logger.setLevel(logging.INFO)
    return logger

def save_checkpoint(model_path, epoch, model, optimizer):
    state = {
        #'epoch': epoch,
        #'iter': iteration,
        'state_dict': model.state_dict(),
        #'optimizer': optimizer.state_dict(),
    }
    #torch.save(state, os.path.join(model_path, 'net_%depoch_%diter.pth' % (epoch, iteration)))
    torch.save(state, os.path.join(model_path, 'net_%depoch.pth' % (epoch)))

class Loss_MRAE(nn.Module):
    def __init__(self):
        super(Loss_MRAE, self).__init__()

    def forward(self, outputs, label):
        assert outputs.shape == label.shape
        error = torch.abs(outputs - label) / (label + 1e-6)
        # mrae = torch.mean(error.view(-1))
        mrae = torch.mean(error.reshape(-1))
        return mrae

class Loss_RMSE(nn.Module):
    def __init__(self):
        super(Loss_RMSE, self).__init__()

    def forward(self, outputs, label):
        assert outputs.shape == label.shape
        error = outputs-label
        sqrt_error = torch.pow(error,2)
        # rmse = torch.sqrt(torch.mean(sqrt_error.view(-1))) # NTIRE2022
        rmse = torch.sqrt(torch.mean(sqrt_error.reshape(-1)))
        return rmse

class Loss_PSNR(nn.Module):
    def __init__(self):
        super(Loss_PSNR, self).__init__()

    def forward(self, im_true, im_fake, data_range=255):
        N = im_true.size()[0]
        C = im_true.size()[1]
        H = im_true.size()[2]
        W = im_true.size()[3]
        # Itrue = im_true.clamp(0., 1.).mul_(data_range).resize_(N, C * H * W)
        # Ifake = im_fake.clamp(0., 1.).mul_(data_range).resize_(N, C * H * W)
        Itrue = im_true.clamp(0., 1.).mul_(data_range).reshape(N, C * H * W)
        Ifake = im_fake.clamp(0., 1.).mul_(data_range).reshape(N, C * H * W)
        #mse = nn.MSELoss(reduce=False)
        mse = nn.MSELoss(reduction='none')
        err = mse(Itrue, Ifake).sum(dim=1, keepdim=True).div_(C * H * W)
        psnr = 10. * torch.log((data_range ** 2) / err) / np.log(10.) # torch.log和np.log都是以e为底；log换底公式
        return torch.mean(psnr)

class Loss_PSNR2(nn.Module):
    def __init__(self):
        super(Loss_PSNR2, self).__init__()

    def forward(self, im_true, im_fake, max_val=1.):
        mse = torch.mean((im_true - im_fake) ** 2, [0,2,3])
        psnr = 10*torch.mean(torch.log10(max_val / mse))
        return psnr

# degree
class SAM(nn.Module):
    def __init__(self):
        super(SAM, self).__init__()

    def forward(self, im_true, im_fake):
        # print(im_fake.size())
        im_true = im_true.squeeze(0)
        im_fake = im_fake.squeeze(0)
        # print(im_true.shape)
        C = im_true.size()[0]
        H = im_true.size()[1]
        W = im_true.size()[2]
        im_fake.reshape(C, H * W)
        im_true.reshape(C, H * W)
        esp = 1e-12
        Itrue = im_true.clone()  # .resize_(C, H*W)
        Ifake = im_fake.clone()  # .resize_(C, H*W)
        nom = torch.mul(Itrue, Ifake).sum(dim=0)  # .resize_(H*W)
        denominator = Itrue.norm(p=2, dim=0, keepdim=True).clamp(min=esp) * \
                      Ifake.norm(p=2, dim=0, keepdim=True).clamp(min=esp)
        denominator = denominator.squeeze()
        sam = torch.div(nom, denominator).acos()
        sam[sam != sam] = 0
        sam_sum = torch.sum(sam) / (H * W) / np.pi * 180
        return sam_sum

class SSIM(nn.Module):
    def __init__(self):
        super(SSIM, self).__init__()

    def forward(self, im_true, im_fake):
        return ssim(im_true, im_fake)

class ERGAS(nn.Module):
    def __init__(self, r=1):
        super(ERGAS, self).__init__()
        self.r = r

    def forward(self, GT, P):
        N = GT.size()[0]
        C = GT.size()[1]
        H = GT.size()[2]
        W = GT.size()[3]
        n_samples = H * W
        # RMSE
        aux = torch.sum(torch.sum((P - GT)**2, dim=-1), dim=-1) / n_samples
        rmse_per_band = torch.sqrt(aux)
        # ERGAS
        mean_y = torch.sum(torch.sum(GT, dim=-1), dim=-1) / n_samples
        ergas = 100 * self.r * torch.sqrt(torch.sum((rmse_per_band / mean_y) ** 2, dim=-1) / C)
        return torch.mean(ergas)

def pixel_shuffle_inv(np_array, scale_factor):
    """
    Implementation of inverted pixel shuffle using numpy

    Parameters:
    -----------
    tensor: input ndarray, shape is [H, W, C]
    scale_factor: scale factor to down-sample tensor

    Returns:
    --------
    tensor: tensor after pixel shuffle, shape is [H/s, W/s, (s*s)*C],
        where s refers to scale factor
    """
    height, width, ch = np_array.shape
    if height % scale_factor != 0 or width % scale_factor != 0:
        raise ValueError('height and widht of tensor must be divisible by '
                         'scale_factor.')

    new_ch = ch * (scale_factor * scale_factor)
    new_height = height // scale_factor
    new_width = width // scale_factor
    # [H,W,C] -> [H/s, s, W/s, s, C] -> [H/s, W/s, s, s, C] -> [H/s, W/s, (s*s)*C]
    np_array = np_array.reshape(new_height, scale_factor, new_width, scale_factor, ch)
    np_array = np.transpose(np_array, (0, 2, 1, 3, 4))
    np_array = np_array.reshape(new_height, new_width, new_ch)

    return np_array

class self_evaluation_index(nn.Module):
    def __init__(self, msfa_size):
        super().__init__()
        self.msfa_size = msfa_size

    @staticmethod
    def pixel_shuffle_inv(tensor, scale_factor): # 没必要，因为torch.nn.functional有pixel_unshuffle
        '''
        Implementation of inverted pixel shuffle

        Args:
            tensor: torch.Tensor, [b,c,h,w]
            scale_factor: int, scale factor to down-sample tensor
        Returns:
            shuffled_tensor: shuffled tensor, [b, (s*s)*c, h//s, w//s], s is scale_factor

        '''
        num, ch, height, width = tensor.shape
        if height % scale_factor != 0 or width % scale_factor != 0:
            raise ValueError('height and widht of tensor must be divisible by '
                             'scale_factor.')

        new_ch = ch * (scale_factor * scale_factor)
        new_height = height // scale_factor
        new_width = width // scale_factor

        shuffled_tensor = tensor.reshape(
            num, ch, new_height, scale_factor, new_width, scale_factor)
        # new axis: [num, ch, scale_factor, scale_factor, new_height, new_width]
        shuffled_tensor = shuffled_tensor.permute(0, 1, 3, 5, 2, 4)
        shuffled_tensor = shuffled_tensor.reshape(num, new_ch, new_height, new_width)
        return shuffled_tensor

    def forward(self, reconstructed_cube):
        '''
        Calculating unsupervised metrics Self-Evaluation-Index (SEI)

        Args:
            reconstructed_cube: torch.Tensor, [b,c,h,w]
        Returns:
            SEI_cube: SEI metrics for single reconstructed cube
        '''
        num_band = self.msfa_size ** 2
        var_cube = torch.ones(1, num_band)
        for i in range(num_band):
            singleband = reconstructed_cube[:, i, :, :].unsqueeze(1) # singleband: [b,1,h,w]
            singleband_subimgs = self.pixel_shuffle_inv(singleband, scale_factor=self.msfa_size) # singleband_subimgs: [b,1,h,w]->[b,s*s,h//s,w//s]
            singleband_subimgs_avg = torch.mean(torch.mean(singleband_subimgs,-1), -1) # singleband_subimgs_avg: [b,s*s,h//s,w//s]->[b,s*s]
            singleband_subimgs_var = singleband_subimgs_avg.var(dim=-1) # singleband_subimgs_var: [b,s*s]->[b]
            var_cube[0, i] = singleband_subimgs_var # Recording var of each band
        SEI_cube = torch.mean(var_cube, dim=1)

        return SEI_cube

class L1_Charbonnier_mean_loss(nn.Module):
    """L1 Charbonnierloss."""

    def __init__(self):
        super(L1_Charbonnier_mean_loss, self).__init__()
        self.eps = 1e-6

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss

def poly_lr_scheduler(optimizer, init_lr, iteraion, lr_decay_iter=1,
                      max_iter=100, power=0.9):
    """Polynomial decay of learning rate
        :param init_lr is base learning rate
        :param iteraion is a current iteration
        :param lr_decay_iter how frequently decay occurs, default is 1
        :param max_iter is number of maximum iterations
        :param power is a polymomial power

    """
    if iteraion % lr_decay_iter or iteraion > max_iter:
        return optimizer

    lr = init_lr*(1 - iteraion/max_iter)**power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr

def time2file_name(time):
    year = time[0:4]
    month = time[5:7]
    day = time[8:10]
    hour = time[11:13]
    minute = time[14:16]
    second = time[17:19]
    time_filename = year + '_' + month + '_' + day + '_' + hour + '_' + minute + '_' + second
    return time_filename


# for MCAN
def input_matrix_wpn(inH, inW, msfa_size):

    h_offset_coord = torch.zeros(inH, inW, 1)
    w_offset_coord = torch.zeros(inH, inW, 1)
    for i in range(0,msfa_size):
        h_offset_coord[i::msfa_size, :, 0] = (i+1)/msfa_size
        w_offset_coord[:, i::msfa_size, 0] = (i+1)/msfa_size

    pos_mat = torch.cat((h_offset_coord, w_offset_coord), 2)
    pos_mat = pos_mat.contiguous().view(1, -1, 2)
    return pos_mat

class unsup_ssim_loss_hyper(nn.Module):
    '''unsupervised ssim loss
    This loss is based on spectral similarity prior,
    i.e., nearby bands of a full MSI should show very similar structual similarity
    '''
    def __init__(self, alpha=1):
        super(unsup_ssim_loss_hyper, self).__init__()
        self.ssim = SSIM()
        self.alpha = alpha

    def forward(self, hyper):
        '''
        Args:
            hyper: HSI/MSI image, torch.Tensor, [b,c,h,w]
        Returns:
            loss_ssim_hyper
        '''
        channels = hyper.shape[1]
        ssim_list = []
        for i in range(channels-1):
            ssim_1 = self.ssim(hyper[:, i:i+1, :, :], hyper[:, i+1:i+2, :, :]) # 相邻波段两两比较
            ssim_list.append(ssim_1)
        ssim_tensor = torch.Tensor(ssim_list)
        ssim_all = torch.mean(ssim_tensor)
        loss_ssim_hyper = self.alpha * (1 - ssim_all)
        return loss_ssim_hyper

# class reconstruction_loss_SSIM(nn.Module):
#     """reconstruction loss of raw_msfa (Mosaic loss)"""
#     def __init__(self, msfa_size):
#         super(reconstruction_loss_SSIM, self).__init__()
#         self.msfa_size = msfa_size
#         self.SSIM_loss = SSIM()
#
#     @staticmethod
#     def get_msfa(img_tensor, msfa_size):
#         mask = torch.zeros_like(img_tensor)
#         for i in range(0, msfa_size):
#             for j in range(0, msfa_size):
#                 mask[:, i * msfa_size + j, i::msfa_size, j::msfa_size] = 1
#         return torch.sum(mask.mul(img_tensor), 1, keepdim=True)
#
#     def forward(self, X, Y):
#         loss = 1 - self.SSIM_loss(self.get_msfa(X, self.msfa_size), self.get_msfa(Y, self.msfa_size))
#         return loss

# deprecated
class loss_SSIM(nn.Module):
    def __init__(self):
        super(loss_SSIM, self).__init__()
        self.ssim = SSIM()

    def forward(self, im_true, im_fake):
        ssim_loss = 1 - self.ssim(im_true, im_fake)
        return ssim_loss

# unsupervised spectral l1 sparsity loss
class spectral_l1_sparsity_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        '''
        Args:
            x: [b,c,h,w]
        Returns:
            loss: [1,]
        '''
        x = x.permute(0, 2, 3, 1) # x:[b,c,h,w] -> [b,h,w,c]
        fft_result = torch.fft.fft(x, dim=-1)
        magnitude = torch.abs(fft_result)
        # 鼓励整个光谱频谱趋于稀疏，减少高频跳变
        loss = magnitude.mean()
        return loss

# unsupervised high-frequency penalty loss
class spectral_highfreq_penalty_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        ''' 为每个光谱频率分配一个惩罚权重weights(相当于soft mask)，高频权重大，低频权重小
        Args:
            x: [b,c,h,w]
        Returns:
            loss: [1,]
        '''
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)  # x:[b,c,h,w] -> [b,h,w,c]
        # spectral DFT
        fft_result = torch.fft.fft(x, dim=-1)
        magnitude = torch.abs(fft_result)
        # weight of spectral frequency index
        freqs = torch.fft.fftfreq(C, d=1/C).abs().to(x.device) # 构造频率坐标，使得结果中的最大频率值是C/2， abs()代表只关心频率的大小，不关心正负（反正数值大即高频率，就是要压制的）
        weights = freqs / freqs.max() # 根据频率坐标分配权重，并归一化到[0,1]
        # apply weights
        weighted_mag = magnitude * weights.view(1, 1, 1, -1)
        loss = weighted_mag.mean()
        return loss


class reconstruction_loss(nn.Module):
    """reconstruction loss of raw_msfa (Mosaic loss)"""
    def __init__(self, msfa_size):
        super(reconstruction_loss, self).__init__()
        #self.wt = 1
        self.msfa_size = msfa_size
        # self.mse_loss = nn.MSELoss(reduce=True, size_average=False)
        self.mse_loss = L1_Charbonnier_mean_loss()

    @staticmethod
    def get_msfa(img_tensor, msfa_size):
        mask = torch.zeros_like(img_tensor)
        for i in range(0, msfa_size):
            for j in range(0, msfa_size):
                mask[:, i * msfa_size + j, i::msfa_size, j::msfa_size] = 1
        return torch.sum(mask.mul(img_tensor), 1, keepdim=True)

    def forward(self, X, Y, is_simulate=True):
        # is simulate experiment?
        if is_simulate:
            # if so, X is reconstructed, Y is gt.
            loss = self.mse_loss(self.get_msfa(X, self.msfa_size), self.get_msfa(Y, self.msfa_size))
        else:
            # if it is real experiment, X is reconstructed, Y is mosaic
            loss = self.mse_loss(self.get_msfa(X, self.msfa_size), Y)
        return loss

def mask_input(GT_image, msfa_size):
    """
    generate sparse raw image from GT MSI
    :params GT_image: ndarray,[h,w,c]
    :params msfa_size: int (5 for 25 bands, 4 for 16 bands)
    :return input_image: ndarray,[h,w,c]
    """
    mask = np.zeros((GT_image.shape[0], GT_image.shape[1], msfa_size ** 2), dtype=np.float32)
    for i in range(0, msfa_size):
        for j in range(0, msfa_size):
            mask[i::msfa_size, j::msfa_size, i*msfa_size+j] = 1
    input_image = mask * GT_image
    return input_image

def mask_input_Real(GT_image, msfa_size):
    """
        generate sparse raw image from GT MSI
        :params GT_image: ndarray,[h,w,c]
        :params msfa_size: int (5 for 25 bands, 4 for 16 bands)
        :return
            input_image: ndarray,[h,w,c]
            mask_arraged: ndarray, [h,w,c]
    """
    mask = np.zeros((GT_image.shape[0], GT_image.shape[1], msfa_size ** 2), dtype=np.float32)
    mask_arranged = np.zeros((GT_image.shape[0], GT_image.shape[1], msfa_size ** 2), dtype=np.float32)
    for i in range(0, msfa_size):
        for j in range(0, msfa_size):
            mask[i::msfa_size, j::msfa_size, i * msfa_size + j] = 1

    mask_arranged[:, :, 0] = mask[:, :, 10]
    mask_arranged[:, :, 1] = mask[:, :, 6]
    mask_arranged[:, :, 2] = mask[:, :, 2]
    mask_arranged[:, :, 3] = mask[:, :, 14]
    mask_arranged[:, :, 4] = mask[:, :, 9]
    mask_arranged[:, :, 5] = mask[:, :, 5]
    mask_arranged[:, :, 6] = mask[:, :, 1]
    mask_arranged[:, :, 7] = mask[:, :, 13]
    mask_arranged[:, :, 8] = mask[:, :, 8]
    mask_arranged[:, :, 9] = mask[:, :, 4]
    mask_arranged[:, :, 10] = mask[:, :, 0]
    mask_arranged[:, :, 11] = mask[:, :, 12]
    mask_arranged[:, :, 12] = mask[:, :, 11]
    mask_arranged[:, :, 13] = mask[:, :, 7]
    mask_arranged[:, :, 14] = mask[:, :, 3]
    mask_arranged[:, :, 15] = mask[:, :, 15]

    input_image = mask_arranged * GT_image
    return input_image, mask_arranged

# calculate Model Parameters & FLOPs
def My_summary(model, input_data):
    # tool1: torchinfo.summary
    print('Torchinfo.summary')
    summary(model, input_data=input_data)
    print('\n')

    # tool2: thop
    print('thop')
    MACs, params = profile(model, inputs=(input_data))
    FLOPs = MACs * 2
    print(f'FLOPs(G): {FLOPs/(1024*1024*1024)}.')
    print(f'Params(M): {params/(1024*1024)}.')
    print('\n')

    # # tool3: nn.fvcore.FlopCountAnalysis
    # print('nn.fvcore.FlopCountAnalysis')
    # FLOPs = FlopCountAnalysis(model, input_data)
    # print(f'FLOPs(G): {FLOPs.total()/(1024*1024*1024)}.')
    # print('\n')
