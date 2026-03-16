import torch
import torch.nn as nn

# version adaptation for PyTorch > 1.7.1
IS_HIGH_VERSION = tuple(map(int, torch.__version__.split('+')[0].split('.'))) > (1, 7, 1)
if IS_HIGH_VERSION:
    import torch.fft

class FocalFrequencyLoss(nn.Module):

    def __init__(self, loss_weight=1.0, alpha=1.0, patch_factor=1, ave_spectrum=False, log_matrix=False, batch_matrix=False):
        super(FocalFrequencyLoss, self).__init__()
        self.loss_weight = loss_weight
        self.alpha = alpha
        self.patch_factor = patch_factor
        self.ave_spectrum = ave_spectrum
        self.log_matrix = log_matrix
        self.batch_matrix = batch_matrix

    def tensor2freq(self, x):
        '''
        Args:
            x: [b,c,h,w]

        Returns:
            freq: [b,1,c,h,w,2], 增加了patch维度和fft实部虚部的维度, 2的维度是先real后imag
        '''
        # crop image patches
        patch_factor = self.patch_factor
        _, _, h, w = x.shape
        assert h % patch_factor == 0 and w % patch_factor == 0, (
            'Patch factor should be divisible by image height and width')
        patch_list = []
        patch_h = h // patch_factor
        patch_w = w // patch_factor
        for i in range(patch_factor):
            for j in range(patch_factor):
                patch_list.append(x[:, :, i * patch_h:(i + 1) * patch_h, j * patch_w:(j + 1) * patch_w])

        # stack to patch tensor
        y = torch.stack(patch_list, 1) # y: [b, 1, c, h, w]

        # perform 2D DFT (real-to-complex, orthonormalization)
        if IS_HIGH_VERSION:
            freq = torch.fft.fft2(y, norm='ortho')
            freq = torch.stack([freq.real, freq.imag], -1) # 把实部和虚部分别揪出来
            # freq: [b,1,c,h,w,2]
        else:
            freq = torch.rfft(y, 2, onesided=False, normalized=True)
        return freq

    def loss_formulation(self, recon_freq, real_freq, need_matrix=False):
        # frequency distance using (squared) Euclidean distance
        tmp = (recon_freq - real_freq) ** 2
        freq_distance = tmp[..., 0] + tmp[..., 1]

        # spectrum weight matrix
        if need_matrix:
            # if the matrix is calculated online: continuous, dynamic, based on current Euclidean distance
            matrix_tmp = (recon_freq - real_freq) ** 2
            matrix_tmp = torch.sqrt(matrix_tmp[..., 0] + matrix_tmp[..., 1]) ** self.alpha

            # whether to adjust the spectrum weight matrix by logarithm
            if self.log_matrix:
                matrix_tmp = torch.log(matrix_tmp + 1.0)

            # whether to calculate the spectrum weight matrix using batch-based statistics
            if self.batch_matrix:
                matrix_tmp = matrix_tmp / matrix_tmp.max()
            else:
                matrix_tmp = matrix_tmp / matrix_tmp.max(-1).values.max(-1).values[:, :, :, None, None]

            matrix_tmp[torch.isnan(matrix_tmp)] = 0.0
            matrix_tmp = torch.clamp(matrix_tmp, min=0.0, max=1.0)
            weight_matrix = matrix_tmp.clone().detach()

            assert weight_matrix.min().item() >= 0 and weight_matrix.max().item() <= 1, (
                    'The values of spectrum weight matrix should be in the range [0, 1], '
                    'but got Min: %.10f Max: %.10f' % (weight_matrix.min().item(), weight_matrix.max().item()))
        else:
            weight_matrix = 1

        # dynamic spectrum weighting (Hadamard product)
        loss = weight_matrix * freq_distance # [b, patch, c, h, w]
        return torch.mean(loss)

    @staticmethod
    def get_msfa(img_tensor, msfa_size=4):
        mask = torch.zeros_like(img_tensor)
        for i in range(0, msfa_size):
            for j in range(0, msfa_size):
                mask[:, i * msfa_size + j, i::msfa_size, j::msfa_size] = 1
        return torch.sum(mask.mul(img_tensor), 1, keepdim=True)

    def forward(self, pred, target, need_mask=False, need_matrix=False, is_simulate=False, **kwargs):
        """Forward function to calculate focal frequency loss.

        Args:
            pred (torch.Tensor): of shape (N, C, H, W). Predicted tensor.
            target (torch.Tensor): of shape (N, C, H, W). Target tensor.
            need_matrix (torch.Tensor, optional): need Element-wise spectrum weight matrix?
                Default: False (If set to False: do not calculate).
        """

        # need mosaic sampling?
        if need_mask:
            pred = self.get_msfa(pred) # [b,1,h,w]
            if is_simulate:
                target = self.get_msfa(target) # [b,1,h,w]

        pred_freq = self.tensor2freq(pred) # 把tensor转换到频率域
        target_freq = self.tensor2freq(target)

        # whether to use minibatch average spectrum
        if self.ave_spectrum:
            pred_freq = torch.mean(pred_freq, 0, keepdim=True)
            target_freq = torch.mean(target_freq, 0, keepdim=True)

        # calculate focal frequency loss
        return self.loss_formulation(pred_freq, target_freq, need_matrix) * self.loss_weight

if __name__ == '__main__':
    pred = torch.randn(1,16,100,100)
    target = torch.randn(1,16,100,100)
    loss = FocalFrequencyLoss()
    l = loss(pred, target, need_mask=False, is_simulate=True)
    print(l)