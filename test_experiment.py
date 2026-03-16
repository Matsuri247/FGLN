import torch
import numpy as np
import os
import glob
import argparse
import torch.backends.cudnn as cudnn
from architecture import *
from utils import *
from torch.utils.data import DataLoader
from CAVE_dataset_2 import CAVEDataset
from ARAD_dataset import TestARADDataset
from Real_dataset import RealDataset

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

parser = argparse.ArgumentParser(description="Multispectral Image Demosaicing Toolbox")
parser.add_argument('--method', type=str, default='My')
parser.add_argument('--msfa_size', type=int, default=4)
parser.add_argument('--dataset', type=str, default='CAVE', help='CAVE,ARAD')
parser.add_argument('--pretrained_model_path', type=str, default=None)
parser.add_argument('--test_dir', type=str, default='./dataset/CAVE/test/')
parser.add_argument('--outf', type=str, default='./test_exp/My/', help='output files')
parser.add_argument('--cache', type=str, default='./training_cache/', help='training data cache')
parser.add_argument('--gpu_id', type=str, default='0')
opt = parser.parse_args()
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

if not os.path.exists(opt.outf):
    os.makedirs(opt.outf)

# load dataset
print(f"Start testing: {opt.method}")
print(f"\nLoading dataset: {opt.dataset}...")

# loading cache
cache_path = os.path.join(opt.cache, opt.dataset)
if not os.path.exists(cache_path):
    os.mkdir(cache_path)
opt.cache_path = cache_path

# select dataset
if opt.dataset == 'CAVE':
    test_data = CAVEDataset(opt, type='test')
elif opt.dataset == 'ARAD':
    test_data = TestARADDataset(data_root=opt.test_dir, msfa_size=opt.msfa_size)
elif opt.dataset == 'Real':
    test_data = RealDataset(opt, type='test')
test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=False, num_workers=10, pin_memory=True)

# gt list (get save file_name)
file_type = '*.mat' # '*.mat' or '*.tif'
file_list = glob.glob(os.path.join(opt.test_dir, file_type))
file_list.sort()
filename_list = [os.path.split(file)[1] for file in file_list]

# criterion
criterion_psnr = Loss_PSNR()
criterion_ssim = SSIM()
criterion_sam = SAM()
criterion_ergas = ERGAS()
criterion_SEI = self_evaluation_index(msfa_size=opt.msfa_size)

if torch.cuda.is_available():
    criterion_psnr.cuda()
    criterion_ssim.cuda()
    criterion_sam.cuda()
    criterion_ergas.cuda()

def test(opt, test_loader, method, model):
    model.eval()
    metric_psnr = AverageMeter()
    metric_ssim = AverageMeter()
    metric_sam = AverageMeter()
    metric_ergas = AverageMeter()
    metric_SEI = AverageMeter()

    # 如果有使用ACBlock，那么就在推理之前，将ACBlock中的三个卷积核进行合并
    if method == 'FGLN':
        for m in model.modules():
            if hasattr(m, 'switch_to_deploy'):
                m.switch_to_deploy()

    with torch.no_grad():
        # 模拟实验有GT，计算全参考质量指标
        #if opt.dataset != 'Real':
        if 'real' not in opt.dataset.lower():
            for i, (raw, sparse_raw, msi) in enumerate(test_loader):
                input_raw, input_sparse_raw, gt = raw, sparse_raw, msi
                _, _, H, W = raw.size()
                input_raw = input_raw.cuda()
                input_sparse_raw = input_sparse_raw.cuda()
                gt = gt.cuda()

                output = model(input_raw, input_sparse_raw)

                # calculate metrics
                psnr = criterion_psnr(output, gt)
                ssim = criterion_ssim(output, gt)
                sam = criterion_sam(output, gt)
                ergas = criterion_ergas(output, gt)
                SEI = criterion_SEI(output)  # 无论模拟实验还是真实实验，都需要计算SEI

                # record metrics
                metric_psnr.update(psnr.data)
                metric_ssim.update(ssim.data)
                metric_sam.update(sam.data)
                metric_ergas.update(ergas.data)
                metric_SEI.update(SEI.data)

                # save to .mat
                result = output.cpu().numpy() * 1.0  # ndarray, [b,c,h,w]
                result = np.transpose(np.squeeze(result), [1, 2, 0])  # ndarray,[b,c,h,w] -> [h,w,c]
                result = np.minimum(result, 1.0)
                result = np.maximum(result, 0)
                result_name = filename_list[i]
                result_dir = os.path.join(opt.outf, result_name)
                save_matv73(result_dir, 'cube', result)

        else:  # 真实实验没有GT，不计算全参考质量指标，计为0
            for i, (raw, sparse_raw) in enumerate(test_loader):
                input_raw, input_sparse_raw = raw, sparse_raw
                _, _, H, W = raw.size()
                input_raw = input_raw.cuda()
                input_sparse_raw = input_sparse_raw.cuda()

                output = model(input_raw, input_sparse_raw)

                # calculate metrics
                psnr = 0
                ssim = 0
                sam = 0
                ergas = 0
                SEI = criterion_SEI(output)  # 无论模拟实验还是真实实验，都需要计算SEI

                # record metrics
                metric_psnr.update(psnr)
                metric_ssim.update(ssim)
                metric_sam.update(sam)
                metric_ergas.update(ergas)
                metric_SEI.update(SEI.data)

                # save to .mat
                result = output.cpu().numpy() * 1.0  # ndarray, [b,c,h,w]
                result = np.transpose(np.squeeze(result), [1, 2, 0])  # ndarray,[b,c,h,w] -> [h,w,c]
                result = np.minimum(result, 1.0)
                result = np.maximum(result, 0)
                result_name = filename_list[i]
                result_dir = os.path.join(opt.outf, result_name)
                save_matv73(result_dir, 'cube', result)

        return metric_psnr.avg, metric_ssim.avg, metric_sam.avg, metric_ergas.avg, metric_SEI.avg

if __name__ == '__main__':
    cudnn.benchmark = True
    pretrained_model_path = opt.pretrained_model_path
    method = opt.method
    msfa_size = opt.msfa_size
    model = model_generator(method, msfa_size, pretrained_model_path).cuda()
    psnr, ssim, sam, ergas, SEI = test(opt, test_loader, method, model)
    print(f'method:{method}, PSNR:{psnr}, SSIM:{ssim}, SAM:{sam}, ERGAS:{ergas}, SEI:{SEI}.')