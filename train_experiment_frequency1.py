import torch
import torch.nn as nn
import argparse
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import os
import random
from tqdm import tqdm
from utils import *
from architecture import *
from CAVE_dataset_2 import CAVEDataset
from ARAD_dataset import TrainARADDataset, TestARADDataset
import datetime
from focal_frequency_loss import FocalFrequencyLoss

#torch.autograd.set_detect_anomaly(True)

parser = argparse.ArgumentParser(description="Multispectral Image Demosaicing Toolbox")
parser.add_argument('--method', type=str, default='My')
parser.add_argument('--msfa_size', type=int, default=4)
parser.add_argument('--dataset', type=str, default='CAVE', help='CAVE,ARAD')
parser.add_argument('--pretrained_model_path', type=str, default=None)
parser.add_argument('--batch_size', type=int, default=20, help="batch size")
parser.add_argument('--end_epoch', type=int, default=7500, help="number of iterations")
parser.add_argument('--init_lr', type=float, default=4e-4, help="initial learning rate")
parser.add_argument('--lr_step', type=int, default=1000, help="adjust learning rate")
parser.add_argument('--outf', type=str, default='./train_exp/My/', help='output files')
parser.add_argument('--train_dir', type=str, default='/home/lab206/mycode/demosaicing_MPEFormer/dataset/CAVE/train/')
parser.add_argument('--test_dir', type=str, default='/home/lab206/mycode/demosaicing_MPEFormer/dataset/CAVE/test/')
parser.add_argument('--cache', type=str, default='./training_cache/', help='training data cache')
parser.add_argument('--is_freq_loss', action='store_false') # TODO: 是否启用频域loss， 不写就是True，写了就是False
parser.add_argument('--gpu_id', type=str, default='0')
opt = parser.parse_args()

# 获取环境变量
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
# load dataset
print(f"Start training: {opt.method}")
print(f"\nLoading dataset: {opt.dataset}...")

# loading cache
cache_path = os.path.join(opt.cache, opt.dataset)
if not os.path.exists(cache_path):
    os.mkdir(cache_path)
opt.cache_path = cache_path

# select dataset
if opt.dataset == 'CAVE':
    patch_size = 64
    stride = 32
    train_data = CAVEDataset(opt, type='train', patch_size=patch_size, stride=stride, augment=False)
    test_data = CAVEDataset(opt, type='test')
elif opt.dataset == 'ARAD':
    patch_size = 160
    train_data = TrainARADDataset(data_root=opt.train_dir, msfa_size=opt.msfa_size, patch_size=patch_size, augment=False)
    test_data = TestARADDataset(data_root=opt.test_dir, msfa_size=opt.msfa_size)
train_loader = DataLoader(dataset=train_data, batch_size=opt.batch_size, shuffle=True, num_workers=10, pin_memory=True)
test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=False, num_workers=10, pin_memory=True)

#per_epoch_iteration = len(train_data) // opt.batch_size
print(f'Training set samples: {len(train_data)}')
print(f'Test set samples: {len(test_data)}')
total_iteration = opt.end_epoch * len(train_loader)
print(f'Total iteration: {total_iteration}')

# criterion
criterion_psnr = Loss_PSNR()
criterion_ssim = SSIM()
criterion_sam = SAM()
criterion_ergas = ERGAS()

# select corresponding loss function
criterion = (L1_Charbonnier_mean_loss(), reconstruction_loss(msfa_size=opt.msfa_size), FocalFrequencyLoss())

# model
pretrained_model_path = opt.pretrained_model_path
model = model_generator(opt.method, opt.msfa_size, pretrained_model_path).cuda()
print('Parameters number is ', sum(param.numel() for param in model.parameters()))

# output path
date_time = str(datetime.datetime.now())
date_time = time2file_name(date_time)
opt.outf = opt.outf + date_time
if not os.path.exists(opt.outf):
    os.makedirs(opt.outf)

if torch.cuda.is_available():
    model.cuda()
    criterion_psnr.cuda()
    criterion_ssim.cuda()
    criterion_sam.cuda()
    criterion_ergas.cuda()
    criterion[0].cuda() # cube_loss
    criterion[1].cuda() # mosaic_loss
    criterion[2].cuda() # focal frequency loss

if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

# ADAM optimizer
optimizer = optim.Adam(model.parameters(), lr=opt.init_lr, betas=(0.9, 0.999))
########################################## TODO: check scheduler before training!!! ######################################################
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_iteration, eta_min=1e-6)
#scheduler = None

# logging
log_dir = os.path.join(opt.outf, 'train.log')
logger = initialize_logger(log_dir)

def random_crop_4DTensor(a, crop_size):
    N, C, hei, wid=a.size()
    Height = random.randint(0, hei - crop_size)
    Width = random.randint(0, wid - crop_size)
    return a[:, :, Height:(Height + crop_size), Width:(Width + crop_size)]

def transform_opt(HR_4x, msfa_size, tran_type='random'):
    if tran_type == 'random':
        tran_type = random.choice([0, 1, 3, 5])
        #tran_type = random.choice([0, 1, 5])
    elif tran_type == 'rotation': # 旋转变换
        tran_type = 0
    elif tran_type == 'flip': # 翻转变换
        tran_type = 1
    elif tran_type == 'resize': # 尺寸变化
        tran_type = 3
    elif tran_type == 'patternshift': # 平移变换，最重要
        tran_type = 5
    else:
        raise Exception("wrong tran type")

    if tran_type == 0:
        ## rotation transf
        rn = random.randint(1, 3) # 1,2,3
        new_lable = torch.rot90(HR_4x, rn, [2, 3]) # rot90是逆时针旋转

    if tran_type == 1:
        if np.random.uniform() < 0.5:
            new_lable = torch.flip(HR_4x, [2]) # H, 上下翻转
        else:
            new_lable = torch.flip(HR_4x, [3]) # W, 左右翻转

    scale_lib = [0.2, 0.25, 0.5, 2, 3, 4]

    if tran_type == 3:
        ## resize transf
        while 1:
            scale_num = random.randint(0, len(scale_lib)-1)
            new_lable = torch.nn.functional.interpolate(HR_4x, scale_factor=scale_lib[scale_num], mode='bicubic')
            N, C, H, W = new_lable.size()
            if H >= (patch_size-opt.msfa_size)*scale_lib[0] and H <= patch_size*scale_lib[-1]:
                break
        new_lable = random_crop_4DTensor(new_lable, (H // opt.msfa_size) * opt.msfa_size) # 把resize后的尺寸裁剪成msfa_size的倍数

    if tran_type == 5:
        ## new shift have just (msfa_size-1)*(msfa_size-1) transformations
        while 1:
            i = random.randint(0, msfa_size-1)
            j = random.randint(0, msfa_size-1)
            if i != 0 or j != 0:
                break
        new_lable = torch.roll(HR_4x, (-i, -j), (2, 3)) # 图像向左平移j个像素，向上平移i个像素，并且是周期性平移，左边/上边移出边界的像素会从右边/下边“卷”回来
        N, C, H, W = new_lable.size()
        new_lable = new_lable[:, :, 0: (H - opt.msfa_size), 0: (W - opt.msfa_size)] # 周期移动的特性满足MSFA采样周期的结构，然而，还需小心边界（因为torch.roll“卷回来”的部分现在并不符合图像原有结构，会造成污染，故必须裁掉）

    return new_lable

def get_sparsecube_raw(img_tensor, msfa_size):
    '''
    :param img_tensor: [b,c,h,w]
    :param msfa_size: int
    :return:
        mask.mul(img_tensor): sparse raw image,[b,c,h,w]
        torch.sum(mask.mul(img_tensor), 1).unsqueeze(1): raw image, [b,1,h,w]
    '''
    mask = torch.zeros_like(img_tensor)
    for i in range(0, msfa_size):
        for j in range(0, msfa_size):
            mask[:, i * msfa_size + j, i::msfa_size, j::msfa_size] = 1

    return mask.mul(img_tensor), torch.sum(mask.mul(img_tensor), 1).unsqueeze(1)

def adjust_learning_rate(optimizer, epoch, step=1000):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    lr = opt.init_lr * (0.5 ** (epoch // step))
    return lr

def train(opt, train_loader, optimizer, scheduler, method, model, criterion, epoch, num_epochs):
    # adjust learning rate
    if scheduler != None:
        lr = optimizer.param_groups[0]['lr']
    else:
        lr = adjust_learning_rate(optimizer, epoch - 1, opt.lr_step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    train_bar = tqdm(train_loader)
    model.train()
    batchSize_count = 0

    for batch in train_bar:
        input_raw, input_sparse_raw, gt = Variable(batch[0]), Variable(batch[1]), Variable(batch[2], requires_grad=False)
        N, C, H, W = batch[0].size()
        batchSize_count += N
        input_raw = input_raw.cuda()
        input_sparse_raw = input_sparse_raw.cuda()
        gt = gt.cuda()

        firstcube = model(input_raw, input_sparse_raw)

        # mosaic loss
        mosaic_loss = criterion[1](firstcube, gt)
        loss = mosaic_loss.clone()

        if opt.is_freq_loss:
            # frequency: mosaic loss
            mosaic_loss_freq = criterion[2](firstcube, gt, need_mask=True, need_matrix=False,
                                            is_simulate=True)  # TODO:是否需要权重矩阵,需要即frequency1,否则frequency1'
            loss += mosaic_loss_freq

        new_label = transform_opt(firstcube, opt.msfa_size)
        new_label = new_label.detach() # TODO: gradient stop
        input_sparse_raw, input_raw = get_sparsecube_raw(new_label, opt.msfa_size)
        # cube loss
        curoutput = model(input_raw, input_sparse_raw)
        cube_loss = criterion[0](curoutput, new_label)
        loss += cube_loss

        if opt.is_freq_loss:
            # frequency: cube loss
            cube_loss_freq = criterion[2](curoutput, new_label, need_mask=False, need_matrix=False,
                                          is_simulate=True)  # TODO:是否需要权重矩阵,需要即frequency1,否则frequency1'
            loss += cube_loss_freq

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler != None:
            scheduler.step()

        if opt.is_freq_loss:
            train_bar.set_description(desc='Epoch[%d/%d] Mosaic_Loss: %.9f, Mosaic_Loss_Frequency: %.9f, Cube_Loss: %.9f, Cube_Loss_Frequency: %.9f, All_Loss: %.9f, lr: %.9f' %
                                       (epoch, num_epochs, mosaic_loss.item() / batchSize_count, mosaic_loss_freq.item() / batchSize_count, cube_loss.item() / batchSize_count,
                                        cube_loss_freq.item() / batchSize_count, loss.item() / batchSize_count, lr))
        else:
            train_bar.set_description(desc='Epoch[%d/%d] Mosaic_Loss: %.9f, Cube_Loss: %.9f, All_Loss: %.9f, lr: %.9f' %
                     (epoch, num_epochs, mosaic_loss.item() / batchSize_count, cube_loss.item() / batchSize_count, loss.item() / batchSize_count, lr))

    if opt.is_freq_loss:
        logger.info('Epoch[%d/%d] Mosaic_Loss: %.9f, Mosaic_Loss_Frequency: %.9f, Cube_Loss: %.9f, Cube_Loss_Frequency: %.9f, All_Loss: %.9f, lr: %.9f' %
            (epoch, num_epochs, mosaic_loss.item() / batchSize_count, mosaic_loss_freq.item() / batchSize_count, cube_loss.item() / batchSize_count,
             cube_loss_freq.item() / batchSize_count, loss.item() / batchSize_count, lr))
    else:
        logger.info('Epoch[%d/%d] Mosaic_Loss: %.9f, Cube_Loss: %.9f, All_Loss: %.9f, lr: %.9f' %
            (epoch, num_epochs, mosaic_loss.item() / batchSize_count, cube_loss.item() / batchSize_count, loss.item() / batchSize_count, lr))

# This is actually for validate
def validate(opt, test_loader, method, model, epoch, num_epochs):
    test_bar = tqdm(test_loader)
    model.eval()
    metric_psnr = AverageMeter()
    metric_ssim = AverageMeter()
    metric_sam = AverageMeter()
    metric_ergas = AverageMeter()

    with torch.no_grad():
        for batch in test_bar:
            input_raw, input_sparse_raw, gt = Variable(batch[0]), Variable(batch[1]), Variable(batch[2], requires_grad=False)
            N, C, H, W = batch[0].size()
            input_raw = input_raw.cuda()
            input_sparse_raw = input_sparse_raw.cuda()
            gt = gt.cuda()

            output = model(input_raw, input_sparse_raw)

            psnr = criterion_psnr(output, gt)
            ssim = criterion_ssim(output, gt)
            sam = criterion_sam(output, gt)
            ergas = criterion_ergas(output, gt)

            # record metrics
            metric_psnr.update(psnr.data)
            metric_ssim.update(ssim.data)
            metric_sam.update(sam.data)
            metric_ergas.update(ergas.data)

            test_bar.set_description(desc='Epoch[%d/%d] psnr: %.4f, ssim: %.4f, sam: %.4f, ergas: %.4f' %
                                          (epoch, num_epochs, metric_psnr.avg, metric_ssim.avg, metric_sam.avg, metric_ergas.avg))
    # logger
    logger.info('Epoch[%d/%d] psnr: %.4f, ssim: %.4f, sam: %.4f, ergas: %.4f' %
                                          (epoch, num_epochs, metric_psnr.avg, metric_ssim.avg, metric_sam.avg, metric_ergas.avg))


def main():
    cudnn.benchmark = True
    for epoch in range(1, opt.end_epoch+1):
        train(opt, train_loader, optimizer, scheduler, opt.method, model, criterion, epoch, opt.end_epoch)
        validate(opt, test_loader, opt.method, model, epoch, opt.end_epoch)
        # TODO:做消融实验的时候节约时间空间，不要每个epoch都输出
        if epoch % 10 == 0:
            print(f'Saving to {opt.outf}')
            save_checkpoint(opt.outf, epoch, model, optimizer)


if __name__ == '__main__':
    main()
    print(torch.__version__)
