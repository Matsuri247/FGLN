from torch.utils.data import Dataset
import numpy as np
import random
from scipy.io import loadmat
import h5py
import os
from utils import mask_input

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".tif", ".mat", ".h5"])

def randcrop_one(raw, sparse_raw, target, size):
    '''
    Args:
        raw, sparse_raw, target: ndarray, [h,w,c]
        size: crop_size, int

    Returns:
        raw_patch, sparse_raw_patch, target_patch: ndarray, [h,w,c]
    '''
    assert raw.shape[:2] == sparse_raw.shape[:2] == target.shape[:2], 'the shape of raw, sparse_raw, target must be equal.'
    (H, W) = raw.shape[:2]
    crop_size = size
    Height = random.randint(0, H-crop_size-1)
    Width = random.randint(0, W-crop_size-1)
    return (raw[Height:(Height + crop_size), Width:(Width + crop_size), :],
            sparse_raw[Height:(Height + crop_size), Width:(Width + crop_size), :],
            target[Height:(Height + crop_size), Width:(Width + crop_size), :])

def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)

class TrainARADDataset(Dataset):
    def __init__(self, data_root, msfa_size=4, patch_size=160, augment=False):
        self.image_filenames = [os.path.join(data_root, x) for x in os.listdir(data_root) if is_image_file(x)]
        self.msfa_size = msfa_size
        self.mosaic_bands = self.msfa_size ** 2
        self.crop_size = calculate_valid_crop_size(patch_size, msfa_size)
        #self.crop_size = patch_size
        self.augment = augment

    def data_arguement(self, img, rotTimes, vFlip, hFlip):
        # [c,h,w]
        # Random rotation
        for j in range(rotTimes):
            img_ = np.rot90(img.copy(), axes=(1, 2))
            img = img_.copy()
        # Random horizontal Flip
        for j in range(hFlip):
            img = img[:, :, ::-1].copy()
        # Random vertical Flip
        for j in range(vFlip):
            img = img[:, ::-1, :].copy()
        return img

    def __getitem__(self, index):
        with h5py.File(self.image_filenames[index], 'r') as mat:
            img = np.float32(np.array(mat['cube'])) # [c,w,h]
            img = np.transpose(img, (2, 1, 0)) # [c,w,h] -> [h,w,c]
        mat.close()

        # generate sparse_raw image from GT MSI
        sparse_raw = mask_input(img, msfa_size=self.msfa_size) # [h,w,c]
        # generate raw image
        raw = sparse_raw.sum(axis=2, keepdims=True) # [h,w,c]
        # generate GT
        target = img.copy()

        # 切块：拿一张图出来就随机切出一块crop_size大小的patch，并不是一张图切出好多块patches
        raw_patch, sparse_raw_patch, target_patch = randcrop_one(raw, sparse_raw, target, self.crop_size)

        # permute to [c,h,w]
        raw_patch = np.transpose(raw_patch, (2, 0, 1))  # [c,h,w]
        sparse_raw_patch = np.transpose(sparse_raw_patch, (2, 0, 1)) # [c,h,w]
        target_patch = np.transpose(target_patch, (2, 0, 1))  # [c,h,w]

        # argument
        rotTimes = random.randint(0, 3)
        vFlip = random.randint(0, 1)
        hFlip = random.randint(0, 1)
        if self.augment:
            raw_patch = self.data_arguement(raw_patch, rotTimes, vFlip, hFlip)
            sparse_raw_patch = self.data_arguement(sparse_raw_patch, rotTimes, vFlip, hFlip)
            target_patch = self.data_arguement(target_patch, rotTimes, vFlip, hFlip)

        return np.ascontiguousarray(raw_patch), np.ascontiguousarray(sparse_raw_patch), np.ascontiguousarray(target_patch)

    def __len__(self):
        return len(self.image_filenames)

# Test with image
class TestARADDataset(Dataset):
    def __init__(self, data_root, msfa_size=4):
        self.image_filenames = [os.path.join(data_root, x) for x in os.listdir(data_root) if is_image_file(x)]
        self.image_filenames.sort()
        self.msfa_size = msfa_size
        self.mosaic_bands = self.msfa_size ** 2

    def __getitem__(self, index):
        with h5py.File(self.image_filenames[index], 'r') as mat:
            img = np.float32(np.array(mat['cube'])) # [c,w,h]
            img = np.transpose(img, (2, 1, 0)) # [c,w,h] -> [h,w,c]
        mat.close()

        # generate sparse_raw image from GT MSI
        sparse_raw = mask_input(img, msfa_size=self.msfa_size) # [h,w,c]
        # generate raw image
        raw = sparse_raw.sum(axis=2, keepdims=True) # [h,w,c]

        # permute to [c,h,w]
        raw = np.transpose(raw, (2, 0, 1)) # [c,h,w]
        sparse_raw = np.transpose(sparse_raw, (2, 0, 1)) # [c,h,w]
        target = np.transpose(img.copy(), (2, 0, 1))  # [c,h,w]

        return np.ascontiguousarray(raw), np.ascontiguousarray(sparse_raw), np.ascontiguousarray(target)

    def __len__(self):
        return len(self.image_filenames)





