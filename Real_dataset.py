import torch
from torch.utils.data import Dataset
import numpy as np
import random
from scipy.io import loadmat
import os
from utils import mask_input
import pickle
import tqdm

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".tif", ".mat", ".h5"])

def crop_to_patch(img, size, stride):
    '''
    Args:
        img: ndarray, [h,w,c]
        size: crop_size, int
        stride: crop_stride, int

    Returns:
        patches: list
    '''
    H, W = img.shape[0], img.shape[1]
    patches = []
    for h in range(0, H, stride):
        for w in range(0, W, stride):
            if h + size <= H and w + size <= W:
                patch = img[h:h+size, w:w+size, :]
                patches.append(patch)
    return patches

class RealDataset(Dataset):
    def __init__(self, opt, type='train', batch_size=16, patch_size=64, stride=32, augment=True):
        self.type = type
        self.msfa_size = opt.msfa_size
        self.mosaic_bands = opt.msfa_size ** 2
        self.augment = augment

        if type == 'train':
            cache_path = os.path.join(opt.cache_path,
                                      type + '_batch_size_' + str(batch_size) +
                                      '_patch_' + str(patch_size) + '_stride_' + str(stride) + '_cache.pkl')
        elif type == 'test':
            cache_path = os.path.join(opt.cache_path, type + '_cache.pkl')

        # if no cache, then create
        if not os.path.exists(cache_path):
            self.raw, self.sparse_raw = [], [] # list

            if self.type == 'train':
                data_root = opt.train_dir
            elif self.type == 'test':
                data_root = opt.test_dir
            print(f'No cache file. Generate it from: {data_root}.')
            image_filenames = [os.path.join(data_root, x) for x in os.listdir(data_root) if is_image_file(x)]
            image_filenames.sort()

            for img_file in tqdm.tqdm(image_filenames):
                img = loadmat(img_file)['raw'] # img: [h,w,1], mosaic raw, already normalized to [0,1]
                img = img.astype(np.float32)
                #img = (img-img.min()) / (img.max()-img.min()) # normalization: [0,1]

                # generate sparse_raw image from mosaic raw image
                sparse_raw = np.zeros([img.shape[0], img.shape[1], self.msfa_size**2]).astype(np.float32) # [h,w,c]
                for i in range(self.msfa_size):
                      for j in range(self.msfa_size):
                          sparse_raw[i::self.msfa_size, j::self.msfa_size, i*self.msfa_size+j] = img[i::self.msfa_size, j::self.msfa_size, 0].copy()

                # tmp = np.tile(img, reps=(1,1,opt.msfa_size ** 2)) # tmp, for generate sparse_raw: [h,w,c]
                # sparse_raw = mask_input(tmp, msfa_size=self.msfa_size) # sparse_raw: [h,w,c]

                # generate raw image
                raw = img.copy() # raw: [h,w,1]

                if self.type == 'train':
                    raw_patches = crop_to_patch(raw, patch_size, stride)
                    sparse_raw_patches = crop_to_patch(sparse_raw, patch_size, stride)
                    self.raw += raw_patches # iterable value可以相加
                    self.sparse_raw += sparse_raw_patches

                elif self.type == 'test':
                    self.raw.append(raw)
                    self.sparse_raw.append(sparse_raw)

            with open(cache_path, 'wb') as f:
                pickle.dump([self.raw, self.sparse_raw], f)

        # if cache exist, read it
        else:
            print('Load data from cache file:', cache_path)
            with open(cache_path, 'rb') as f:
                self.raw, self.sparse_raw = pickle.load(f)
    def data_augment(self, img, rotTimes, vFlip, hFlip):
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

    def __len__(self):
        return len(self.raw)

    def __getitem__(self, index):
        # shape: [h,w,c]->[c,h,w] for fetching data
        raw = np.transpose(self.raw[index], (2, 0, 1)) # raw: [c,h,w]
        sparse_raw = np.transpose(self.sparse_raw[index], (2, 0, 1)) # sparse_raw: [c,h,w]

        if self.type == 'train':
            # data argument
            rotTimes = random.randint(0, 3)
            vFlip = random.randint(0, 1)
            hFlip = random.randint(0, 1)
            if self.augment:
                raw = self.data_augment(raw, rotTimes, vFlip, hFlip)
                sparse_raw = self.data_augment(sparse_raw, rotTimes, vFlip, hFlip)

        return np.ascontiguousarray(raw), np.ascontiguousarray(sparse_raw)


