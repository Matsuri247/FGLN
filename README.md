# FGLN
The code implementation of paper "Frequency-Guided Lightweight Network for Unsupervised Spectral Demosaicing".

# Environment Setup Overview
```
Python=3.11.5
torchvision==0.16.1
torchaudio==2.1.1
torch==2.1.1
h5py
hdf5storage
tqdm
torchinfo
```

# Data Preparation
You can find ARAD dataset and Chikusei dataset from ([here](https://github.com/bowenzhao-zju/PPIE-SSARN)). Make sure you place the dataset as the following form:
```
|--demosaicing_MPEFormer
    |--dataset 
        |--ARAD
            |--train
                |--ARAD_1K_0001_16.mat
                |--ARAD_1K_0002_16.mat
                ： 
                |--ARAD_1K_0900_16.mat
            |--test
                |--ARAD_1K_0901_16.mat
                |--ARAD_1K_0902_16.mat
                ： 
                |--ARAD_1K_0950_16.mat
        |--Chikusei
            |--train
                |--001_16.mat
                |--002_16.mat
                ： 
                |--090_16.mat
            |--test
                |--091_16.mat
                |--092_16.mat
                ： 
                |--100_16.mat
```

# Train
【train_experiment.py】
```
--method MPEFormer --msfa_size 4 --dataset ARAD --batch_size 16 --end_epoch 4000 --init_lr 4e-4 --outf ./train_exp/MPEFormer/ --train_dir ./dataset/ARAD/train/ --test_dir ./dataset/ARAD/test/
--method MPEFormer --msfa_size 4 --dataset Chikusei --batch_size 16 --end_epoch 4000 --init_lr 4e-4 --outf ./train_exp/MPEFormer/ --train_dir ./dataset/Chikusei/train/ --test_dir ./dataset/Chikusei/test/
```

# Test
【test_experiment.py】
```
--method MPEFormer --msfa_size 4 --dataset ARAD --pretrained_model_path ./model_pth/MPEFormer/ARAD/MPEFormer_ARAD.pth --test_dir ./dataset/ARAD/test/ --outf ./test_exp/MPEFormer/
--method MPEFormer --msfa_size 4 --dataset Chikusei --pretrained_model_path ./model_pth/MPEFormer/Chikusei/MPEFormer_Chikusei.pth --test_dir ./dataset/Chikusei/test/ --outf ./test_exp/MPEFormer/
```

# Citation
If you find this repo useful, please consider citing our works.
```
@ARTICLE{MPEFormer,
  author={Wu, Yaohang and Liu, Jinyang and Dian, Renwei and Li, Shutao and Yang, Yining},
  journal={IEEE Transactions on Image Processing}, 
  title={Mosaic Pattern Excavation Transformer for Spectral Imaging}, 
  year={2025},
  volume={},
  number={},
  pages={1-1},
  keywords={Deep learning;spectral imaging;multispectral filter array;attention mechanism},
  doi={10.1109/TIP.2025.3633159}}
```
