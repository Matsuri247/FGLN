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
Dataset downloads:  

You can find ARAD-1k dataset from [here](https://github.com/bowenzhao-zju/PPIE-SSARN)  

CAVE dataset is provided [here](https://drive.google.com/drive/folders/1C9P36WrEf7kD4vQHIL6tU5rIs8MQzrjr?usp=sharing)  

Make sure you place the dataset as the following form:
```
|--demosaicing_FGLN
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
        |--CAVE
            |--train
                |--xxx.mat
                |--xxx.mat
                ： 
                |--xxx.mat
            |--test
                |--xxx.mat
                |--xxx.mat
                ： 
                |--xxx.mat
```


# Train
【train_experiment_frequency1.py】
```
# ARAD dataset training
--method FGLN --msfa_size 4 --dataset ARAD --batch_size 10 --end_epoch 1000 --init_lr 4e-4 --lr_step 100000 --outf ./train_exp/FGLN/ --train_dir ./dataset/ARAD/train/ --test_dir /home/lab206/mycode/demosaicing_MPEFormer/dataset/ARAD/test/ --cache ./training_cache/
# CAVE dataset training
--method FGLN --msfa_size 4 --dataset CAVE --batch_size 20 --end_epoch 1000 --init_lr 4e-4 --lr_step 100000 --outf ./train_exp/FGLN/ --train_dir ./dataset/CAVE/train/ --test_dir /home/lab206/mycode/demosaicing_MPEFormer/dataset/CAVE/test/ --cache ./training_cache/
```

# Test
【test_experiment.py】
```
# ARAD dataset testing
--method FGLN --msfa_size 4 --dataset ARAD --pretrained_model_path ./model_zoo/FGLN/ARAD/net_1000epoch.pth --test_dir ./dataset/ARAD/test/ --outf ./test_exp/FGLN/ --cache ./training_cache/
# CAVE dataset tesing
--method FGLN --msfa_size 4 --dataset CAVE --pretrained_model_path ./model_zoo/FGLN/CAVE/net_1000epoch.pth --test_dir ./dataset/CAVE/test/ --outf ./test_exp/FGLN/ --cache ./training_cache/
```


# Citation
If you find this repo useful, please consider citing our works.
```
    Waiting for acceptance
```
