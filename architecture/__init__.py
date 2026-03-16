import torch
from .MynewModel import FGLN


def model_generator(method, msfa_size, pretrained_model_path=None):
    if method == 'FGLN':
        model = FGLN(msfa_size=4, SI_type='FPC', num_blocks=2).cuda() ## TODO: 注意SI_type再开始训练

    else:
        print(f'Method {method} is not defined !!!!')
    # 读取.pth文件
    if pretrained_model_path is not None:
        print(f'load model from {pretrained_model_path}')
        checkpoint = torch.load(pretrained_model_path)
        model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()},
                              strict=True)
    return model

