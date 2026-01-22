import torch
from exps.det.CRN_r18_256x704_128x128_4key import CRNLightningModel

if __name__ == '__main__':
    model = CRNLightningModel()
    total_params = sum(p.numel() for p in model.parameters())
    total_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)
    print(f'模型参数总数: {total_params}')
    print(f'模型参数大小: {total_size:.2f} MB')

    