SECONDFPN(
  (deblocks): ModuleList(
    (0): Sequential(
      (0): Conv2d(64, 64, kernel_size=(4, 4), stride=(4, 4), bias=False)
      (1): BatchNorm2d(64, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
      (2): ReLU(inplace=True)
    )
    (1): Sequential(
      (0): Conv2d(128, 64, kernel_size=(2, 2), stride=(2, 2), bias=False)
      (1): BatchNorm2d(64, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
      (2): ReLU(inplace=True)
    )
    (2): Sequential(
      (0): ConvTranspose2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (1): BatchNorm2d(64, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
      (2): ReLU(inplace=True)
    )
    (3): Sequential(
      (0): ConvTranspose2d(512, 64, kernel_size=(2, 2), stride=(2, 2), bias=False)
      (1): BatchNorm2d(64, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
      (2): ReLU(inplace=True)
    )
  )
)





SECONDFPN(
  (lateral_convs): ModuleList(
    (0): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (1): Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (2): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (3): Conv2d(512, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
  (fpn_convs): ModuleList(
    (0): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
  )
)


=== 前向传播结果 ===
输出 0: torch.Size([2, 64, 32, 32])
  预期输出通道: 64
输出 1: torch.Size([2, 64, 32, 32])
  预期输出通道: 64
输出 2: torch.Size([2, 64, 32, 32])
  预期输出通道: 64
输出 3: torch.Size([2, 64, 32, 32])
  预期输出通道: 64
