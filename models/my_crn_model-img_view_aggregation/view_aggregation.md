=== view_aggregation_net 模型结构 ===
ViewAggregation(
  (reduce_conv): Sequential(
    (0): Conv2d(160, 160, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (1): BatchNorm2d(160, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
  )
  (conv): Sequential(
    (0): Conv2d(160, 160, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (1): BatchNorm2d(160, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
    (3): Conv2d(160, 160, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (4): BatchNorm2d(160, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (5): ReLU(inplace=True)
  )
  (out_conv): Sequential(
    (0): Conv2d(160, 80, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  )
)
view_aggregation_net 输出形状: torch.Size([1, 80, 32, 32])
view_aggregation_net 模型参数总数: 807440