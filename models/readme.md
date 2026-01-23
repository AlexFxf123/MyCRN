# 根据开源代码自定义的CRN模型
# 默认是原版模型，python文件是重写的模型代码
# resnet18和原版不一样，主要是ReLU的位置不一样，但不影响结果，另外输出4层特征，而不是原来的最后1层，参考代码为crn_img_backbone.py
# secondfpn对应resnet18输出的4组特征，经过网络后合并到一起，参考代码为crn_img_neck3.py
# depth_net和view_aggregation_net有源代码，直接copy即可
# depth_net的BasicBlock使用和resnet18中的一样,ReLU的位置不太一样

