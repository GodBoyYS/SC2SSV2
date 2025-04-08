import torch
import torch.nn as nn

class ClassSpecificBandEnhancement(nn.Module):
    def __init__(self, input_size=200, num_classes=16):
        super(ClassSpecificBandEnhancement, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        # 类别特异性波段权重参数
        self.class_weights = nn.Parameter(torch.ones(num_classes, input_size))

    def forward(self, class_labels):
        # 输入：class_labels [batch]
        # 输出：w_class [batch, input_size]
        w_class = torch.sigmoid(self.class_weights[class_labels])
        return w_class