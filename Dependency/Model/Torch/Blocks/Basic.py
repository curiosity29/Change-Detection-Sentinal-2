import torch
import torch.nn as nn

class CoBaRe(nn.Module):
    def __init__(self, in_ch = 3, out_ch = 3, dilate = 1, padding = 1):
        super(CoBaRe, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding = dilate, dilation = dilate)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace = True)
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class CoSigUp(nn.Module):
    def __init__(self, in_ch = 3, out_ch = 3, size = 256, dilate = 1, padding = 1):
        super(CoSigUp, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding = padding, dilation = dilate)
        self.sigmoid = nn.Sigmoid()
        self.up = nn.Upsample(size = size, mode = "bilinear", align_corners = False)
    def forward(self, x):
        return self.up(self.sigmoid(self.conv(x)))
    

