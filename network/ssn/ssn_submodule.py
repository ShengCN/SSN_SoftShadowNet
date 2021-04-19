import sys
sys.path.append("..")

import torch 
import torch.nn as nn
import torch.nn.functional as F

from utils.net_utils import compute_differentiable_params
from params import params

def get_layer_info(out_channels, activation_func='relu'):
    parameter = params().get_params()
    if parameter.norm == 'batch_norm':
        norm_layer = nn.BatchNorm2d(out_channels, momentum=0.9)
    elif parameter.norm == 'group_norm':
        if out_channels >= 32:
            group_num = 32
        else:
            group_num = 1
            
        norm_layer = nn.GroupNorm(group_num, out_channels)
    elif parameter.norm == 'None':
        norm_layer = None
    else:
        raise Exception('norm name error')
    
    if activation_func == 'relu':
        activation_func = nn.ReLU()
    elif activation_func == 'prelu':
        activation_func = nn.PReLU(out_channels)
        
    return norm_layer, activation_func

# add coord_conv
class add_coords(nn.Module):
    def __init__(self, use_cuda=True):
        super(add_coords, self).__init__()
        self.use_cuda = use_cuda
        
    def forward(self, input_tensor):
        b, c, dim_y, dim_x = input_tensor.shape
        xx_ones = torch.ones([1, 1, 1, dim_x], dtype=torch.int32)
        yy_ones = torch.ones([1, 1, 1, dim_y], dtype=torch.int32)

        xx_range = torch.arange(dim_y, dtype=torch.int32)
        yy_range = torch.arange(dim_x, dtype=torch.int32)
        xx_range = xx_range[None, None, :, None]
        yy_range = yy_range[None, None, :, None]

        xx_channel = torch.matmul(xx_range, xx_ones)
        yy_channel = torch.matmul(yy_range, yy_ones)

        # transpose y
        yy_channel = yy_channel.permute(0, 1, 3, 2)

        xx_channel = xx_channel.float() / (dim_y - 1)
        yy_channel = yy_channel.float() / (dim_x - 1)

        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        xx_channel = xx_channel.repeat(b, 1, 1, 1)
        yy_channel = yy_channel.repeat(b, 1, 1, 1)

        if torch.cuda.is_available and self.use_cuda:
            input_tensor = input_tensor.cuda()
            xx_channel = xx_channel.cuda()
            yy_channel = yy_channel.cuda()
        out = torch.cat([input_tensor, xx_channel, yy_channel], dim=1)
        return out

class Conv(nn.Module):
    """ (convolution => [BN] => ReLU) """
    
    def __init__(self, in_channels, out_channels, kernel_size=3, conv_stride=1, padding=1, bias=True, activation_func='relu'):
        super().__init__()
        
        parameter = params().get_params()
        
        if parameter.prelu:
            activation_func = 'prelu'
        else:
            activation_func = 'relu'
        norm_layer, activation_func = get_layer_info(out_channels, activation_func)
        if norm_layer is not None:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,stride=conv_stride, kernel_size=kernel_size, padding=padding, bias=bias),
                norm_layer,
                activation_func)
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,stride=conv_stride, kernel_size=kernel_size, padding=padding, bias=bias),
                activation_func)

    def forward(self, x):
        return self.conv(x)

class Up(nn.Module):
    """ Upscaling then conv """
    
    def __init__(self, in_channels, out_channels, activation_func='relu'):
        super().__init__()
        
        parameter = params().get_params()
        if parameter.prelu:
            activation_func = 'prelu'
        else:
            activation_func = 'relu'
        norm_layer, activation_func = get_layer_info(out_channels, activation_func)
        
        up_layer = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up = nn.Sequential(
                    up_layer,
                    Conv(in_channels, in_channels//4, activation_func= activation_func),
                    norm_layer,
                    activation_func)

    def forward(self, x):
        return self.up(x)

class PSPUpsample(nn.Module):
    def __init__(self, in_channels, out_channels, scale_x, scale_y):
        super().__init__()
        self.conv = Conv(in_channels, out_channels)
        self.scale_x = scale_x
        self.scale_y = scale_y

    def forward(self, x):
        h, w = self.scale_y * x.size(2), self.scale_x * x.size(3)
        p = F.upsample(input=x, size=(h, w), mode='bilinear')
        return self.conv(p)

class PSP(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # pooling
        self.pool2 = nn.AdaptiveAvgPool2d((2,2))
        self.pool4 = nn.AdaptiveAvgPool2d((4,4))
        self.pool8 = nn.AdaptiveAvgPool2d((8,8))

        # conv -> compress channels
        avg_channel = in_channels//4
        self.conv2 = Conv(in_channels, avg_channel)
        self.conv4 = Conv(in_channels, avg_channel)
        self.conv8 = Conv(in_channels, avg_channel)
        self.conv16 = Conv(in_channels, avg_channel)

        # up sapmle -> match dimension
        self.up2 = PSPUpsample(avg_channel, avg_channel, 16//2, 16//2)
        self.up4 = PSPUpsample(avg_channel, avg_channel, 16//4, 16//4)
        self.up8 = PSPUpsample(avg_channel, avg_channel, 16//8, 16//8)

    def forward(self, x):
        x2 = self.up2(self.conv2(self.pool2(x))) 
        x4 = self.up4(self.conv4(self.pool4(x))) 
        x8 = self.up8(self.conv8(self.pool8(x))) 
        x16 = self.conv16(x)
        return  torch.cat((x2, x4, x8, x16), dim=1)

class Up_Stream(nn.Module):
    """ Up Stream Sequence """
    
    def __init__(self, out_channels=3):
        super(Up_Stream, self).__init__()
        
        parameter = params().get_params()
        if parameter.prelu:
            activation_func='prelu'
        else:
            activation_func='relu'

        input_channel = 512
        norm_layer, activation_func = get_layer_info(input_channel, activation_func)

        self.up_16_16_1 = Conv(input_channel, 256,  activation_func=activation_func)
        self.up_16_16_2 = Conv(768, 512, activation_func=activation_func)
        self.up_16_16_3 = Conv(1024, 512, activation_func=activation_func)

        self.up_16_32 = Up(1024, 256, activation_func=activation_func)
        self.up_32_32_1 = Conv(512, 256, activation_func=activation_func)

        self.up_32_64 = Up(512, 128, activation_func=activation_func)
        self.up_64_64_1 = Conv(256, 128, activation_func=activation_func)

        self.up_64_128 = Up(256, 64, activation_func=activation_func)
        self.up_128_128_1 = Conv(128, 64, activation_func=activation_func)

        self.up_128_256 = Up(128, 32, activation_func=activation_func)
        self.out_conv = Conv(64, out_channels, activation_func='relu')
        
    def forward(self, l, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11):
        # batch_size, c, h, w = l.size()
        
        # import pdb; pdb.set_trace()
        # multiple channel ibl
        y = l.view(-1, 512, 1, 1).repeat(1, 1, 16, 16)

        y = self.up_16_16_1(y)    # 256 x 16 x 16

        y = torch.cat((x10,y), dim=1)   # 768 x 16 x 16
        # print(y.size())

        y = self.up_16_16_2(y)          # 512 x 16 x 16
        # print(y.size())

        y= torch.cat((x9,y), dim=1)     # 1024 x 16 x 16
        # print(y.size())
        
        # import pdb; pdb.set_trace()
        y = self.up_16_16_3(y)          # 512 x 16 x 16
        # print(y.size())

        y = torch.cat((x8, y), dim=1)   # 1024 x 16 x 16
        # print(y.size())
        
        # import pdb; pdb.set_trace()
        y = self.up_16_32(y)            # 256 x 32 x 32
        # print(y.size())
        
        
        y = torch.cat((x7, y), dim=1)
        y = self.up_32_32_1(y)          # 256 x 32 x 32
        # print(y.size())

        y = torch.cat((x6, y), dim=1)
        y = self.up_32_64(y)
        # print(y.size())
        y = torch.cat((x5, y), dim=1)
        y = self.up_64_64_1(y)          # 128 x 64 x 64
        # print(y.size())

        y = torch.cat((x4, y), dim=1)
        y= self.up_64_128(y)
        # print(y.size())
        y = torch.cat((x3, y), dim=1)
        y = self.up_128_128_1(y)        # 64 x 128 x 128
        # print(y.size())

        y = torch.cat((x2, y), dim=1)
        y = self.up_128_256(y)          # 32 x 256 x 256
        # print(y.size())
        
        y = torch.cat((x1, y), dim=1)

        y = self.out_conv(y)          # 3 x 256 x 256
        
        return y