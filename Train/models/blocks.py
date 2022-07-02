import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import logging

def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find(
                'Linear') == 0) and hasattr(m, 'weight'):
            if init_type == 'gaussian':
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    return init_fun

def freeze(module):
    for param in module.parameters():
        param.requires_grad = False

def unfreeze(module):
    for param in module.parameters():
        param.requires_grad = True


def get_optimizer(opt, model):
    lr           = float(opt['hyper_params']['lr'])
    beta1        = float(opt['model']['beta1'])
    weight_decay = float(opt['model']['weight_decay'])
    opt_name     = opt['model']['optimizer']

    optim_params = []
    # weight decay
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue  # frozen weights

        if key[-4:] == 'bias':
            optim_params += [{'params': value, 'weight_decay': 0.0}]
        else:
            optim_params += [{'params': value,
                              'weight_decay': weight_decay}]

    if opt_name == 'Adam':
        return optim.Adam(optim_params,
                            lr=lr,
                            betas=(beta1, 0.999),
                            eps=1e-5)
    else:
        err = '{} not implemented yet'.format(opt_name)
        logging.error(err)
        raise NotImplementedError(err)


def get_activation(activation):
    act_func = {
        'relu':nn.ReLU(),
        'sigmoid':nn.Sigmoid(),
        'tanh':nn.Tanh(),
        'prelu':nn.PReLU(),
        'leaky':nn.LeakyReLU(0.2)
        }
    if activation not in act_func.keys():
        logging.error("activation {} is not implemented yet".format(activation))
        assert False

    return act_func[activation]

def get_norm(out_channels, norm_type='Instance'):
    norm_set = ['Instance', 'Batch', 'Group']
    if norm_type not in norm_set:
        err = "Normalization {} has not been implemented yet"
        logging.error(err)
        raise ValueError(err)

    if norm_type == 'Instance':
        return nn.InstanceNorm2d(out_channels, affine=True)

    if norm_type == 'Batch':
        return nn.BatchNorm2d(out_channels)

    if norm_type == 'Group':
        raise NotImplementedError


class Conv_1x1(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, norm_type='Batch', activation='relu'):
        super().__init__()

        act_func =get_activation(activation)
        norm_layer = get_norm(out_channels, norm_type)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=True, padding_mode='reflect'),
            norm_layer,
            act_func)

    def forward(self, x):
        return self.conv(x)


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, norm_type='Batch', activation='relu'):
        super().__init__()

        act_func   = get_activation(activation)
        norm_layer = get_norm(out_channels, norm_type)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True, padding_mode='reflect'),
            norm_layer,
            act_func)

    def forward(self, x):
        return self.conv(x)


class DConv(nn.Module):
    """ Double Conv Layer
    """
    def __init__(self, in_channels, out_channels, norm_type='Batch', activation='relu', resnet=True):
        super().__init__()

        self.in_equal_out = in_channels == out_channels
        self.resnet = resnet
        self.conv1 = Conv(in_channels, out_channels, norm_type=norm_type, activation=activation)
        self.conv2 = Conv(out_channels, out_channels, norm_type=norm_type, activation=activation)

    def forward(self, x):
        if self.resnet and self.in_equal_out:
            x = x + self.conv1(x)
            return self.conv2(x) + x

        elif self.resnet:
            x = self.conv1(x)
            return self.conv2(x) + x

        else:
            return self.conv2(self.conv1(x))


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, norm_type='Batch', activation = 'relu', resnet=True):
        super().__init__()

        self.up    = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dconv = DConv(in_channels + in_channels//2, out_channels, norm_type=norm_type, resnet=resnet)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x  = torch.cat([x2, x1], dim=1)
        return self.dconv(x)


class Encoder(nn.Module):
    def __init__(self, in_channels, norm_type='Batch', mid_act='relu', resnet=True):
        """ Default auto encoder
            From U-net paper, default output for encoder part is 512
        """
        super().__init__()
        self.indconv = Conv(in_channels, 64-in_channels, norm_type=norm_type, activation=mid_act)

        self.to_128 = Conv(64, 128, stride=2, norm_type=norm_type, activation=mid_act)
        self.dconv128 = DConv(128, 128, norm_type=norm_type, activation=mid_act, resnet=resnet)

        self.to_256 = Conv(128, 256, stride=2, norm_type=norm_type, activation=mid_act)
        self.dconv256 = DConv(256, 256, norm_type=norm_type, activation=mid_act, resnet=resnet)

        self.to_512 = Conv(256, 512, stride=2, norm_type=norm_type, activation=mid_act)
        self.dconv512 = DConv(512, 512, norm_type=norm_type, activation=mid_act, resnet=resnet)

        self.to_1024 = Conv(512, 1024, stride=2, norm_type=norm_type, activation=mid_act)


    def forward(self, x):
        x_in = self.indconv(x)
        x64 = torch.cat((x_in, x), dim=1)
        x128 = self.dconv128(self.to_128(x64))
        x256 = self.dconv256(self.to_256(x128))
        x512 = self.dconv512(self.to_512(x256))
        x1024 = self.to_1024(x512)

        return x1024, x512, x256, x128, x64


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, norm_type='Batch', mid_act='relu', resnet=True, out_act='sigmoid'):
        """ Default auto encoder
            From U-net paper, default input for decoder part is 1024
        """
        super().__init__()
        self.up1024 = Up(in_channels, 512, norm_type=norm_type, activation=mid_act, resnet=resnet)
        self.up512  = Up(512, 256, norm_type=norm_type, activation=mid_act, resnet=resnet)
        self.up256  = Up(256, 128, norm_type=norm_type, activation=mid_act, resnet=resnet)
        self.up128  = Up(128, 64, norm_type=norm_type, activation=mid_act, resnet=resnet)
        self.out    = Conv(64, out_channels, norm_type=norm_type, activation=out_act)

    def forward(self, x, prev_x):
        y = self.up1024(x, prev_x[1])
        y = self.up512(y, prev_x[2])
        y = self.up256(y, prev_x[3])
        y = self.up128(y, prev_x[4])
        return self.out(y)


class Unet(nn.Module):
    """ Standard Unet Implementation
        src: https://arxiv.org/pdf/1505.04597.pdf
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 norm_type='Batch',
                 out_act='sigmoid',
                 resnet=False,
                 freeze_layers=False):

        super(Unet, self).__init__()
        self.encoder = Encoder(in_channels, norm_type=norm_type)
        self.bottle  = self.get_bottle(resnet, norm_type)
        self.decoder = Decoder(out_channels, norm_type=norm_type,out_act=out_act)


    def forward(self, x):
        import pdb; pdb.set_trace()
        prev_x = self.encoder(x)
        bottle = self.bottle(prev_x[0])
        return self.decoder(bottle, prev_x)


    def get_bottle(self, resnet, norm_type):
        if not resnet:
            bottle = nn.Sequential(
                DConv(1024, 1024, norm_type=norm_type, resnet=False)
            )
        else:
            bottle = nn.Sequential(
                DConv(1024, 1024, norm_type=norm_type, resnet=True),
                DConv(1024, 1024, norm_type=norm_type, resnet=True),
                DConv(1024, 1024, norm_type=norm_type, resnet=True),
            )
        return bottle


class Refine_Net(nn.Module):
    def __init__(self, in_channels:int, layer_channels:list, out_channels:int, act='relu', norm_type='Batch', out_act=None):
        if len(layer_channels)  == 0:
            raise ValueError("Empty layer channels")

        super(Refine_Net, self).__init__()

        refine_layers = [Conv(in_channels, layer_channels[0], stride=1, norm_type=norm_type)]
        for i in range(1, len(layer_channels)):
           refine_layers.append(Conv(layer_channels[i-1], layer_channels[i], stride=1, norm_type=norm_type))

        self.refine_layers = nn.Sequential(*refine_layers)
        self.out_conv      = nn.Conv2d(layer_channels[-1], out_channels, kernel_size=3, stride=1, bias=True, padding=1, padding_mode='reflect')

        if out_act is not None:
            act_func      = get_activation(out_act)
            self.out_conv = nn.Sequential(self.out_conv, act_func)


    def forward(self, x):
        return self.out_conv(self.refine_layers(x)), None



if __name__ == '__main__':
    net   = Conv(3,3)
    dnet  = Conv(3, 3, stride=2)
    dconv = DConv(3, 64)
    unet  = Unet(3,3)

    test_input = torch.rand(1, 3, 256, 256)
    out        = net(test_input)
    print("Before Conv",test_input.shape)
    print("After Conv",out.shape)
    print("")

    out = dnet(test_input)
    print("Before conv stride 2",test_input.shape)
    print("After conv stride 2",out.shape)
    print("")

    out = dconv(test_input)
    print("-------------")
    print(dconv)
    print("-------------")
    print("Before DConv",test_input.shape)
    print("After DConv",out.shape)
    print("")

    import pdb; pdb.set_trace()
    out = unet(test_input)
    print("-------------")
    print(unet)
    print("-------------")
    print("Before Unet",test_input.shape)
    print("After Unet",out.shape)
    print("")

    refine_model = Refine_Net(1, [64, 64, 64], 1)
    test_input   = torch.randn(5, 1, 256, 256)
    out, vis     = refine_model(test_input)

    print("-------------")
    print(refine_model)
    print("-------------")
    print("Before refinement", test_input.shape)
    print("After Unet",out.shape)
    print("")
