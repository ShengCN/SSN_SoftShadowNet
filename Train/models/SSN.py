import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils
from torchvision.transforms import Resize
from collections import OrderedDict
import numpy as np

from .abs_model import abs_model
from .blocks import *
from .Loss.Loss import norm_loss

class SSN_Model(nn.Module):
    """ Standard Unet Implementation
        src: https://arxiv.org/pdf/1505.04597.pdf
    """
    def __init__(self, in_channels, out_channels, norm_type='Batch', mid_act='relu', out_act='relu', resnet=True):
        super(SSN_Model, self).__init__()

        self.indconv = Conv(in_channels, 64-in_channels, norm_type=norm_type, activation=mid_act)
        self.to_128 = Conv(64, 128, stride=2, norm_type=norm_type, activation=mid_act)
        self.dconv128 = DConv(128, 128, norm_type=norm_type, activation=mid_act, resnet=resnet)
        self.to_256 = Conv(128, 256, stride=2, norm_type=norm_type, activation=mid_act)
        self.dconv256 = DConv(256, 256, norm_type=norm_type, activation=mid_act, resnet=resnet)
        self.to_512 = Conv(256, 512, stride=2, norm_type=norm_type, activation=mid_act)
        self.dconv512 = DConv(512, 512, norm_type=norm_type, activation=mid_act, resnet=resnet)

        self.up512  = Up(512, 256, norm_type=norm_type, activation=mid_act, resnet=resnet)
        self.up256  = Up(256, 128, norm_type=norm_type, activation=mid_act, resnet=resnet)
        self.up128  = Up(128, 64, norm_type=norm_type, activation=mid_act, resnet=resnet)
        self.out    = Conv(64, out_channels, norm_type=norm_type, activation=out_act)


    def forward(self, x, ibl):
        x_in = self.indconv(x)
        x64 = torch.cat((x_in, x), dim=1)
        x128 = self.dconv128(self.to_128(x64))
        x256 = self.dconv256(self.to_256(x128))
        x512 = self.dconv512(self.to_512(x256))

        b,c,h,w = x.shape

        if h == 512:
            resize_t = Resize((16, 32))
            ibl = resize_t(ibl)

        b,c,h,w = x512.shape
        ibl = ibl.view(-1, 512, 1, 1).repeat(1, 1, h, w)

        y = self.up512(ibl, x256)
        y = self.up256(y, x128)
        y = self.up128(y, x64)

        return self.out(y)



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


class SSN(abs_model):
    def __init__(self, opt):
        mid_act      = opt['model']['mid_act']
        out_act      = opt['model']['out_act']
        in_channels  = opt['model']['in_channels']
        out_channels = opt['model']['out_channels']
        self.ncols   = opt['hyper_params']['n_cols']

        self.model = SSN_Model(in_channels=in_channels, out_channels=out_channels, mid_act=mid_act, out_act=out_act)

        self.optimizer = get_optimizer(opt, self.model)
        self.visualization = {}

        self.norm_loss_ = norm_loss(norm=1)

        # block WARNING
        root_logger = logging.getLogger('param')
        root_logger.setLevel(logging.ERROR)

    def setup_input(self, x):
        return x


    def forward(self, x):
        x, ibl = x['x'], x['ibl']
        return self.model(x, ibl)


    def compute_loss(self, y, pred):
        total_loss = self.norm_loss_.loss(y, pred)
        return total_loss


    def supervise(self, input_x, y, is_training:bool)->float:
        optimizer = self.optimizer
        model = self.model

        optimizer.zero_grad()
        pred = self.forward(input_x)
        loss = self.compute_loss(y, pred)

        if is_training:
            loss.backward()
            optimizer.step()

        self.visualization['y']    = y.detach()
        self.visualization['pred'] = pred.detach()

        return loss.item()


    def get_visualize(self) -> OrderedDict:
        """ Convert to visualization numpy array
        """
        nrows          = self.ncols
        visualizations = self.visualization
        ret_vis        = OrderedDict()

        for k, v in visualizations.items():
            batch = v.shape[0]
            n     = min(nrows, batch)

            plot_v = v[:n]
            ret_vis[k] = np.clip(utils.make_grid(plot_v.cpu(), nrow=nrows).numpy().transpose(1,2,0), 0.0, 1.0)

        return ret_vis


    def get_logs(self):
        pass


    def inference(self, x):
        pass

    def batch_inference(self, x):
        # TODO
        pass


    """ Getter & Setter
    """
    def get_models(self) -> dict:
        return {'model': self.model}


    def get_optimizers(self) -> dict:
        return {'optimizer': self.optimizer}


    def set_models(self, models: dict) :
        # input test
        if 'model' not in models.keys():
            raise ValueError('{} not in self.model'.format('model'))

        self.model = models['model']


    def set_optimizers(self, optimizer: dict):
        self.optimizer = optimizer['optimizer']


    ####################
    # Personal Methods #
    ####################
