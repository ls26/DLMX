import os
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
from sklearn.linear_model import LogisticRegression
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch

from models.sync_batchnorm.batchnorm import BatchNorm2d
from models.aspp import build_aspp
from models.decoder import build_decoder
from models.backbone import build_backbone

class ConvLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 padding=1,
                 norm_layer=True,
                 relu=True,
                 n_tasks=None):

        super(ConvLayer, self).__init__()
        modules = [('CONV', nn.Conv2d(in_channels,
                                      out_channels,
                                      kernel_size=kernel_size,
                                      padding=padding))]
        if norm_layer:
            modules.append(('BN', nn.BatchNorm2d(num_features=out_channels,
                                                 track_running_stats=False)))
        if relu:
            modules.append(('relu', nn.ReLU(inplace=True)))

        self.conv_block = nn.Sequential(OrderedDict(modules))

    def forward(self, x):
        return self.conv_block(x)

    def get_weight(self):
        return self.conv_block[0].weight

    def get_routing_block(self):
        return self.conv_block[-2]

    def get_routing_masks(self):
        mapping = self.conv_block[-2].unit_mapping.detach().cpu().numpy()
        tested = self.conv_block[-2].tested_tasks.detach().cpu().numpy()
        return mapping, tested


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channels=None,
                 n_tasks=None,
                 ):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            ConvLayer(in_channels,
                      mid_channels,
                      n_tasks=n_tasks),
            ConvLayer(mid_channels,
                      out_channels,
                      n_tasks=n_tasks),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self,
                 in_channels,
                 out_channels,
                 n_tasks=None,
                 ):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels,
                       out_channels,
                       n_tasks=n_tasks,
                       )
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self,
                 in_channels,
                 out_channels,
                 bilinear=True,
                 n_tasks=None,
                 ):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels,
                                   out_channels,
                                   mid_channels=in_channels // 2,
                                   n_tasks=n_tasks,
                                   )
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels,
                                   out_channels,
                                   n_tasks=n_tasks,
                                   )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class OutFC(nn.Module):
    def __init__(self, in_channels, out_channels, p=0.8):
        super(OutFC, self).__init__()
        self.pool = nn.AdaptiveMaxPool2d(1)
        self.dropout = nn.Dropout(p=p)
        self.fc = nn.Linear(in_channels, out_channels, bias=True)

    def forward(self, x):
        flat_pool = self.pool(x).squeeze(2).squeeze(2)
        return self.fc(self.dropout(flat_pool)).squeeze(1)


def get_blocks(model, inst_type):
    """
    Returns all instance of the requested type in a model.
    """
    if isinstance(model, inst_type):
        blocks = [model]
    else:
        blocks = []
        for child in model.children():
            blocks += get_blocks(child, inst_type)
    return blocks


def post_proc_losses(task_losses, task_groups):
    out_losses = []
    loss_types = []
    for group in task_groups:
        if not group['type'] in loss_types:
            loss_types.append(group['type'])
            type_losses = [task_losses[k] for k in range(len(task_groups)) if task_groups[k]['type'] == group['type']]
            out_losses.append(sum(type_losses))
    return out_losses
class MTL_UNet_v2(nn.Module):
    def __init__(self,
                 opt,
                 partitioning=None,
                 bilinear=True):
        super(MTL_UNet_v2, self).__init__()
        self.n_tasks = 3
        self.bilinear = bilinear

        self.size = 18

        self.inc = DoubleConv(3,
                              64,
                              n_tasks=self.n_tasks)
        self.down1 = Down(64,
                          128,
                          n_tasks=self.n_tasks)
        self.down2 = Down(128,
                          256,
                          n_tasks=self.n_tasks)
        self.down3 = Down(256,
                          512,
                          n_tasks=self.n_tasks)
        factor = 2 if bilinear else 1
        self.down4 = Down(512,
                          1024 // factor,
                          n_tasks=self.n_tasks)
        self.up1 = Up(1024,
                      512 // factor,
                      n_tasks=self.n_tasks,
                      bilinear=bilinear)
        self.up2 = Up(512,
                      256 // factor,
                      n_tasks=self.n_tasks,
                      bilinear=bilinear)
        self.up3 = Up(256,
                      128 // factor,
                      n_tasks=self.n_tasks,
                      bilinear=bilinear)
        self.up4 = Up(128,
                      64,
                      n_tasks=self.n_tasks,
                      bilinear=bilinear)
        self.outcs = nn.ModuleList([OutConv(64, 1), OutConv(64, 6)])
        self.outfc = OutFC(1024 // factor, 2)

    def forward(self, x, task=None):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        if task == 0:
            return self.outfc(x5)
        elif task == 1:
            return torch.sigmoid(self.outcs[0](x))
        elif task == 2:
            return self.outcs[1](x)
        else:
            return [self.outfc(x5), torch.sigmoid(self.outcs[0](x)), self.outcs[1](x)]

    def initialize(self,
                   opt,
                   device,
                   model_dir):

        # Nothing if no load required
        if opt.recover:
            source_dir = os.path.join(opt.checkpoint_path, opt.reco_name) if opt.reco_name else model_dir
            ckpt_file = os.path.join(source_dir, opt.reco_type + '_weights.pth')
            ckpt = torch.load(ckpt_file, map_location=device)

            # Gets what needed in the checkpoint
            pretrained_dict = {k: v for k, v in ckpt['model_state_dict'].items() if
                               'CONV' in k or 'BN' in k or 'FC' in k or 'outcs' in k or 'outfc' in k}

            # Loads the weights
            self.load_state_dict(pretrained_dict, strict=False)
            self.clf = ckpt['classifier']
            print('Weights and classifier recovered from {}.'.format(ckpt_file))

            # For recovering only
            if source_dir == model_dir:
                self.n_epoch = ckpt['epoch']
                self.n_iter = ckpt['n_iter'] + 1

        elif opt.pretrained:
            model_dict = {k: v for k, v in self.state_dict().items() if
                          (not 'up' in k and not 'out' in k and not 'num_batches_tracked' in k)}
            url = 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth'
            pretrained_dict = model_zoo.load_url(url)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                               (not 'classifier' in k and not 'running' in k)}
            new_pretrained_dict = {}
            pretrained_list = [(k, v) for k, v in pretrained_dict.items()]
            model_list = [(k, v) for k, v in model_dict.items()]
            for k in range(len(pretrained_dict.items())):
                kp, vp = pretrained_list[k]
                km, vm = model_list[k]
                if vp.shape == vm.shape:
                    new_pretrained_dict[km] = vp
                else:
                    new_pretrained_dict[km] = vm
            self.load_state_dict(new_pretrained_dict, strict=False)

    def checkpoint(self):
        # Prepares checkpoint
        ckpt = {'model_state_dict': self.state_dict(),
                'classifier': self.clf,
                'epoch': self.n_epoch,
                'n_iter': self.n_iter}
        return ckpt

class MTL_UNet(nn.Module):
    def __init__(self,
                 opt,
                 partitioning=None,
                 bilinear=True):
        super(MTL_UNet, self).__init__()
        self.n_tasks = 3
        self.bilinear = bilinear

        self.size = 18

        self.inc = DoubleConv(3,
                              64,
                              n_tasks=self.n_tasks)
        self.down1 = Down(64,
                          128,
                          n_tasks=self.n_tasks)
        self.down2 = Down(128,
                          256,
                          n_tasks=self.n_tasks)
        self.down3 = Down(256,
                          512,
                          n_tasks=self.n_tasks)
        factor = 2 if bilinear else 1
        self.down4 = Down(512,
                          1024 // factor,
                          n_tasks=self.n_tasks)
        self.up1 = Up(1024,
                      512 // factor,
                      n_tasks=self.n_tasks,
                      bilinear=bilinear)
        self.up2 = Up(512,
                      256 // factor,
                      n_tasks=self.n_tasks,
                      bilinear=bilinear)
        self.up3 = Up(256,
                      128 // factor,
                      n_tasks=self.n_tasks,
                      bilinear=bilinear)
        self.up4 = Up(128,
                      64,
                      n_tasks=self.n_tasks,
                      bilinear=bilinear)
        self.outcs = nn.ModuleList([OutConv(64, 1), OutConv(64, 1), OutConv(64, 1),OutConv(64, 1), OutConv(64, 1), OutConv(64, 1)])
        self.outfc = OutFC(1024 // factor, 2)

    def forward(self, x, task=None):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        if task == 0:
            return self.outfc(x5)
        elif task == 1:
            return torch.sigmoid(self.outcs[0](x))
        elif task == 2:
            return [torch.sigmoid(self.outcs[1](x)),torch.sigmoid(self.outcs[2](x)), torch.sigmoid(self.outcs[3](x)), torch.sigmoid(self.outcs[4](x)),
                torch.sigmoid(self.outcs[5](x))]
        else:
            return [self.outfc(x5), torch.sigmoid(self.outcs[0](x)), torch.sigmoid(self.outcs[1](x)),
                    torch.sigmoid(self.outcs[2](x)), torch.sigmoid(self.outcs[3](x)), torch.sigmoid(self.outcs[4](x)),
                    torch.sigmoid(self.outcs[5](x))]

    def initialize(self,
                   opt,
                   device,
                   model_dir):

        # Nothing if no load required
        if opt.recover:
            source_dir = os.path.join(opt.checkpoint_path, opt.reco_name) if opt.reco_name else model_dir
            ckpt_file = os.path.join(source_dir, opt.reco_type + '_weights.pth')
            ckpt = torch.load(ckpt_file, map_location=device)

            # Gets what needed in the checkpoint
            pretrained_dict = {k: v for k, v in ckpt['model_state_dict'].items() if
                               'CONV' in k or 'BN' in k or 'FC' in k or 'outcs' in k or 'outfc' in k}

            # Loads the weights
            self.load_state_dict(pretrained_dict, strict=False)
            self.clf = ckpt['classifier']
            print('Weights and classifier recovered from {}.'.format(ckpt_file))

            # For recovering only
            if source_dir == model_dir:
                self.n_epoch = ckpt['epoch']
                self.n_iter = ckpt['n_iter'] + 1

        elif opt.pretrained:
            model_dict = {k: v for k, v in self.state_dict().items() if
                          (not 'up' in k and not 'out' in k and not 'num_batches_tracked' in k)}
            url = 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth'
            pretrained_dict = model_zoo.load_url(url)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                               (not 'classifier' in k and not 'running' in k)}
            new_pretrained_dict = {}
            pretrained_list = [(k, v) for k, v in pretrained_dict.items()]
            model_list = [(k, v) for k, v in model_dict.items()]
            for k in range(len(pretrained_dict.items())):
                kp, vp = pretrained_list[k]
                km, vm = model_list[k]
                if vp.shape == vm.shape:
                    new_pretrained_dict[km] = vp
                else:
                    new_pretrained_dict[km] = vm
            self.load_state_dict(new_pretrained_dict, strict=False)

    def checkpoint(self):
        # Prepares checkpoint
        ckpt = {'model_state_dict': self.state_dict(),
                'classifier': self.clf,
                'epoch': self.n_epoch,
                'n_iter': self.n_iter}
        return ckpt

from collections import OrderedDict
from .layers import *
from .efficientnet import EfficientNet


__all__ = ['EfficientUnet', 'get_efficientunet_b0', 'get_efficientunet_b1', 'get_efficientunet_b2',
           'get_efficientunet_b3', 'get_efficientunet_b4', 'get_efficientunet_b5', 'get_efficientunet_b6',
           'get_efficientunet_b7']


def get_blocks_to_be_concat(model, x):
    shapes = set()
    blocks = OrderedDict()
    hooks = []
    count = 0

    def register_hook(module):

        def hook(module, input, output):
            try:
                nonlocal count
                if module.name == f'blocks_{count}_output_batch_norm':
                    count += 1
                    shape = output.size()[-2:]
                    if shape not in shapes:
                        shapes.add(shape)
                        blocks[module.name] = output

                elif module.name == 'head_swish':
                    # when module.name == 'head_swish', it means the program has already got all necessary blocks for
                    # concatenation. In my dynamic unet implementation, I first upscale the output of the backbone,
                    # (in this case it's the output of 'head_swish') concatenate it with a block which has the same
                    # Height & Width (image size). Therefore, after upscaling, the output of 'head_swish' has bigger
                    # image size. The last block has the same image size as 'head_swish' before upscaling. So we don't
                    # really need the last block for concatenation. That's why I wrote `blocks.popitem()`.
                    blocks.popitem()
                    blocks[module.name] = output

            except AttributeError:
                pass

        if (
                not isinstance(module, nn.Sequential)
                and not isinstance(module, nn.ModuleList)
                and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))

    # register hook
    model.apply(register_hook)

    # make a forward pass to trigger the hooks
    model(x)

    # remove these hooks
    for h in hooks:
        h.remove()

    return blocks
class EfficientUnet_seg(nn.Module):
    def __init__(self, encoder, out_channels=2, concat_input=True):
        super().__init__()

        self.encoder = encoder
        self.concat_input = concat_input

        self.up_conv1 = up_conv(self.n_channels, 512)
        self.double_conv1 = double_conv(self.size[0], 512)
        self.up_conv2 = up_conv(512, 256)
        self.double_conv2 = double_conv(self.size[1], 256)
        self.up_conv3 = up_conv(256, 128)
        self.double_conv3 = double_conv(self.size[2], 128)
        self.up_conv4 = up_conv(128, 64)
        self.double_conv4 = double_conv(self.size[3], 64)

        if self.concat_input:
            self.up_conv_input = up_conv(64, 32)
            self.double_conv_input = double_conv(self.size[4], 32)
        # self.final_conv = nn.ModuleList([nn.Conv2d(self.size[5], 1, kernel_size=1),nn.Conv2d(self.size[5], 6, kernel_size=1)])
        self.final_conv = nn.Sequential(nn.Dropout(p=0.95), nn.Conv2d(self.size[5], 6, kernel_size=1))
        # self.final_conv = nn.Conv2d(self.size[5], out_channels, kernel_size=1)

    @property
    def n_channels(self):
        n_channels_dict = {'efficientnet-b0': 1280, 'efficientnet-b1': 1280, 'efficientnet-b2': 1408,
                           'efficientnet-b3': 1536, 'efficientnet-b4': 1792, 'efficientnet-b5': 2048,
                           'efficientnet-b6': 2304, 'efficientnet-b7': 2560}
        return n_channels_dict[self.encoder.name]

    @property
    def size(self):
        size_dict = {'efficientnet-b0': [592, 296, 152, 80, 35, 32], 'efficientnet-b1': [592, 296, 152, 80, 35, 32],
                     'efficientnet-b2': [600, 304, 152, 80, 35, 32], 'efficientnet-b3': [608, 304, 160, 88, 35, 32],
                     'efficientnet-b4': [624, 312, 160, 88, 35, 32], 'efficientnet-b5': [640, 320, 168, 88, 35, 32],
                     'efficientnet-b6': [656, 328, 168, 96, 35, 32], 'efficientnet-b7': [672, 336, 176, 96, 35, 32]}
        return size_dict[self.encoder.name]

    def forward(self, x, task=None):
        input_ = x

        blocks = get_blocks_to_be_concat(self.encoder, x)
        _, x_previous = blocks.popitem()
        # print(x_previous.shape)

        x = self.up_conv1(x_previous)
        x = torch.cat([x, blocks.popitem()[1]], dim=1)
        x = self.double_conv1(x)

        x = self.up_conv2(x)
        x = torch.cat([x, blocks.popitem()[1]], dim=1)
        x = self.double_conv2(x)

        x = self.up_conv3(x)
        x = torch.cat([x, blocks.popitem()[1]], dim=1)
        x = self.double_conv3(x)

        x = self.up_conv4(x)
        x = torch.cat([x, blocks.popitem()[1]], dim=1)
        x = self.double_conv4(x)

        if self.concat_input:
            x = self.up_conv_input(x)
            x = torch.cat([x, input_], dim=1)
            x = self.double_conv_input(x)

        # x = self.final_conv(x)
  
        # if task == 1:
        #     return torch.sigmoid(self.final_conv[0](x))
        # elif task == 2:
        return self.final_conv(x)

class EfficientUnet_cls(nn.Module):
    def __init__(self, encoder, out_channels=2, concat_input=False):
        super().__init__()

        self.encoder = encoder
        self.concat_input = concat_input

        self.up_conv1 = up_conv(self.n_channels, 512)
        self.double_conv1 = double_conv(self.size[0], 512)
        self.up_conv2 = up_conv(512, 256)
        self.double_conv2 = double_conv(self.size[1], 256)
        self.up_conv3 = up_conv(256, 128)
        self.double_conv3 = double_conv(self.size[2], 128)
        self.up_conv4 = up_conv(128, 64)
        self.double_conv4 = double_conv(self.size[3], 64)

        if self.concat_input:
            self.up_conv_input = up_conv(64, 32)
            self.double_conv_input = double_conv(self.size[4], 32)
        self.outfc = nn.Sequential(nn.Dropout(p=0.997), OutFC(self.n_channels, 2))
        

    @property
    def n_channels(self):
        n_channels_dict = {'efficientnet-b0': 1280, 'efficientnet-b1': 1280, 'efficientnet-b2': 1408,
                           'efficientnet-b3': 1536, 'efficientnet-b4': 1792, 'efficientnet-b5': 2048,
                           'efficientnet-b6': 2304, 'efficientnet-b7': 2560}
        return n_channels_dict[self.encoder.name]

    @property
    def size(self):
        size_dict = {'efficientnet-b0': [592, 296, 152, 80, 35, 32], 'efficientnet-b1': [592, 296, 152, 80, 35, 32],
                     'efficientnet-b2': [600, 304, 152, 80, 35, 32], 'efficientnet-b3': [608, 304, 160, 88, 35, 32],
                     'efficientnet-b4': [624, 312, 160, 88, 35, 32], 'efficientnet-b5': [640, 320, 168, 88, 35, 32],
                     'efficientnet-b6': [656, 328, 168, 96, 35, 32], 'efficientnet-b7': [672, 336, 176, 96, 35, 32]}
        return size_dict[self.encoder.name]

    def forward(self, x, task=None):
        input_ = x

        blocks = get_blocks_to_be_concat(self.encoder, x)
        _, x_previous = blocks.popitem()
        # print(x_previous.shape)

        x = self.up_conv1(x_previous)
        x = torch.cat([x, blocks.popitem()[1]], dim=1)
        x = self.double_conv1(x)

        x = self.up_conv2(x)
        x = torch.cat([x, blocks.popitem()[1]], dim=1)
        x = self.double_conv2(x)

        x = self.up_conv3(x)
        x = torch.cat([x, blocks.popitem()[1]], dim=1)
        x = self.double_conv3(x)

        x = self.up_conv4(x)
        x = torch.cat([x, blocks.popitem()[1]], dim=1)
        x = self.double_conv4(x)

        if self.concat_input:
            x = self.up_conv_input(x)
            x = torch.cat([x, input_], dim=1)
            x = self.double_conv_input(x)

        # x = self.final_conv(x)

        return self.outfc(x_previous)

class EfficientUnet(nn.Module):
    def __init__(self, encoder, out_channels=2, concat_input=True):
        super().__init__()

        self.encoder = encoder
        self.concat_input = concat_input

        self.up_conv1 = up_conv(self.n_channels, 512)
        self.double_conv1 = double_conv(self.size[0], 512)
        self.up_conv2 = up_conv(512, 256)
        self.double_conv2 = double_conv(self.size[1], 256)
        self.up_conv3 = up_conv(256, 128)
        self.double_conv3 = double_conv(self.size[2], 128)
        self.up_conv4 = up_conv(128, 64)
        self.double_conv4 = double_conv(self.size[3], 64)

        if self.concat_input:
            self.up_conv_input = up_conv(64, 32)
            self.double_conv_input = double_conv(self.size[4], 32)
        self.final_conv = nn.ModuleList([nn.Conv2d(self.size[5], 1, kernel_size=1),nn.Conv2d(self.size[5], 6, kernel_size=1)])
        self.outfc = OutFC(self.n_channels, 2)
        # self.final_conv = nn.Conv2d(self.size[5], out_channels, kernel_size=1)

    @property
    def n_channels(self):
        n_channels_dict = {'efficientnet-b0': 1280, 'efficientnet-b1': 1280, 'efficientnet-b2': 1408,
                           'efficientnet-b3': 1536, 'efficientnet-b4': 1792, 'efficientnet-b5': 2048,
                           'efficientnet-b6': 2304, 'efficientnet-b7': 2560}
        return n_channels_dict[self.encoder.name]

    @property
    def size(self):
        size_dict = {'efficientnet-b0': [592, 296, 152, 80, 35, 32], 'efficientnet-b1': [592, 296, 152, 80, 35, 32],
                     'efficientnet-b2': [600, 304, 152, 80, 35, 32], 'efficientnet-b3': [608, 304, 160, 88, 35, 32],
                     'efficientnet-b4': [624, 312, 160, 88, 35, 32], 'efficientnet-b5': [640, 320, 168, 88, 35, 32],
                     'efficientnet-b6': [656, 328, 168, 96, 35, 32], 'efficientnet-b7': [672, 336, 176, 96, 35, 32]}
        return size_dict[self.encoder.name]

    def forward(self, x, task=None):
        input_ = x

        blocks = get_blocks_to_be_concat(self.encoder, x)
        _, x_previous = blocks.popitem()
        # print(x_previous.shape)

        x = self.up_conv1(x_previous)
        x = torch.cat([x, blocks.popitem()[1]], dim=1)
        x = self.double_conv1(x)

        x = self.up_conv2(x)
        x = torch.cat([x, blocks.popitem()[1]], dim=1)
        x = self.double_conv2(x)

        x = self.up_conv3(x)
        x = torch.cat([x, blocks.popitem()[1]], dim=1)
        x = self.double_conv3(x)

        x = self.up_conv4(x)
        x = torch.cat([x, blocks.popitem()[1]], dim=1)
        x = self.double_conv4(x)

        if self.concat_input:
            x = self.up_conv_input(x)
            x = torch.cat([x, input_], dim=1)
            x = self.double_conv_input(x)

        # x = self.final_conv(x)
        # if task == 0:
        #     return self.outfc(x_previous)
        # elif task == 1:
        #     return torch.sigmoid(self.final_conv[0](x))
        # elif task == 2:
        #     return self.final_conv[1](x)
        # else:
        return [self.outfc(x_previous), torch.sigmoid(self.final_conv[0](x)), self.final_conv[1](x)]
def get_efficientunet_b0_cls(out_channels=2, concat_input=True, pretrained=True):
    encoder = EfficientNet.encoder('efficientnet-b0', pretrained=pretrained)
    model = EfficientUnet_cls(encoder, out_channels=out_channels, concat_input=concat_input)
    return model


def get_efficientunet_b3_cls(out_channels=2, concat_input=True, pretrained=True):
    encoder = EfficientNet.encoder('efficientnet-b3', pretrained=pretrained)
    model = EfficientUnet_cls(encoder, out_channels=out_channels, concat_input=concat_input)
    return model
    
def get_efficientunet_b0_seg(out_channels=2, concat_input=True, pretrained=True):
    encoder = EfficientNet.encoder('efficientnet-b0', pretrained=pretrained)
    model = EfficientUnet_seg(encoder, out_channels=out_channels, concat_input=concat_input)
    return model


def get_efficientunet_b3_seg(out_channels=2, concat_input=True, pretrained=True):
    encoder = EfficientNet.encoder('efficientnet-b3', pretrained=pretrained)
    model = EfficientUnet_seg(encoder, out_channels=out_channels, concat_input=concat_input)
    return model    
    
    
def get_efficientunet_b0(out_channels=2, concat_input=True, pretrained=True):
    encoder = EfficientNet.encoder('efficientnet-b0', pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=out_channels, concat_input=concat_input)
    return model


def get_efficientunet_b1(out_channels=2, concat_input=True, pretrained=True):
    encoder = EfficientNet.encoder('efficientnet-b1', pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=out_channels, concat_input=concat_input)
    return model


def get_efficientunet_b2(out_channels=2, concat_input=True, pretrained=True):
    encoder = EfficientNet.encoder('efficientnet-b2', pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=out_channels, concat_input=concat_input)
    return model


def get_efficientunet_b3(out_channels=2, concat_input=True, pretrained=True):
    encoder = EfficientNet.encoder('efficientnet-b3', pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=out_channels, concat_input=concat_input)
    return model


def get_efficientunet_b4(out_channels=2, concat_input=True, pretrained=True):
    encoder = EfficientNet.encoder('efficientnet-b4', pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=out_channels, concat_input=concat_input)
    return model


def get_efficientunet_b5(out_channels=2, concat_input=True, pretrained=True):
    encoder = EfficientNet.encoder('efficientnet-b5', pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=out_channels, concat_input=concat_input)
    return model


def get_efficientunet_b6(out_channels=2, concat_input=True, pretrained=True):
    encoder = EfficientNet.encoder('efficientnet-b6', pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=out_channels, concat_input=concat_input)
    return model


def get_efficientunet_b7(out_channels=2, concat_input=True, pretrained=True):
    encoder = EfficientNet.encoder('efficientnet-b7', pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=out_channels, concat_input=concat_input)
    return model
class DeepLab_seg(nn.Module):
    def __init__(self, backbone='resnet', output_stride=16, num_classes_list=[1,6],
                 sync_bn=True, freeze_bn=False, method='prototype'):
        super(DeepLab_seg, self).__init__()
        if backbone == 'drn':
            output_stride = 8
        if backbone == 'resnet' or backbone == 'drn':
            low_level_inplanes = 256
        elif backbone == 'xception':
            low_level_inplanes = 128
        elif backbone == 'mobilenet':
            low_level_inplanes = 24
        else:
            raise NotImplementedError
        if sync_bn == True:
            # print("=====================================>使用batchnorm")
            # BatchNorm = SynchronizedBatchNorm2d
            BatchNorm = nn.BatchNorm2d
        else:
            # print("=====================================>使用transnorm")
            BatchNorm = BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm)
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.decoder = nn.Sequential(nn.Dropout(p=0.8), build_decoder(num_classes_list, backbone, method, BatchNorm))
        
        if backbone == 'resnet' or backbone == 'drn':
            self.outfc = OutFC(2048, 2)
        elif backbone == 'xception':
            self.outfc = OutFC(2048, 2)
        elif backbone == 'mobilenet':
            self.outfc = OutFC(320, 2)
        if freeze_bn:
            self.freeze_bn()

    def forward(self, input, task=None):
        x_previous, low_level_feat = self.backbone(input) ## torch.Size([16, 320, 16, 16]) torch.Size([16, 24, 64, 64])

        #print(x_previous.shape, low_level_feat.shape)
        x = self.aspp(x_previous) ##torch.Size([8, 1280, 32, 32])

        x_optic,x_lesion = self.decoder(x, low_level_feat) ###torch.Size([16, 1, 64, 64]) torch.Size([16, 6, 64, 64])

        x_optic = F.interpolate(x_optic, size=input.size()[2:], mode='bilinear', align_corners=True)
        x_lesion = F.interpolate(x_lesion, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x_lesion

class DeepLab_cls(nn.Module):
    def __init__(self, backbone='resnet', output_stride=16, num_classes_list=[1,6],
                 sync_bn=True, freeze_bn=False, method='prototype'):
        super(DeepLab_cls, self).__init__()
        if backbone == 'drn':
            output_stride = 8
        if backbone == 'resnet' or backbone == 'drn':
            low_level_inplanes = 256
        elif backbone == 'xception':
            low_level_inplanes = 128
        elif backbone == 'mobilenet':
            low_level_inplanes = 24
        else:
            raise NotImplementedError
        if sync_bn == True:
            # print("=====================================>使用batchnorm")
            # BatchNorm = SynchronizedBatchNorm2d
            BatchNorm = nn.BatchNorm2d
        else:
            # print("=====================================>使用transnorm")
            BatchNorm = BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm)
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.decoder = build_decoder(num_classes_list, backbone, method, BatchNorm)
        
        if backbone == 'resnet' or backbone == 'drn':
            self.outfc = nn.Sequential(nn.Dropout(p=0.9999), OutFC(2048, 2))
        elif backbone == 'xception':
            self.outfc = nn.Sequential(nn.Dropout(p=0.9999), OutFC(2048, 2))
        elif backbone == 'mobilenet':
            self.outfc = nn.Sequential(nn.Dropout(p=0.9999), OutFC(320, 2))
        if freeze_bn:
            self.freeze_bn()

    def forward(self, input, task=None):
        x_previous, low_level_feat = self.backbone(input) ## torch.Size([16, 320, 16, 16]) torch.Size([16, 24, 64, 64])

        #print(x_previous.shape, low_level_feat.shape)
        x = self.aspp(x_previous) ##torch.Size([8, 1280, 32, 32])

        x_optic,x_lesion = self.decoder(x, low_level_feat) ###torch.Size([16, 1, 64, 64]) torch.Size([16, 6, 64, 64])

        x_optic = F.interpolate(x_optic, size=input.size()[2:], mode='bilinear', align_corners=True)
        x_lesion = F.interpolate(x_lesion, size=input.size()[2:], mode='bilinear', align_corners=True)


        return self.outfc(x_previous)
       

class DeepLab(nn.Module):
    def __init__(self, backbone='resnet', output_stride=16, num_classes_list=[1,6],
                 sync_bn=True, freeze_bn=False, method='prototype'):
        super(DeepLab, self).__init__()
        if backbone == 'drn':
            output_stride = 8
        if backbone == 'resnet' or backbone == 'drn':
            low_level_inplanes = 256
        elif backbone == 'xception':
            low_level_inplanes = 128
        elif backbone == 'mobilenet':
            low_level_inplanes = 24
        else:
            raise NotImplementedError
        if sync_bn == True:
            # print("=====================================>使用batchnorm")
            # BatchNorm = SynchronizedBatchNorm2d
            BatchNorm = nn.BatchNorm2d
        else:
            # print("=====================================>使用transnorm")
            BatchNorm = BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm)
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.decoder = build_decoder(num_classes_list, backbone, method, BatchNorm)
        
        if backbone == 'resnet' or backbone == 'drn':
            self.outfc = OutFC(2048, 2)
        elif backbone == 'xception':
            self.outfc = OutFC(2048, 2)
        elif backbone == 'mobilenet':
            self.outfc = OutFC(320, 2)
        if freeze_bn:
            self.freeze_bn()

    def forward(self, input, task=None):
        x_previous, low_level_feat = self.backbone(input) ## torch.Size([16, 320, 16, 16]) torch.Size([16, 24, 64, 64])

        #print(x_previous.shape, low_level_feat.shape)
        x = self.aspp(x_previous) ##torch.Size([8, 1280, 32, 32])

        x_optic,x_lesion = self.decoder(x, low_level_feat) ###torch.Size([16, 1, 64, 64]) torch.Size([16, 6, 64, 64])

        x_optic = F.interpolate(x_optic, size=input.size()[2:], mode='bilinear', align_corners=True)
        x_lesion = F.interpolate(x_lesion, size=input.size()[2:], mode='bilinear', align_corners=True)

        if task == 0:
            return self.outfc(x_previous)
        elif task == 1:
            return torch.sigmoid(x_optic)
        elif task == 2:
            return x_lesion
        else:
            return [self.outfc(x_previous), torch.sigmoid(x_optic), x_lesion]

    def freeze_bn(self):
        for m in self.modules():
            # if isinstance(m, SynchronizedBatchNorm2d):
            #     m.eval()
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
            elif isinstance(m, BatchNorm2d):
                m.eval()
    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) \
                        or isinstance(m[1], nn.BatchNorm2d) or isinstance(m[1], BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) \
                        or isinstance(m[1], nn.BatchNorm2d) or isinstance(m[1], BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p



class ResidualConv(nn.Module):
    def __init__(self, input_dim, output_dim, stride, padding):
        super(ResidualConv, self).__init__()

        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(input_dim),
            nn.ReLU(),
            nn.Conv2d(
                input_dim, output_dim, kernel_size=3, stride=stride, padding=padding
            ),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1),
        )
        self.conv_skip = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(output_dim),
        )

    def forward(self, x):

        return self.conv_block(x) + self.conv_skip(x)


class Upsample(nn.Module):
    def __init__(self, input_dim, output_dim, kernel, stride):
        super(Upsample, self).__init__()

        self.upsample = nn.ConvTranspose2d(
            input_dim, output_dim, kernel_size=kernel, stride=stride
        )

    def forward(self, x):
        return self.upsample(x)


class Squeeze_Excite_Block(nn.Module):
    def __init__(self, channel, reduction=16):
        super(Squeeze_Excite_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ASPP(nn.Module):
    def __init__(self, in_dims, out_dims, rate=[6, 12, 18]):
        super(ASPP, self).__init__()

        self.aspp_block1 = nn.Sequential(
            nn.Conv2d(
                in_dims, out_dims, 3, stride=1, padding=rate[0], dilation=rate[0]
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_dims),
        )
        self.aspp_block2 = nn.Sequential(
            nn.Conv2d(
                in_dims, out_dims, 3, stride=1, padding=rate[1], dilation=rate[1]
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_dims),
        )
        self.aspp_block3 = nn.Sequential(
            nn.Conv2d(
                in_dims, out_dims, 3, stride=1, padding=rate[2], dilation=rate[2]
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_dims),
        )

        self.output = nn.Conv2d(len(rate) * out_dims, out_dims, 1)
        self._init_weights()

    def forward(self, x):
        x1 = self.aspp_block1(x)
        x2 = self.aspp_block2(x)
        x3 = self.aspp_block3(x)
        out = torch.cat([x1, x2, x3], dim=1)
        return self.output(out)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class Upsample_(nn.Module):
    def __init__(self, scale=2):
        super(Upsample_, self).__init__()

        self.upsample = nn.Upsample(mode="bilinear", scale_factor=scale)

    def forward(self, x):
        return self.upsample(x)


class AttentionBlock(nn.Module):
    def __init__(self, input_encoder, input_decoder, output_dim):
        super(AttentionBlock, self).__init__()

        self.conv_encoder = nn.Sequential(
            nn.BatchNorm2d(input_encoder),
            nn.ReLU(),
            nn.Conv2d(input_encoder, output_dim, 3, padding=1),
            nn.MaxPool2d(2, 2),
        )

        self.conv_decoder = nn.Sequential(
            nn.BatchNorm2d(input_decoder),
            nn.ReLU(),
            nn.Conv2d(input_decoder, output_dim, 3, padding=1),
        )

        self.conv_attn = nn.Sequential(
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Conv2d(output_dim, 1, 1),
        )

    def forward(self, x1, x2):
        out = self.conv_encoder(x1) + self.conv_decoder(x2)
        out = self.conv_attn(out)
        return out * x2


class ResUnet(nn.Module):
    def __init__(self, channel, filters=[64, 128, 256, 512]):
        super(ResUnet, self).__init__()

        self.input_layer = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1)
        )

        self.residual_conv_1 = ResidualConv(filters[0], filters[1], 2, 1)
        self.residual_conv_2 = ResidualConv(filters[1], filters[2], 2, 1)

        self.bridge = ResidualConv(filters[2], filters[3], 2, 1)

        self.upsample_1 = Upsample(filters[3], filters[3], 2, 2)
        self.up_residual_conv1 = ResidualConv(filters[3] + filters[2], filters[2], 1, 1)

        self.upsample_2 = Upsample(filters[2], filters[2], 2, 2)
        self.up_residual_conv2 = ResidualConv(filters[2] + filters[1], filters[1], 1, 1)

        self.upsample_3 = Upsample(filters[1], filters[1], 2, 2)
        self.up_residual_conv3 = ResidualConv(filters[1] + filters[0], filters[0], 1, 1)

        self.final_conv = nn.ModuleList([nn.Conv2d(filters[0], 1, kernel_size=1),nn.Conv2d(filters[0], 6, kernel_size=1)])
        self.outfc = OutFC(filters[3], 2)

    def forward(self, x, task=None):
        # Encode
        x1 = self.input_layer(x) + self.input_skip(x)
        x2 = self.residual_conv_1(x1)
        x3 = self.residual_conv_2(x2)
        # Bridge
        x4 = self.bridge(x3)
        #print(x4.shape)
        # Decode
        x4 = self.upsample_1(x4)
        x5 = torch.cat([x4, x3], dim=1)

        x6 = self.up_residual_conv1(x5)

        x6 = self.upsample_2(x6)
        x7 = torch.cat([x6, x2], dim=1)

        x8 = self.up_residual_conv2(x7)

        x8 = self.upsample_3(x8)
        x9 = torch.cat([x8, x1], dim=1)

        x10 = self.up_residual_conv3(x9)

        if task == 0:
            return self.outfc(x4)
        elif task == 1:
            return torch.sigmoid(self.final_conv[0](x10))
        elif task == 2:
            return self.final_conv[1](x10)
        else:
            return [self.outfc(x4), torch.sigmoid(self.final_conv[0](x10)), self.final_conv[1](x10)]
        


class ResUnetPlusPlus(nn.Module):
    def __init__(self, channel, filters=[32, 64, 128, 256, 512]):
        super(ResUnetPlusPlus, self).__init__()

        self.input_layer = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1)
        )

        self.squeeze_excite1 = Squeeze_Excite_Block(filters[0])

        self.residual_conv1 = ResidualConv(filters[0], filters[1], 2, 1)

        self.squeeze_excite2 = Squeeze_Excite_Block(filters[1])

        self.residual_conv2 = ResidualConv(filters[1], filters[2], 2, 1)

        self.squeeze_excite3 = Squeeze_Excite_Block(filters[2])

        self.residual_conv3 = ResidualConv(filters[2], filters[3], 2, 1)

        self.aspp_bridge = ASPP(filters[3], filters[4])

        self.attn1 = AttentionBlock(filters[2], filters[4], filters[4])
        self.upsample1 = Upsample_(2)
        self.up_residual_conv1 = ResidualConv(filters[4] + filters[2], filters[3], 1, 1)

        self.attn2 = AttentionBlock(filters[1], filters[3], filters[3])
        self.upsample2 = Upsample_(2)
        self.up_residual_conv2 = ResidualConv(filters[3] + filters[1], filters[2], 1, 1)

        self.attn3 = AttentionBlock(filters[0], filters[2], filters[2])
        self.upsample3 = Upsample_(2)
        self.up_residual_conv3 = ResidualConv(filters[2] + filters[0], filters[1], 1, 1)

        self.aspp_out = ASPP(filters[1], filters[0])

        self.final_conv = nn.ModuleList([nn.Conv2d(filters[0], 1, kernel_size=1),nn.Conv2d(filters[0], 6, kernel_size=1)])
        self.outfc = OutFC(filters[3], 2)

    def forward(self, x, task=None):
        x1 = self.input_layer(x) + self.input_skip(x)

        x2 = self.squeeze_excite1(x1)
        x2 = self.residual_conv1(x2)

        x3 = self.squeeze_excite2(x2)
        x3 = self.residual_conv2(x3)

        x4 = self.squeeze_excite3(x3)
        x4 = self.residual_conv3(x4)

        x5 = self.aspp_bridge(x4)

        x6 = self.attn1(x3, x5)
        x6 = self.upsample1(x6)
        x6 = torch.cat([x6, x3], dim=1)
        x6 = self.up_residual_conv1(x6)

        x7 = self.attn2(x2, x6)
        x7 = self.upsample2(x7)
        x7 = torch.cat([x7, x2], dim=1)
        x7 = self.up_residual_conv2(x7)

        x8 = self.attn3(x1, x7)
        x8 = self.upsample3(x8)
        x8 = torch.cat([x8, x1], dim=1)
        x8 = self.up_residual_conv3(x8)

        x9 = self.aspp_out(x8)
    
        if task == 0:
            return self.outfc(x4)
        elif task == 1:
            return torch.sigmoid(self.final_conv[0](x9))
        elif task == 2:
            return self.final_conv[1](x9)
        else:
            return [self.outfc(x4), torch.sigmoid(self.final_conv[0](x9)), self.final_conv[1](x9)]

