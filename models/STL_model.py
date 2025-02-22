import numpy as np

import os
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
from sklearn.linear_model import LogisticRegression
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
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


def create_optimizer(opt_type, params, lr):
    """
    Creates the desired optimizer.
    """
    if opt_type.upper() == 'ADAM':
        opt = optim.Adam(params, lr=lr)
    elif opt_type.upper() == 'SGD':
        opt = optim.SGD(params, lr=lr)
    return opt


def post_proc_losses(task_losses, task_groups):
    out_losses = []
    loss_types = []
    for group in task_groups:
        if not group['type'] in loss_types:
            loss_types.append(group['type'])
            type_losses = [task_losses[k] for k in range(len(task_groups)) if task_groups[k]['type'] == group['type']]
            out_losses.append(sum(type_losses))
    return out_losses

class UNet(nn.Module):
    def __init__(self,
                 task_groups,
                 opt,
                 partitioning=None,
                 bilinear=True):
        super(UNet, self).__init__()
        self.task_groups = task_groups
        self.n_tasks = len(task_groups)
        self.bilinear = bilinear
        self.n_iter = 0
        self.n_epoch = 0
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
        self.outcs = nn.ModuleList([OutConv(64, 1), OutConv(64, 1), OutConv(64, 1)])
        self.outfc = OutFC(1024 // factor, 1)
        self.outfc_ppa = OutFC(1024 // factor, 1)
        self.clf = LogisticRegression(random_state=0, solver='lbfgs')

    def train_clf(self, vCDRs, classif_labs):
        self.clf.fit(vCDRs, classif_labs)

    def forward(self, x, task=None):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        if task == 0:
            return torch.sigmoid(self.outfc(x5))
        elif task == 4:
            return torch.sigmoid(self.outfc_ppa(x5))
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        if task:
            if task == 1 or task == 2 or task == 3:
                seg_logits = torch.sigmoid(self.outcs[task - 1](x))
                return seg_logits
        else:
            return [torch.sigmoid(self.outfc(x5)), torch.sigmoid(self.outcs[0](x)), torch.sigmoid(self.outcs[1](x)),
                    torch.sigmoid(self.outcs[2](x)), torch.sigmoid(self.outfc_ppa(x5))]

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


class STL_model(UNet):
    def __init__(self, 
                 task_groups,
                 opt,
                 bilinear=True):
        super(STL_model, self).__init__(task_groups, opt, None, bilinear)
        self.per_batch_step = opt.per_batch_step
        self.one_optim_per_task = opt.one_optim_per_task
        self.learning_rate = opt.learning_rate
        self.active_task = opt.active_task
        
        # Optimizer
        self.optimizer = create_optimizer(opt.optimizer, self.parameters(), self.learning_rate)
            
    
    def optim_step(self, task=None):
        self.optimizer.step()
            
    def optim_zero_grad(self, task=None):
        self.optimizer.zero_grad()
            
    def train_step(self, 
                   data, 
                   gts, 
                   loss_func):
        # Forward pass
        logits = self.forward(data)
        task_losses = loss_func(logits, gts)
        loss = task_losses[self.active_task]
        task_losses = task_losses.detach()

        # Backward pass
        self.optim_zero_grad()
        loss.backward()
        self.optim_step()

        # Incr iter nb
        self.n_iter += 1
        preds = [elt.detach() for elt in logits]
            
        return task_losses, preds
    

    def test_step(self, data, gts, loss_func):
        # Forward pass
        logits = self.forward(data)
        task_losses = loss_func(logits, gts)
        loss = task_losses[self.active_task]
        
        return task_losses, logits

    
    def initialize(self, 
                   opt, 
                   device, 
                   model_dir, 
                   saver):
        super(STL_model, self).initialize(opt, 
                                          device, 
                                          model_dir)
        # Nothing if no load required
        if opt.recover:
            source_dir = os.path.join(opt.checkpoint_path, opt.reco_name) if opt.reco_name else model_dir
            ckpt_file = os.path.join(source_dir, opt.reco_type+'_weights.pth')
            ckpt = torch.load(ckpt_file, map_location=device)
            # Recovers optimizers if existing
            if 'optimizer_state_dict' in ckpt:
                if self.one_optim_per_task:
                    for k in range(len(ckpt['optimizer_state_dict'])):
                        self.optimizer[k].load_state_dict(ckpt['optimizer_state_dict'][str(k)])
                        print('Optimizer {} recovered from {}.'.format(k, ckpt_file))
                else:
                    self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                    print('Optimizer recovered from {}.'.format(ckpt_file))
            # Recovers saver
            saver.load()
