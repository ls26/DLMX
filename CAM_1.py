
import cv2
import torch, os
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
from PIL import Image
import torch
import glob
import sys
sys.path.append('..')
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import argparse
import torchvision.transforms.functional as TF
from models import *
# generate class activation mapping for the top1 prediction
def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (256, 256)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[class_idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam

def get_cam(net, features_blobs, img_pil, classes, root_img,img_name,output_dir):
    params = list(net.parameters())
    weight_softmax = np.squeeze(params[-2].data.cpu().numpy())

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])

    img_tensor = preprocess(img_pil)
    img_variable = Variable(img_tensor.unsqueeze(0)).cuda()
    logit = net(img_variable)

    h_x = F.softmax(logit, dim=1).data.squeeze()
    probs, idx = h_x.sort(0, True)

    # output: the prediction
    for i in range(0, 2):
        line = '{:.3f} -> {}'.format(probs[i], classes[idx[i].item()])
        print(line)

    CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0].item()])

    # render the CAM and output
    print('output CAM.jpg for the top1 prediction: %s' % classes[idx[0].item()])
    img = cv2.imread(root_img)
    height, width, _ = img.shape
    CAM = cv2.resize(CAMs[0], (width, height))
    heatmap = cv2.applyColorMap(CAM, cv2.COLORMAP_JET)
    result = heatmap * 0.3 + img * 0.5

    out_dir_id = os.path.join(output_dir, img_name)
    path_cam_img = out_dir_id+"_cam.jpg"
    path_raw_img = out_dir_id+"_raw.jpg"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    cv2.imwrite(path_cam_img, result)
    cv2.imwrite(path_raw_img, img)
parser = argparse.ArgumentParser()
parser.add_argument('--active_task', default=0, type=int, help='which task to train for STL model'
                                                               ' (0: classification, 1: OD segmentation, 2: lesion segmentation)')
parser.add_argument('--checkpoint_path', default='./checkpoints/')
parser.add_argument('--recover', default=False, type=bool, help='recover from a checkpoint')
parser.add_argument('--pretrained', default=False, type=bool)
parser.add_argument('--reco_name', default=None, type=str, help='model to recover from')
parser.add_argument('--reco_type', default='last_checkpoint', type=str,
                    help='which type of recovery (best_error or iter_XXX)')

parser.add_argument('--workers', default=6, type=int)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--optimizer', default='adam', type=str)

parser.add_argument('--per_batch_step', default=False, type=bool, help='optimize tasks altogether or one by one')
parser.add_argument('--learning_rate', default=1e-4, type=float)
parser.add_argument('--eval_interval', default=1, type=int, help='interval between two validations')
parser.add_argument('--total_epoch', default=300, type=int, help='number of training epochs to proceed')
parser.add_argument('--seed', default=None, type=int)

args = parser.parse_args()

img_size = 224
df = pd.read_excel('/mnt/workdir/fengwei/AMD_code/Annotation-DF-Training400/Training400/adam_data.xlsx',
                   index_col='ID')
output_dir = "backward_hook_cam"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
task_groups = [{'ids': 0,
                'type': 'classif',
                'size': 1}]
net = cls_model(task_groups, args).to(device)

classes = ('Non AMD', 'AMD')
# Saving settings
model_dir = args.checkpoint_path
path_net = os.path.join(model_dir, 'AMD_CLS.pkl')
assert os.path.isfile(path_net), "=> no checkpoint found at '{}'".format(path_net)
print("=> loading checkpoint '{}'".format(path_net))
checkpoint = torch.load(path_net)

net.load_state_dict(checkpoint)

net.eval()
# for id in path_img_id:
count = 0
# hook the feature extractor
features_blobs = []
final_conv = "down4"
def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())

net._modules.get(final_conv).register_forward_hook(hook_feature)

img_size = 224
df = pd.read_excel('/mnt/workdir/fengwei/AMD_code/Annotation-DF-Training400/Training400/adam_data.xlsx',
                   index_col='ID')
output_dir = "backward_hook_cam"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
task_groups = [{'ids': 0,
                'type': 'classif',
                'size': 1}]
net = cls_model(task_groups, args).to(device)
classes = {0: 'Non AMD', 1: 'AMD'}

# Saving settings
model_dir = args.checkpoint_path
path_net = os.path.join(model_dir, 'AMD_CLS.pkl')
assert os.path.isfile(path_net), "=> no checkpoint found at '{}'".format(path_net)
print("=> loading checkpoint '{}'".format(path_net))
checkpoint = torch.load(path_net)

net.load_state_dict(checkpoint)
output_dir = "backward_hook_cam"
net.eval()
# for id in path_img_id:
count = 0
for idx_ in df.index:
    path_img = df.loc[idx_, "img_path"]
    img_name = df.loc[idx_, "imgName"]

    img = Image.open(path_img)
    get_cam(net, features_blobs, img, classes, path_img,img_name,output_dir)
