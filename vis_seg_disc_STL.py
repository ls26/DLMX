
from torch import nn
from medpy.metric.binary import assd,dc
from datetime import datetime
import scipy.io as scio
import os.path as osp
import torch.backends.cudnn as cudnn
import os
import cv2
from PIL import Image
from torch.nn import functional as F
import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure


def get_pascal_labels():
    """Load the mapping that associates pascal classes with label colors
    Returns:
        np.ndarray with dimensions (21, 3)
    """
    return np.asarray([[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                       [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                       [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                       [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                       [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                       [0, 64, 128]])


def encode_segmap(mask):
    """Encode segmentation label images as pascal classes
    Args:
        mask (np.ndarray): raw segmentation label image of dimension
          (M, N, 3), in which the Pascal classes are encoded as colours.
    Returns:
        (np.ndarray): class map with dimensions (M,N), where the value at
        a given location is the integer denoting the class index.
    """
    mask = mask.astype(int)
    label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
    for ii, label in enumerate(get_pascal_labels()):
        label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
    label_mask = label_mask.astype(int)
    return label_mask


def decode_seg_map_sequence(label_mask, n_classes = 6):
    rgb_mask = decode_segmap(label_mask, n_classes = n_classes)
    rgb_masks = np.array(rgb_mask)
    return rgb_masks


def decode_seg_map_sequence_disc(label_mask, plot=False):
    label_colours = get_pascal_labels()

    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()

    r[label_mask == 1] = 0
    g[label_mask == 1] = 255
    b[label_mask == 1] = 0
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    if plot:
        plt.imshow(rgb)
        plt.show()
    else:
        return rgb
    rgb_masks = np.array(rgb)
    return rgb_masks

def decode_segmap(label_mask, n_classes = 5, plot=False):
    """Decode segmentation class labels into a color image
    Args:
        label_mask (np.ndarray): an (M,N) array of integer values denoting
          the class label at each spatial location.
        plot (bool, optional): whether to show the resulting color image
          in a figure.
    Returns:
        (np.ndarray, optional): the resulting decoded color image.
    """

    label_colours = get_pascal_labels()

    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for ll in range(0, n_classes):
        r[label_mask == ll] = label_colours[ll, 0]
        g[label_mask == ll] = label_colours[ll, 1]
        b[label_mask == ll] = label_colours[ll, 2]
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    if plot:
        plt.imshow(rgb)
        plt.show()
    else:
        return rgb
def _compute_metric(pred,target,NUMCLASS):

    pred = pred.astype(int)
    target = target.astype(int)
    dice_list  = []
    pred_each_class_number = []
    true_each_class_number = []


    for c in range(1,NUMCLASS):
        y_true    = target.copy()
        test_pred = pred.copy()
        test_pred[test_pred != c] = 0
        test_pred[test_pred == c] = 1
        y_true[y_true != c] = 0
        y_true[y_true == c] = 1
        pred_each_class_number.append(np.sum(test_pred))
        true_each_class_number.append(np.sum(y_true))

    for c in range(1, NUMCLASS):
        test_pred = pred.copy()
        test_pred[test_pred != c] = 0

        test_gt = target.copy()
        test_gt[test_gt != c] = 0

        dice = dc(test_pred, test_gt)

        dice_list.append(dice)

    return  np.array(dice_list)


import torchvision.transforms as transforms
import os
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.stats import multivariate_normal
from torchvision.transforms import InterpolationMode
import numpy as np
import argparse
import utils
import copy
from dataset import AMDDataset_v2
from metrics import *
from collections import Counter
import tqdm
import random
from sklearn.metrics import precision_recall_curve, auc
from models import *
import torch.backends.cudnn as cudnn
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import torch.optim as optim
from utils import *
parser = argparse.ArgumentParser()
parser.add_argument('--active_task', default=0, type=int, help='which task to train for STL model'
                                                               ' (0: classification, 1: OD segmentation, 2: lesion segmentation)')
parser.add_argument('--checkpoint_path', default='./checkpoints/')
parser.add_argument('--recover', default=False, type=bool, help='recover from a checkpoint')
parser.add_argument('--pretrained', default=False, type=bool)

parser.add_argument('--workers', default=6, type=int)
parser.add_argument('--batch_size', default=8, type=int)

parser.add_argument('--seed', default=None, type=int)

opt = parser.parse_args()
# model = get_efficientunet_b0(out_channels=2, concat_input=True, pretrained=False).cuda()
# model = MTL_UNet_v2(opt).to(device)
# model = MTL_UNet(opt).cuda()

model = get_efficientunet_b0(out_channels=2, concat_input=True, pretrained=True).cuda()
model_dir = opt.checkpoint_path
os.mkdir(model_dir) if not os.path.isdir(model_dir) else None

pretrained_model_pth = os.path.join(model_dir,'AMD_disc_seg.pkl')
interp = nn.Upsample(size=(256, 256), mode='bilinear', align_corners=True)
save_images = True
if not osp.exists(pretrained_model_pth):
    print('')
print('Evaluating model {}'.format(pretrained_model_pth))

saved_state_dict = torch.load(pretrained_model_pth, map_location='cpu')
model.load_state_dict(saved_state_dict)
model.eval()
model.cuda()
cudnn.benchmark = True
cudnn.enabled = True

df = pd.read_excel('/mnt/sda/fengwei/AMD_code/Training400/adam_data.xlsx', index_col='ID')
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_feature_list = []
for i in range(len(train_df)):
    train_feature_list.append(train_df.iloc[i:i + 1].values.tolist()[0])
# print(len(train_feature_list[0]))
test_feature_list = []
for i in range(len(test_df)):
    test_feature_list.append(test_df.iloc[i:i + 1].values.tolist()[0])

dice_list = []

for file_one in train_feature_list:
    img_name = file_one[1]
    img_label = file_one[4]

    disc_path = file_one[5]
    img_path = file_one[6]
    drusen_path = file_one[7]
    exudate_path = file_one[8]

    hemorrhage_path = file_one[9]
    others_path = file_one[10]
    scar_path = file_one[11]
    # print(img_name,img_label,disc_path,drusen_path,exudate_path)
    save_img_path = "./save_results/STL_disc/lesion/" + "train" + "/img/"
    save_disc_gt_path = "./save_results/STL_disc/lesion/" + "train" + "/disc_gt/"
    save_disc_pred_path = "./save_results/STL_disc/lesion/" + "train" + "/disc_pred/"
    if not os.path.exists(save_img_path):
        os.makedirs(save_img_path)

    if not os.path.exists(save_disc_gt_path):
        os.makedirs(save_disc_gt_path)
    if not os.path.exists(save_disc_pred_path):
        os.makedirs(save_disc_pred_path)
    # Image
    img_input = Image.open(img_path).convert('RGB')
    # print("img_size:",np.array(img).shape)
    # w, h = img.size
    img_ori = transforms.functional.resize(img_input, (256, 256), interpolation=InterpolationMode.BILINEAR)

    disc_seg = np.array(Image.open(disc_path)).copy()
    # print("ori disc size:", np.unique(disc_seg), img_name)
    disc_seg = 255. - disc_seg
    disc_mask = Image.fromarray((disc_seg >= 250.).astype(np.float32))
    disc_mask = transforms.functional.resize(disc_mask, (256, 256),
                                             interpolation=InterpolationMode.NEAREST)
    disc_mask = np.array(disc_mask)


    save_img = True

    disc_gt_save = decode_seg_map_sequence_disc(disc_mask) * 255
    disc_gt_save = disc_gt_save*0.5 + np.array(img_ori).astype('uint8')*0.5
    disc_gt_save = Image.fromarray(np.uint8(disc_gt_save))
    if save_img:
        disc_gt_save = transforms.functional.resize(disc_gt_save, (2000,2000),
                                                 interpolation=InterpolationMode.BILINEAR)
        disc_gt_save.save(save_disc_gt_path+img_name)

    img_save = Image.fromarray(np.array(img_ori).astype('uint8'))
    if save_img:
        img_save = transforms.functional.resize(img_save, (2000,2000),
                                                 interpolation=InterpolationMode.BILINEAR)
        img_save.save(save_img_path+img_name)
    # print(np.array(img),np.array(img).shape)
    img = transforms.functional.to_tensor(np.array(img_ori))
    # print(img.shape)
    img = img.unsqueeze(0)
    with torch.no_grad():

        pred_b_main_soft = model(img.cuda(),task=1)

        disc_pred_main = (pred_b_main_soft.cpu().data.numpy() >= 0.5).astype(np.float32)
        # print(disc_pred_main.shape)
        disc_pred_trg = decode_seg_map_sequence_disc(disc_pred_main[0,0, ...].copy()) * 255

        disc_pred_trg = disc_pred_trg * 0.5 + np.array(img_ori).astype('uint8') * 0.5
        disc_pred_trg = Image.fromarray(np.uint8(disc_pred_trg))
        if save_img:
            disc_pred_trg = transforms.functional.resize(disc_pred_trg, (2000,2000),
                                                    interpolation=InterpolationMode.BILINEAR)
            disc_pred_trg.save(save_disc_pred_path + img_name)

dice_list = []
for file_one in test_feature_list:
    img_name = file_one[1]
    img_label = file_one[4]

    disc_path = file_one[5]
    img_path = file_one[6]
    drusen_path = file_one[7]
    exudate_path = file_one[8]

    hemorrhage_path = file_one[9]
    others_path = file_one[10]
    scar_path = file_one[11]
    # print(img_name,img_label,disc_path,drusen_path,exudate_path)
    save_img_path = "./save_results/STL_disc/lesion/" + "test" + "/img/"

    save_disc_gt_path = "./save_results/STL_disc/lesion/" + "test" + "/disc_gt/"
    save_disc_pred_path = "./save_results/STL_disc/lesion/" + "test" + "/disc_pred/"
    if not os.path.exists(save_img_path):
        os.makedirs(save_img_path)

    if not os.path.exists(save_disc_gt_path):
        os.makedirs(save_disc_gt_path)
    if not os.path.exists(save_disc_pred_path):
        os.makedirs(save_disc_pred_path)
    # Image
    img_input = Image.open(img_path).convert('RGB')
    # print("img_size:",np.array(img).shape)
    # w, h = img.size
    img_ori = transforms.functional.resize(img_input, (256, 256), interpolation=InterpolationMode.BILINEAR)

    disc_seg = np.array(Image.open(disc_path)).copy()
    # print("ori disc size:", np.unique(disc_seg), img_name)
    disc_seg = 255. - disc_seg
    disc_mask = Image.fromarray((disc_seg >= 250.).astype(np.float32))
    disc_mask = transforms.functional.resize(disc_mask, (256, 256),
                                             interpolation=InterpolationMode.NEAREST)
    disc_mask = np.array(disc_mask)
    save_img = True

    disc_gt_save = decode_seg_map_sequence_disc(disc_mask) * 255
    disc_gt_save = disc_gt_save*0.5 + np.array(img_ori).astype('uint8')*0.5
    disc_gt_save = Image.fromarray(np.uint8(disc_gt_save))
    if save_img:
        disc_gt_save = transforms.functional.resize(disc_gt_save, (2000,2000),
                                                     interpolation=InterpolationMode.BILINEAR)
        disc_gt_save.save(save_disc_gt_path+img_name)

    img_save = Image.fromarray(np.array(img_ori).astype('uint8'))
    if save_img:
        img_save = transforms.functional.resize(img_save, (2000,2000),
                                                     interpolation=InterpolationMode.BILINEAR)
        img_save.save(save_img_path+img_name)
    # print(np.array(img),np.array(img).shape)
    img = transforms.functional.to_tensor(np.array(img_ori))
    # print(img.shape)
    img = img.unsqueeze(0)
    with torch.no_grad():

        pred_b_main_soft = model(img.cuda(),task=1)

        disc_pred_main = (pred_b_main_soft.cpu().data.numpy() >= 0.5).astype(np.float32)
        # print(disc_pred_main.shape)
        disc_pred_trg = decode_seg_map_sequence_disc(disc_pred_main[0,0, ...].copy()) * 255

        disc_pred_trg = disc_pred_trg * 0.5 + np.array(img_ori).astype('uint8') * 0.5
        disc_pred_trg = Image.fromarray(np.uint8(disc_pred_trg))
        if save_img:
            disc_pred_trg = transforms.functional.resize(disc_pred_trg, (2000,2000),
                                                    interpolation=InterpolationMode.BILINEAR)
            disc_pred_trg.save(save_disc_pred_path + img_name)
