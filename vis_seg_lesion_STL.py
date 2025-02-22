
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
def dice_eval(pred,label,n_class):
    '''
    pred:  b*c*h*w
    label: b*h*w
    '''
    pred     = torch.argmax(pred,dim=1)  # b*h*w
    dice     = 0
    dice_arr = []
    each_class_number = []
    pred_each_class_number = []
    eps      = 1e-7
    # print(pred,label)
    for i in range(n_class):
        A = (pred  == i)
        B = (label == i)
        each_class_number.append(torch.sum(B).cpu().data.numpy())
        pred_each_class_number.append(torch.sum(A).cpu().data.numpy())
        inse  = torch.sum(A*B).float()
        union = (torch.sum(A)+torch.sum(B)).float()
        dice  += 2*inse/(union+eps)
        dice_arr.append((2*inse/(union+eps)).cpu().data.numpy())

    return dice,dice_arr,np.hstack(each_class_number),np.hstack(pred_each_class_number)


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
parser.add_argument('--arc', default='efficient', type=str)
parser.add_argument('--seed', default=None, type=int)

opt = parser.parse_args()
if opt.arc=="efficient":
    model = get_efficientunet_b3_seg(out_channels=2, concat_input=True, pretrained=False).cuda()
elif opt.arc=="deeplab":
    model = DeepLab_seg(backbone='resnet', output_stride=16).cuda()
elif opt.arc=="vgg":
    model = get_efficientunet_b0_seg(out_channels=2, concat_input=True, pretrained=False).cuda()
# model = MTL_UNet(opt).cuda()
# model = DeepLab(backbone='mobilenet', output_stride=16).cuda()
model_dir = opt.checkpoint_path
os.mkdir(model_dir) if not os.path.isdir(model_dir) else None

pretrained_model_pth = os.path.join(model_dir,'AMD_seg_lesion_{}.pkl'.format(opt.arc))
interp = nn.Upsample(size=(256, 256), mode='bilinear', align_corners=True)


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
train_df, test_df = train_test_split(df, test_size=0.2, random_state=2024)
train_feature_list = []
for i in range(len(train_df)):
    train_feature_list.append(train_df.iloc[i:i + 1].values.tolist()[0])
# print(len(train_feature_list[0]))
test_feature_list = []
for i in range(len(test_df)):
    test_feature_list.append(test_df.iloc[i:i + 1].values.tolist()[0])

pred_img_all = []
label_img_all = []
if not osp.exists("./{}/save_results/STL/lesion/".format(opt.arc) + "train"):
    os.makedirs("./{}/save_results/STL/lesion/".format(opt.arc) + "train")

save_img = True
with open(osp.join("./{}/save_results/STL/lesion/".format(opt.arc) + "train",'train_log.csv'), 'w') as f:
    log = [ ["img_name"] +["dice drusen"]+["dice exudate"]+["dice hemorrhage"]+["dice others"]+["dice scar"]+
            ["area drusen"]+["area exudate"]+["area hemorrhage"]+["area others"]+["area scar"]
           +["area pred drusen"]+["area pred exudate"]+["area pred hemorrhage"]+["area pred others"]+["area pred scar"]]
    log = map(str, log)
    f.write(','.join(log) + '\n')

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
    save_img_path = "./{}/save_results/STL/lesion/".format(opt.arc) + "train" + "/img/"
    save_gt_path = "./{}/save_results/STL/lesion/".format(opt.arc) + "train" + "/gt/"
    save_pred_path = "./{}/save_results/STL/lesion/".format(opt.arc) + "train" + "/pred/"
    if not os.path.exists(save_img_path):
        os.makedirs(save_img_path)
    if not os.path.exists(save_gt_path):
        os.makedirs(save_gt_path)
    if not os.path.exists(save_pred_path):
        os.makedirs(save_pred_path)

    # Image
    img_input = Image.open(img_path).convert('RGB')
    # print("img_size:",np.array(img).shape)
    # w, h = img.size
    img_ori = transforms.functional.resize(img_input, (256, 256), interpolation=InterpolationMode.BILINEAR)

    if drusen_path != "none":
        drusen_seg = np.array(Image.open(drusen_path)).copy()

        drusen_seg = 255. - drusen_seg
        drusen_mask = (drusen_seg >= 250.).astype(np.float32)
        drusen_mask = Image.fromarray(drusen_mask)
        drusen_mask = transforms.functional.resize(drusen_mask, (256, 256),
                                                   interpolation=InterpolationMode.NEAREST)
        drusen_mask = np.array(drusen_mask)
        # # print("drusen size:",torch.unique(drusen_mask))
        # print(drusen_mask.shape)
    else:
        drusen_mask = np.zeros((256, 256))

    if exudate_path != "none":
        exudate_seg = np.array(Image.open(exudate_path)).copy()
        exudate_seg = 255. - exudate_seg
        exudate_mask = (exudate_seg >= 250.).astype(np.float32)
        exudate_mask = Image.fromarray(exudate_mask)
        exudate_mask = transforms.functional.resize(exudate_mask, (256, 256),
                                                    interpolation=InterpolationMode.NEAREST)
        exudate_mask = np.array(exudate_mask)
        # print(exudate_mask.shape)
    else:
        exudate_mask = np.zeros((256, 256))

    if hemorrhage_path != "none":
        hemorrhage_seg = np.array(Image.open(hemorrhage_path)).copy()
        hemorrhage_seg = 255. - hemorrhage_seg
        hemorrhage_mask = (hemorrhage_seg >= 250.).astype(np.float32)
        hemorrhage_mask = Image.fromarray(hemorrhage_mask)
        hemorrhage_mask = transforms.functional.resize(hemorrhage_mask, (256, 256),
                                                       interpolation=InterpolationMode.NEAREST)
        hemorrhage_mask = np.array(hemorrhage_mask)
        # print(hemorrhage_mask.shape)
    else:
        hemorrhage_mask = np.zeros((256, 256))

    if others_path != "none":
        others_seg = np.array(Image.open(others_path)).copy()
        others_seg = 255. - others_seg
        others_mask = (others_seg >= 250.).astype(np.float32)
        others_mask = Image.fromarray(others_mask)
        others_mask = transforms.functional.resize(others_mask, (256, 256),
                                                   interpolation=InterpolationMode.NEAREST)
        others_mask = np.array(others_mask)
    else:
        others_mask = np.zeros((256, 256))

    if scar_path != "none":
        scar_seg = np.array(Image.open(scar_path)).copy()
        scar_seg = 255. - scar_seg
        scar_mask = (scar_seg >= 250.).astype(np.float32)
        scar_mask = Image.fromarray(scar_mask)
        scar_mask = transforms.functional.resize(scar_mask, (256, 256),
                                                 interpolation=InterpolationMode.NEAREST)
        scar_mask = np.array(scar_mask)

    else:
        scar_mask = np.zeros((256, 256))

    mask_all = np.zeros((256, 256))
    mask_all[drusen_mask == 1] = 1
    mask_all[exudate_mask == 1] = 2
    mask_all[hemorrhage_mask == 1] = 3
    mask_all[others_mask == 1] = 4
    mask_all[scar_mask == 1] = 5

    gt_save = decode_seg_map_sequence(mask_all) * 255

    gt_save = gt_save*0.5 + np.array(img_ori).astype('uint8')*0.5
    # print(gt.max(),gt.min(),gt.shape)
    gt_save = Image.fromarray(np.uint8(gt_save))


    if save_img:
        gt_save = transforms.functional.resize(gt_save, (2000,2000),
                                                 interpolation=InterpolationMode.BILINEAR)
        gt_save.save(save_gt_path+img_name)

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

        pred_b_main_soft = model(img.cuda(), task=2)

        pred_b_main = torch.argmax(pred_b_main_soft, dim=1)

        pred_img_all.append(pred_b_main_soft)
        label_img_all.append(transforms.functional.to_tensor(mask_all).type(torch.LongTensor).cuda())

        _, dice_arr_one, trg_class_number_one,pred_class_number_one = dice_eval(pred=pred_b_main_soft, label=transforms.functional.to_tensor(mask_all).type(torch.LongTensor).cuda(),
                                              n_class=6)
        dice_arr_one = np.hstack(dice_arr_one)

        pred_b_main = pred_b_main.cpu().data.numpy()
        # print(pred_b_main.shape)
        pred_trg = decode_seg_map_sequence(pred_b_main[0, ...].copy()) * 255
        # print(gt.max(),gt.min(),gt.shape)
        pred_trg = pred_trg * 0.5 + np.array(img_ori).astype('uint8') * 0.5
        pred_trg = Image.fromarray(np.uint8(pred_trg))
        if save_img:
            pred_trg = transforms.functional.resize(pred_trg, (2000,2000),
                                                    interpolation=InterpolationMode.BILINEAR)
            pred_trg.save(save_pred_path+img_name)

        metric_start_time      = datetime.now()
        with open(osp.join("./{}/save_results/STL/lesion/".format(opt.arc) + "train", 'train_log.csv'), 'a') as f:
            # print(vCDR_error)
            log = [img_name,dice_arr_one[1],dice_arr_one[2],dice_arr_one[3],dice_arr_one[4],dice_arr_one[5],
                        trg_class_number_one[1],trg_class_number_one[2],trg_class_number_one[3],trg_class_number_one[4],trg_class_number_one[5]
                  ,pred_class_number_one[1],pred_class_number_one[2],pred_class_number_one[3],pred_class_number_one[4],pred_class_number_one[5]
                    ]
            log = map(str, log)
            f.write(','.join(log) + '\n')

pred_img_all = torch.cat(pred_img_all, dim=0)  ## Flat tensor
label_img_all = torch.cat(label_img_all, dim=0)
_, dice_arr, trg_class_number,pred_class_number = dice_eval(pred=pred_img_all, label=label_img_all,
                                          n_class=6)
dice_arr = np.hstack(dice_arr)

print('dice arr is {}'.format(dice_arr.shape))
print('Dice:')
print(dice_arr, trg_class_number,pred_class_number)
print('dsc_drusen :%.1f' % (dice_arr[1]))
print('dsc_exudate:%.1f' % (dice_arr[2]))
print('dsc_hemorrhage:%.1f' % (dice_arr[3]))
print('dsc_others:%.1f' % (dice_arr[4]))
print('dsc_scar:%.1f' % (dice_arr[5]))

pred_img_all = []
label_img_all = []
if not osp.exists("./{}/save_results/STL/lesion/".format(opt.arc) + "test"):
    os.makedirs("./{}/save_results/STL/lesion/".format(opt.arc) + "test")

with open(osp.join("./{}/save_results/STL/lesion/".format(opt.arc) + "test",'test_log.csv'), 'w') as f:
    log = [ ["img_name"] +["dice drusen"]+["dice exudate"]+["dice hemorrhage"]+["dice others"]+["dice scar"]+
            ["area drusen"]+["area exudate"]+["area hemorrhage"]+["area others"]+["area scar"]
           +["area pred drusen"]+["area pred exudate"]+["area pred hemorrhage"]+["area pred others"]+["area pred scar"]]
    log = map(str, log)
    f.write(','.join(log) + '\n')

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
    save_img_path = "./{}/save_results/STL/lesion/".format(opt.arc) + "test" + "/img/"
    save_gt_path = "./{}/save_results/STL/lesion/".format(opt.arc) + "test" + "/gt/"
    save_pred_path = "./{}/save_results/STL/lesion/".format(opt.arc) + "test" + "/pred/"

    if not os.path.exists(save_img_path):
        os.makedirs(save_img_path)
    if not os.path.exists(save_gt_path):
        os.makedirs(save_gt_path)
    if not os.path.exists(save_pred_path):
        os.makedirs(save_pred_path)
    # Image
    img_input = Image.open(img_path).convert('RGB')
    # print("img_size:",np.array(img).shape)
    # w, h = img.size
    img_ori = transforms.functional.resize(img_input, (256, 256), interpolation=InterpolationMode.BILINEAR)

    if drusen_path != "none":
        drusen_seg = np.array(Image.open(drusen_path)).copy()

        drusen_seg = 255. - drusen_seg
        drusen_mask = (drusen_seg >= 250.).astype(np.float32)
        drusen_mask = Image.fromarray(drusen_mask)
        drusen_mask = transforms.functional.resize(drusen_mask, (256, 256),
                                                   interpolation=InterpolationMode.NEAREST)
        drusen_mask = np.array(drusen_mask)
        # # print("drusen size:",torch.unique(drusen_mask))
        # print(drusen_mask.shape)
    else:
        drusen_mask = np.zeros((256, 256))

    if exudate_path != "none":
        exudate_seg = np.array(Image.open(exudate_path)).copy()
        exudate_seg = 255. - exudate_seg
        exudate_mask = (exudate_seg >= 250.).astype(np.float32)
        exudate_mask = Image.fromarray(exudate_mask)
        exudate_mask = transforms.functional.resize(exudate_mask, (256, 256),
                                                    interpolation=InterpolationMode.NEAREST)
        exudate_mask = np.array(exudate_mask)
        # print(exudate_mask.shape)
    else:
        exudate_mask = np.zeros((256, 256))

    if hemorrhage_path != "none":
        hemorrhage_seg = np.array(Image.open(hemorrhage_path)).copy()
        hemorrhage_seg = 255. - hemorrhage_seg
        hemorrhage_mask = (hemorrhage_seg >= 250.).astype(np.float32)
        hemorrhage_mask = Image.fromarray(hemorrhage_mask)
        hemorrhage_mask = transforms.functional.resize(hemorrhage_mask, (256, 256),
                                                       interpolation=InterpolationMode.NEAREST)
        hemorrhage_mask = np.array(hemorrhage_mask)
        # print(hemorrhage_mask.shape)
    else:
        hemorrhage_mask = np.zeros((256, 256))

    if others_path != "none":
        others_seg = np.array(Image.open(others_path)).copy()
        others_seg = 255. - others_seg
        others_mask = (others_seg >= 250.).astype(np.float32)
        others_mask = Image.fromarray(others_mask)
        others_mask = transforms.functional.resize(others_mask, (256, 256),
                                                   interpolation=InterpolationMode.NEAREST)
        others_mask = np.array(others_mask)
    else:
        others_mask = np.zeros((256, 256))

    if scar_path != "none":
        scar_seg = np.array(Image.open(scar_path)).copy()
        scar_seg = 255. - scar_seg
        scar_mask = (scar_seg >= 250.).astype(np.float32)
        scar_mask = Image.fromarray(scar_mask)
        scar_mask = transforms.functional.resize(scar_mask, (256, 256),
                                                 interpolation=InterpolationMode.NEAREST)
        scar_mask = np.array(scar_mask)

    else:
        scar_mask = np.zeros((256, 256))

    mask_all = np.zeros((256, 256))
    mask_all[drusen_mask == 1] = 1
    mask_all[exudate_mask == 1] = 2
    mask_all[hemorrhage_mask == 1] = 3
    mask_all[others_mask == 1] = 4
    mask_all[scar_mask == 1] = 5

    gt_save = decode_seg_map_sequence(mask_all) * 255

    gt_save = gt_save*0.5 + np.array(img_ori).astype('uint8')*0.5
    # print(gt.max(),gt.min(),gt.shape)
    gt_save = Image.fromarray(np.uint8(gt_save))


    if save_img:
        gt_save = transforms.functional.resize(gt_save, (2000,2000),
                                                     interpolation=InterpolationMode.BILINEAR)
        gt_save.save(save_gt_path+img_name)

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

        pred_b_main_soft = model(img.cuda(),task=2)

        pred_b_main = torch.argmax(pred_b_main_soft, dim=1)
        pred_img_all.append(pred_b_main_soft)
        label_img_all.append(transforms.functional.to_tensor(mask_all).type(torch.LongTensor).cuda())
        _, dice_arr_one, trg_class_number_one, pred_class_number_one = dice_eval(pred=pred_b_main_soft, label=transforms.functional.to_tensor(mask_all).type(torch.LongTensor).cuda(),
                                              n_class=6)
        dice_arr_one = np.hstack(dice_arr_one)

        pred_b_main = pred_b_main.cpu().data.numpy()
        # print(pred_b_main.shape)
        pred_trg = decode_seg_map_sequence(pred_b_main[0, ...].copy()) * 255
        # print(gt.max(),gt.min(),gt.shape)
        pred_trg = pred_trg * 0.5 + np.array(img_ori).astype('uint8') * 0.5
        pred_trg = Image.fromarray(np.uint8(pred_trg))
        if save_img:
            pred_trg = transforms.functional.resize(pred_trg, (2000,2000),
                                                    interpolation=InterpolationMode.BILINEAR)
            pred_trg.save(save_pred_path+img_name)

        metric_start_time      = datetime.now()
        with open(osp.join("./{}/save_results/STL/lesion/".format(opt.arc) + "test", 'test_log.csv'), 'a') as f:
            # print(vCDR_error)
            log = [img_name,dice_arr_one[1],dice_arr_one[2],dice_arr_one[3],dice_arr_one[4],dice_arr_one[5]
                    ,  trg_class_number_one[1],trg_class_number_one[2],trg_class_number_one[3],trg_class_number_one[4],trg_class_number_one[5]
                  ,pred_class_number_one[1],pred_class_number_one[2],pred_class_number_one[3],pred_class_number_one[4],pred_class_number_one[5]
                    ]
            log = map(str, log)
            f.write(','.join(log) + '\n')


pred_img_all = torch.cat(pred_img_all, dim=0)  ## Flat tensor
label_img_all = torch.cat(label_img_all, dim=0)
_, dice_arr, trg_class_number,pred_class_number = dice_eval(pred=pred_img_all, label=label_img_all,
                                          n_class=6)
dice_arr = np.hstack(dice_arr)

print('dice arr is {}'.format(dice_arr.shape))
print('Dice:')
print(dice_arr,trg_class_number,pred_class_number)
print('dsc_drusen :%.1f' % (dice_arr[1]))
print('dsc_exudate:%.1f' % (dice_arr[2]))
print('dsc_hemorrhage:%.1f' % (dice_arr[3]))
print('dsc_others:%.1f' % (dice_arr[4]))
print('dsc_scar:%.1f' % (dice_arr[5]))
