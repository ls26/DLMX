import os

import torch
import subprocess as sp
import numpy as np
from models import *
from skimage import morphology
import scipy
from PIL import Image
from matplotlib.pyplot import imsave
# from keras.preprocessing import image
import skimage
from skimage.measure import regionprops
from scipy.ndimage.measurements import label
from skimage.transform import rotate, resize
from skimage import measure, draw
import os
import os.path as osp
import torch.nn.functional as F
import torch.nn as nn
import math
import matplotlib.pyplot as plt
plt.switch_backend('agg')

import cv2
"""Functions for ramping hyperparameters up or down
Each function takes the current training step or epoch, and the
ramp length in the same format, and returns a multiplier between
0 and 1.
"""

import numpy as np

EPS = 1e-7

def check_gpu():
    """
    Selects an available GPU
    """
    available_gpu = -1
    ACCEPTABLE_USED_MEMORY = 500
    COMMAND = "nvidia-smi --query-gpu=memory.used --format=csv"
    _output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]
    memory_used_info = _output_to_list(sp.check_output(COMMAND.split()))[1:]
    memory_used_values = [int(x.split()[0]) for i, x in enumerate(memory_used_info)]
    for k in range(len(memory_used_values)):
        if memory_used_values[k]<ACCEPTABLE_USED_MEMORY:
            available_gpu = k
            break
    return available_gpu


def cross_entropy_2d(pred,label,task_seg=None):
    '''
    Args:
        predict:(n, c, h, w)
        target: (n, h, w)
    '''
    assert not label.requires_grad

    assert pred.dim()   == 4
    assert label.dim()  == 3
    assert pred.size(0) == label.size(0), f'{pred.size(0)}vs{label.size(0)}'
    assert pred.size(2) == label.size(1), f'{pred.size(2)}vs{label.size(2)}'
    assert pred.size(3) == label.size(2), f'{pred.size(3)}vs{label.size(3)}'

    n,c,h,w = pred.size()
    label   = label.view(-1)
    class_count = torch.bincount(label).float()
    try:
        assert class_count.size(0) == 6
        new_class_count = class_count
    except:
        new_class_count = torch.zeros(6).cuda().float()
        new_class_count[:class_count.size(0)] = class_count

    #print(class_count,new_class_count)

    weight      = (1 - new_class_count/label.size(0))
    #print(weight)
    #if task_seg=="stl_lesion":
    weight+=1e-7
    #print(weight)
    pred    = pred.transpose(1,2).transpose(2,3).contiguous() #n*c*h*w->n*h*c*w->n*h*w*c
    pred    = pred.view(-1,c)
    loss    = F.cross_entropy(input=pred,target=label,weight=weight)

    return loss



def entropy_loss(v):
    """
        Entropy loss for probabilistic prediction vectors
        input: batch_size x channels x h x w
        output: batch_size x 1 x h x w
    """
    assert v.dim() == 4
    n, c, h, w = v.size()
    return -torch.sum(torch.mul(v, torch.log2(v + 1e-30))) / (n * h * w * np.log2(c))

def dice_loss(pred, target):
    """
    input is a torch variable of size [N,C,H,W]
    target: [N,H,W]
    """
    n,c,h,w        = pred.size()
    pred           = pred.cuda()
    target         = target.cuda()
    target_onehot  = torch.zeros([n,c,h,w]).cuda()
    target         = torch.unsqueeze(target,dim=1) # n*1*h*w
    target_onehot.scatter_(1,target,1)

    assert pred.size() == target_onehot.size(), "Input sizes must be equal."
    assert pred.dim()  == 4, "Input must be a 4D Tensor."
    uniques = np.unique(target_onehot.cpu().data.numpy())
    assert set(list(uniques)) <= set([0, 1]), "target must only contain zeros and ones"

    eps = 1e-5
    probs = F.softmax(pred,dim=1)
    num   = probs * target_onehot  # b,c,h,w--p*g
    num   = torch.sum(num, dim=3)  # b,c,h
    num   = torch.sum(num, dim=2)  # b,c,

    den1 = probs * probs  # --p^2
    den1 = torch.sum(den1, dim=3)  # b,c,h
    den1 = torch.sum(den1, dim=2)  # b,c,

    den2 = target_onehot * target_onehot  # --g^2
    den2 = torch.sum(den2, dim=3)  # b,c,h
    den2 = torch.sum(den2, dim=2)  # b,c

    dice = 2.0 * (num / (den1 + den2+eps))  # b,c

    dice_total =  torch.sum(dice) / dice.size(0)  # divide by batch_sz
    return 1 - 1.0 * dice_total/5.0

def refine_seg(pred):
    '''
    Only retain the biggest connected component of a segmentation map.
    '''
    np_pred = pred.numpy()
        
    largest_ccs = []

    for i in range(np_pred.shape[0]):
        labeled, ncomponents = scipy.ndimage.measurements.label(np_pred[i,:,:])
        bincounts = np.bincount(labeled.flat)[1:]
        if len(bincounts) == 0:
            largest_cc = labeled == 0
        else:
            largest_cc = labeled == np.argmax(bincounts)+1
        largest_cc = torch.tensor(largest_cc, dtype=torch.float32)
        largest_ccs.append(largest_cc)
    largest_ccs = torch.stack(largest_ccs)
    
    return largest_ccs
    

def compute_dice_coef(input, target):
    '''
    Compute dice score metric.
    '''
    batch_size = input.shape[0]
    return sum([dice_coef_sample(input[k,:,:], target[k,:,:]) for k in range(batch_size)])/batch_size


def dice_coef_sample(input, target):
    iflat = input.contiguous().view(-1)
    tflat = target.type(torch.float32).contiguous().view(-1)
    intersection = (iflat * tflat).sum()
    res = (2. * intersection) / (iflat.sum() + tflat.sum())
    return res


def vertical_diameter(binary_segmentation):
    '''
    Get the vertical diameter from a binary segmentation.
    The vertical diameter is defined as the "fattest" area of the binary_segmentation parameter.
    '''
    # get the sum of the pixels in the vertical axis
    vertical_axis_diameter = np.sum(binary_segmentation, axis=1)

    # pick the maximum value
    diameter = np.max(vertical_axis_diameter, axis=1)

    # return it
    return diameter


def vertical_cup_to_disc_ratio(od, oc):
    '''
    Compute the vertical cup-to-disc ratio from a given labelling map.
    '''
    cup_diameter = vertical_diameter(oc)
    disc_diameter = vertical_diameter(od)
    return cup_diameter / (disc_diameter + EPS)


def compute_vCDR_error(pred_vCDR, gt_vCDR):
    '''
    Compute vCDR prediction error, along with predicted vCDR and ground truth vCDR.
    '''
    vCDR_err = np.mean(np.abs(gt_vCDR - pred_vCDR))
    return vCDR_err




def fov_error(seg_fov, fov_coord):
    mass_centers = [scipy.ndimage.measurements.center_of_mass(seg_fov[k,0,:,:]) for k in range(seg_fov.shape[0])]
    mass_centers = np.array([[elt[1], elt[0]] for elt in mass_centers])
    err = np.sqrt(np.sum((fov_coord-mass_centers)**2, axis=1)).mean()
    return err


class logger():
    """
    Simple logger, which display the wanted metrics in a proper format.
    """
    def __init__(self, metrics, to_disp=['auc', 'auc_unet', 'auc_vcdr', 'dsc_od', 'dsc_oc', 'vCDR_error', 'fov_error', 'auc_ppa']):
        self.metrics=metrics
        self.to_disp=to_disp
        
    def log(self, n_epoch, n_iter, train_metrics, val_metrics, train_loss, val_loss, metric_best, method):
        if method == "STL":
            print('EVAL epoch {} iter {}: '.format(n_epoch, n_iter), ' ' * 50)
            for k in range(len(self.metrics)):
                if self.metrics[k] in self.to_disp:
                    print('{} : {:.4f} / {:.4f}'.format(self.metrics[k], train_metrics[k]['reduced'],
                                                        val_metrics[k]['reduced']))
            print('_' * 50)
            return metric_best
        else:
            val_best = val_metrics[0]['reduced']+val_metrics[1]['reduced']+val_metrics[4]['reduced']+val_metrics[5]['reduced']

            if val_best>metric_best:
                # print(val_metrics[0]['reduced'], val_metrics[1]['reduced'], val_metrics[4]['reduced'],
                #       val_metrics[5]['reduced'])
                metric_best = val_best
                print('EVAL epoch {} iter {}: '.format(n_epoch, n_iter), ' ' * 50)
                for k in range(len(self.metrics)):
                    if self.metrics[k] in self.to_disp:
                        # print(k)
                        print('{} : {:.4f} / {:.4f}'.format(self.metrics[k], train_metrics[k]['reduced'],
                                                            val_metrics[k]['reduced']))
                print('_' * 50)
            return metric_best
                  


def get_vCDRs(preds, gts):
    pred_od = preds[1][:,0,:,:].cpu().numpy()
    pred_oc = preds[2][:,0,:,:].cpu().numpy()
    gt_od = gts[1][:,0,:,:].cpu().numpy()
    gt_oc = gts[2][:,0,:,:].cpu().numpy()
    pred_vCDR = vertical_cup_to_disc_ratio(pred_od, pred_oc)
    gt_vCDR = vertical_cup_to_disc_ratio(gt_od, gt_oc)
    return pred_vCDR, gt_vCDR

def construct_color_img(prob_per_slice):
    shape = prob_per_slice.shape
    img = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
    img[:, :, 0] = prob_per_slice * 255
    img[:, :, 1] = prob_per_slice * 255
    img[:, :, 2] = prob_per_slice * 255

    im_color = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    return im_color


def normalize_ent(ent):
    '''
    Normalizate ent to 0 - 1
    :param ent:
    :return:
    '''
    min = np.amin(ent)
    return (ent - min) / 0.4


def draw_ent(prediction, save_root, name):
    '''
    Draw the entropy information for each img and save them to the save path
    :param prediction: [2, h, w] numpy
    :param save_path: string including img name
    :return: None
    '''
    if not os.path.exists(os.path.join(save_root, 'disc')):
        os.makedirs(os.path.join(save_root, 'disc'))
    if not os.path.exists(os.path.join(save_root, 'cup')):
        os.makedirs(os.path.join(save_root, 'cup'))
    smooth = 1e-8
    cup = prediction[0]
    disc = prediction[1]
    cup_ent = - cup * np.log(cup + smooth)
    disc_ent = - disc * np.log(disc + smooth)
    cup_ent = normalize_ent(cup_ent)
    disc_ent = normalize_ent(disc_ent)
    disc = construct_color_img(disc_ent)
    cv2.imwrite(os.path.join(save_root, 'disc', name.split('.')[0]) + '.png', disc)
    cup = construct_color_img(cup_ent)
    cv2.imwrite(os.path.join(save_root, 'cup', name.split('.')[0]) + '.png', cup)


def draw_mask(prediction, save_root, name):
    '''
    Draw the mask probability for each img and save them to the save path
   :param prediction: [2, h, w] numpy
   :param save_path: string including img name
   :return: None
   '''
    if not os.path.exists(os.path.join(save_root, 'disc')):
        os.makedirs(os.path.join(save_root, 'disc'))
    if not os.path.exists(os.path.join(save_root, 'cup')):
        os.makedirs(os.path.join(save_root, 'cup'))
    cup = prediction[0]
    disc = prediction[1]

    disc = construct_color_img(disc)
    cv2.imwrite(os.path.join(save_root, 'disc', name.split('.')[0]) + '.png', disc)
    cup = construct_color_img(cup)
    cv2.imwrite(os.path.join(save_root, 'cup', name.split('.')[0]) + '.png', cup)

def draw_boundary(prediction, save_root, name):
    '''
    Draw the mask probability for each img and save them to the save path
   :param prediction: [2, h, w] numpy
   :param save_path: string including img name
   :return: None
   '''
    if not os.path.exists(os.path.join(save_root, 'boundary')):
        os.makedirs(os.path.join(save_root, 'boundary'))
    boundary = prediction[0]
    boundary = construct_color_img(boundary)
    cv2.imwrite(os.path.join(save_root, 'boundary', name.split('.')[0]) + '.png', boundary)


def get_largest_fillhole(binary):
    label_image = skimage.measure.label(binary)
    regions = regionprops(label_image)
    area_list = []
    for region in regions:
        area_list.append(region.area)
    if area_list:
        idx_max = np.argmax(area_list)
        binary[label_image != idx_max + 1] = 0
    return scipy.ndimage.binary_fill_holes(np.asarray(binary).astype(int))

def postprocessing(prediction, threshold=0.5, dataset='G'):
    if dataset[0] == 'D':
        prediction = prediction.numpy()
        prediction_copy = np.copy(prediction)
        disc_mask = prediction[1]
        cup_mask = prediction[0]
        disc_mask = (disc_mask > 0.5)  # return binary mask
        cup_mask = (cup_mask > 0.1)  # return binary mask
        disc_mask = disc_mask.astype(np.uint8)
        cup_mask = cup_mask.astype(np.uint8)
        for i in range(5):
            disc_mask = scipy.signal.medfilt2d(disc_mask, 7)
            cup_mask = scipy.signal.medfilt2d(cup_mask, 7)
        disc_mask = morphology.binary_erosion(disc_mask, morphology.diamond(7)).astype(np.uint8)  # return 0,1
        cup_mask = morphology.binary_erosion(cup_mask, morphology.diamond(7)).astype(np.uint8)  # return 0,1
        disc_mask = get_largest_fillhole(disc_mask).astype(np.uint8)  # return 0,1
        cup_mask = get_largest_fillhole(cup_mask).astype(np.uint8)
        prediction_copy[0] = cup_mask
        prediction_copy[1] = disc_mask
        return prediction_copy
    else:
        prediction = prediction.numpy()
        prediction = (prediction > threshold)  # return binary mask
        prediction = prediction.astype(np.uint8)
        prediction_copy = np.copy(prediction)
        disc_mask = prediction[1]
        cup_mask = prediction[0]

        # for i in range(5):
        #     disc_mask = scipy.signal.medfilt2d(disc_mask, 7)
        #     cup_mask = scipy.signal.medfilt2d(cup_mask, 7)
        # disc_mask = morphology.binary_erosion(disc_mask, morphology.diamond(7)).astype(np.uint8)  # return 0,1
        # cup_mask = morphology.binary_erosion(cup_mask, morphology.diamond(7)).astype(np.uint8)  # return 0,1
        # # print(disc_mask, cup_mask)
        disc_mask = get_largest_fillhole(disc_mask).astype(np.uint8)  # return 0,1
        cup_mask = get_largest_fillhole(cup_mask).astype(np.uint8)
        prediction_copy[0] = cup_mask
        prediction_copy[1] = disc_mask
        return prediction_copy


def joint_val_image(image, prediction, mask):
    ratio = 0.5
    _pred_cup = np.zeros([mask.shape[-2], mask.shape[-1], 3])
    _pred_disc = np.zeros([mask.shape[-2], mask.shape[-1], 3])
    _mask = np.zeros([mask.shape[-2], mask.shape[-1], 3])
    image = np.transpose(image, (1, 2, 0))

    _pred_cup[:, :, 0] = prediction[0]
    _pred_cup[:, :, 1] = prediction[0]
    _pred_cup[:, :, 2] = prediction[0]
    _pred_disc[:, :, 0] = prediction[1]
    _pred_disc[:, :, 1] = prediction[1]
    _pred_disc[:, :, 2] = prediction[1]
    _mask[:,:,0] = mask[0]
    _mask[:,:,1] = mask[1]

    pred_cup = np.add(ratio * image, (1 - ratio) * _pred_cup)
    pred_disc = np.add(ratio * image, (1 - ratio) * _pred_disc)
    mask_img = np.add(ratio * image, (1 - ratio) * _mask)

    joint_img = np.concatenate([image, mask_img, pred_cup, pred_disc], axis=1)
    return joint_img


def save_val_img(path, epoch, img):
    name = osp.join(path, "visualization", "epoch_%d.png" % epoch)
    out = osp.join(path, "visualization")
    if not osp.exists(out):
        os.makedirs(out)
    img_shape = img[0].shape
    stack_image = np.zeros([len(img) * img_shape[0], img_shape[1], img_shape[2]])
    for i in range(len(img)):
        stack_image[i * img_shape[0] : (i + 1) * img_shape[0], :, : ] = img[i]
    imsave(name, stack_image)




def save_per_img(patch_image, data_save_path, img_name, prob_map, mask_path=None, ext="bmp"):
    path1 = os.path.join(data_save_path, 'overlay', img_name.split('.')[0]+'.png')
    path0 = os.path.join(data_save_path, 'original_image', img_name.split('.')[0]+'.png')
    if not os.path.exists(os.path.dirname(path0)):
        os.makedirs(os.path.dirname(path0))
    if not os.path.exists(os.path.dirname(path1)):
        os.makedirs(os.path.dirname(path1))

    disc_map = prob_map[0]
    cup_map = prob_map[1]
    size = disc_map.shape
    disc_map[:, 0] = np.zeros(size[0])
    disc_map[:, size[1] - 1] = np.zeros(size[0])
    disc_map[0, :] = np.zeros(size[1])
    disc_map[size[0] - 1, :] = np.zeros(size[1])
    size = cup_map.shape
    cup_map[:, 0] = np.zeros(size[0])
    cup_map[:, size[1] - 1] = np.zeros(size[0])
    cup_map[0, :] = np.zeros(size[1])
    cup_map[size[0] - 1, :] = np.zeros(size[1])

    disc_mask = (disc_map > 0.5) # return binary mask
    cup_mask = (cup_map > 0.5)
    disc_mask = disc_mask.astype(np.uint8)
    cup_mask = cup_mask.astype(np.uint8)

    for i in range(5):
        disc_mask = scipy.signal.medfilt2d(disc_mask, 7)
        cup_mask = scipy.signal.medfilt2d(cup_mask, 7)
    disc_mask = morphology.binary_erosion(disc_mask, morphology.diamond(7)).astype(np.uint8) # return 0,1
    cup_mask = morphology.binary_erosion(cup_mask, morphology.diamond(7)).astype(np.uint8) # return 0,1
    disc_mask = get_largest_fillhole(disc_mask)
    cup_mask = get_largest_fillhole(cup_mask)

    disc_mask = morphology.binary_dilation(disc_mask, morphology.diamond(7)).astype(np.uint8)  # return 0,1
    cup_mask = morphology.binary_dilation(cup_mask, morphology.diamond(7)).astype(np.uint8)  # return 0,1

    disc_mask = get_largest_fillhole(disc_mask).astype(np.uint8) # return 0,1
    cup_mask = get_largest_fillhole(cup_mask).astype(np.uint8)


    contours_disc = measure.find_contours(disc_mask, 0.5)
    contours_cup = measure.find_contours(cup_mask, 0.5)

    patch_image2 = patch_image.astype(np.uint8)
    patch_image2 = Image.fromarray(patch_image2)

    patch_image2.save(path0)

    for n, contour in enumerate(contours_cup):
        patch_image[(contour[:, 0]).astype(int), (contour[:, 1]).astype(int), :] = [0, 255, 0]
        patch_image[(contour[:, 0] + 1.0).astype(int), (contour[:, 1]).astype(int), :] = [0, 255, 0]
        patch_image[(contour[:, 0] + 1.0).astype(int), (contour[:, 1] + 1.0).astype(int), :] = [0, 255, 0]
        patch_image[(contour[:, 0]).astype(int), (contour[:, 1] + 1.0).astype(int), :] = [0, 255, 0]
        patch_image[(contour[:, 0] - 1.0).astype(int), (contour[:, 1]).astype(int), :] = [0, 255, 0]
        patch_image[(contour[:, 0] - 1.0).astype(int), (contour[:, 1] - 1.0).astype(int), :] = [0, 255, 0]
        patch_image[(contour[:, 0]).astype(int), (contour[:, 1] - 1.0).astype(int), :] = [0, 255, 0]

    for n, contour in enumerate(contours_disc):
        patch_image[contour[:, 0].astype(int), contour[:, 1].astype(int), :] = [0, 0, 255]
        patch_image[(contour[:, 0] + 1.0).astype(int), (contour[:, 1]).astype(int), :] = [0, 0, 255]
        patch_image[(contour[:, 0] + 1.0).astype(int), (contour[:, 1] + 1.0).astype(int), :] = [0, 0, 255]
        patch_image[(contour[:, 0]).astype(int), (contour[:, 1] + 1.0).astype(int), :] = [0, 0, 255]
        patch_image[(contour[:, 0] - 1.0).astype(int), (contour[:, 1]).astype(int), :] = [0, 0, 255]
        patch_image[(contour[:, 0] - 1.0).astype(int), (contour[:, 1] - 1.0).astype(int), :] = [0, 0, 255]
        patch_image[(contour[:, 0]).astype(int), (contour[:, 1] - 1.0).astype(int), :] = [0, 0, 255]

    patch_image = patch_image.astype(np.uint8)
    patch_image = Image.fromarray(patch_image)

    patch_image.save(path1)

def untransform(img, lt):
    # img = (img + 1) * 127.5
    img = img * 255
    # lt = lt * 128
    return img, lt
