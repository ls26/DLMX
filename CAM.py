"""
通过实现Grad-CAM学习module中的forward_hook和backward_hook函数
"""
import cv2
import os
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
def img_transform(img_in, transform):
    """
    将img进行预处理，并转换成模型输入所需的形式—— B*C*H*W
    :param img_roi: np.array
    :return:
    """
    img = img_in.copy()
    img = Image.fromarray(np.uint8(img))
    img = transform(img)
    img = img.unsqueeze(0)    # C*H*W --> B*C*H*W
    return img


def img_preprocess(img_in, img_size):
    """
    读取图片，转为模型可读的形式
    :param img_in: ndarray, [H, W, C]
    :return: PIL.image
    """
    img = img_in.copy()
    img = cv2.resize(img,(img_size, img_size))
    img = img[:, :, ::-1]   # BGR --> RGB

    # img = np.array(img, dtype=np.uint8)
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # img = clahe.apply(np.array(img, dtype=np.uint8))  # CLAHE
    # img = np.array(img, dtype=np.uint8)
    img = Image.fromarray(img)
    img = TF.adjust_gamma(img, gamma=1.2, gain=1)  # gamma校正

    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize([0.4948052, 0.48568845, 0.44682974], [0.24580306, 0.24236229, 0.2603115])
    ])
    img_input = img_transform(img, transform)
    return img_input


def backward_hook(module, grad_in, grad_out):
    grad_block.append(grad_out[0].detach())


def farward_hook(module, input, output):
    fmap_block.append(output)


def show_cam_on_image(img, mask, out_dir, id):
    heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    out_dir_id = os.path.join(out_dir, id)
    path_cam_img = out_dir_id+"_cam.jpg"
    path_raw_img = out_dir_id+"_raw.jpg"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    cam_resize = cv2.resize(np.uint8(255 * cam), (2000, 2000))
    img_resize = cv2.resize(np.uint8(255 * img), (2000, 2000))
    cv2.imwrite(path_cam_img, cam_resize)
    cv2.imwrite(path_raw_img, img_resize)


def comp_class_vec(ouput_vec, index=None):
    """
    计算类向量
    :param ouput_vec: tensor
    :param index: int，指定类别
    :return: tensor
    """
    if not index:
        index = np.argmax(ouput_vec.cpu().data.numpy(),1)
        print("use")
    else:
        index = np.array(index)
    index = index[np.newaxis]
    index = torch.from_numpy(index)
    # print(index.shape,output.shape)
    one_hot = torch.zeros(1, output.shape[1]).scatter_(1, index, 1).cuda()
    one_hot.requires_grad = True
    class_vec = torch.sum(one_hot * output)  # one_hot = 11.8605

    return class_vec

def gen_cam(feature_map, grads, img_size):
    """
    依据梯度和特征图，生成cam
    :param feature_map: np.array， in [C, H, W]
    :param grads: np.array， in [C, H, W]
    :return: np.array, [H, W]
    """
    cam = np.zeros(feature_map.shape[1:], dtype=np.float32)  # cam shape (H, W)

    weights = np.mean(grads, axis=(1, 2))  #

    for i, w in enumerate(weights):
        cam += w * feature_map[i, :, :]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (img_size, img_size))
    cam -= np.min(cam)
    cam /= np.max(cam)

    return cam


if __name__ == '__main__':
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
    num_class = 4
    img_size = 224
    df = pd.read_excel('/mnt/sda/fengwei/AMD_code/Training400/adam_data.xlsx',
                       index_col='ID')
    output_dir = "backward_hook_cam"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    task_groups = [{'ids': 0,
                    'type': 'classif',
                    'size': 1}]
    net = cls_model(task_groups, args).to(device)

    classes = ('Non AMD','AMD')
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
    for idx_ in df.index:
        path_img = df.loc[idx_,"img_path"]
        img_name = df.loc[idx_,"imgName"]
        # print(path_img,img_name)
        fmap_block = list()
        grad_block = list()
        # 图片读取；网络加载
        img = cv2.imread(path_img, 1)  # H*W*C
        img_input = img_preprocess(img, img_size)
        # print(img_input.shape)
        # 注册hook
        net.down4.register_forward_hook(farward_hook)
        net.down4.register_backward_hook(backward_hook)
        # forward
        output = net(img_input.cuda())
        output_soft = F.softmax(output, dim=1)
        # print(output_soft,output_soft.max(dim=1)[0])
        if np.max(output_soft[0].cpu().data.numpy())>0.5:
            idx = np.argmax(output.cpu().data.numpy())
            print("predict: {}".format(classes[idx]))

            # backward
            net.zero_grad()
            class_loss = comp_class_vec(output)
            print(class_loss.shape)
            class_loss.backward()

            # 生成cam
            grads_val = grad_block[0].cpu().data.numpy().squeeze()
            fmap = fmap_block[0].cpu().data.numpy().squeeze()
            cam = gen_cam(fmap, grads_val, img_size)

            # 保存cam图片
            img_show = np.float32(cv2.resize(img, (img_size, img_size))) / 255
            # img_show = np.float32(img) / 255
            show_cam_on_image(img_show, cam, output_dir, img_name)
            count += 1
            # if count>=10:
            #     break
