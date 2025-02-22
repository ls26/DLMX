import os
import torch
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
from loguru import logger
def loss_calc(pred,label,task_seg=None):

    '''
    This function returns cross entropy loss for semantic segmentation
    '''
    # pred shape is batch * c * h * w
    # label shape is b*h*w
    label = label.long().cuda()
    return cross_entropy_2d(pred, label,task_seg=task_seg)
def mean_BCE_loss(pred, gt):
    loss = F.binary_cross_entropy(pred, gt, reduction='mean')
    return loss
def focal_loss(pred, gt, alpha=1, gamma=2):
    BCE_loss = F.binary_cross_entropy(pred, gt, reduction='none')
    pt = torch.exp(-BCE_loss)
    F_loss = alpha * (1-pt)**gamma * BCE_loss

    return torch.mean(F_loss)
def focal_loss_2(pred, gt, alpha=1, gamma=2):
    BCE_loss = F.binary_cross_entropy(pred, gt, reduction='none')

    return torch.mean(BCE_loss)
def dice_eval(pred,label,n_class):
    '''
    pred:  b*c*h*w
    label: b*h*w
    '''
    pred     = torch.argmax(pred,dim=1)  # b*h*w
    dice     = 0
    dice_arr = []
    IOU_arr = []
    each_class_number = []
    eps      = 1e-7
    # print(pred,label)
    for i in range(n_class):
        A = (pred  == i)
        B = (label == i)
        each_class_number.append(torch.sum(B).cpu().data.numpy())
        inse  = torch.sum(A*B).float()
        union = (torch.sum(A)+torch.sum(B)).float()
        dice  += 2*inse/(union+eps)
        dice_arr.append((2*inse/(union+eps)).cpu().data.numpy())
        
        intersection_IOU = (A & B).sum()
        union_IOU = (A | B).sum()
        iou = (intersection_IOU + 1e-5) / (union_IOU + 1e-5)
        IOU_arr.append(iou.cpu().data.numpy())

    return dice,dice_arr,np.hstack(each_class_number),IOU_arr
def test_model_thres_all(model, test_loader, thres_val=0.5):
    total_loss = 0

    val_preds_all = []
    val_lesion_mask_all = []
    model.eval()
    with torch.no_grad():
        for i, (val_data, val_gts) in enumerate(test_loader):
            val_data = val_data.type(torch.float32).to(device)  # Turn into a batch
            val_lesion_mask = val_gts[2]
            val_lesion_mask = val_lesion_mask.squeeze().type(torch.LongTensor).cuda()
            # Test step
            val_preds = model(val_data,task=2)

            task_losses_2 = loss_calc(val_preds, val_lesion_mask)+dice_loss(val_preds, val_lesion_mask)
            #print(val_preds.shape,val_lesion_mask.shape,torch.unique(val_lesion_mask),loss_calc(val_preds, val_lesion_mask),dice_loss(val_preds, val_lesion_mask))
            total_loss += (task_losses_2.data)
            ########################## segmentation metric#############################
            # print(val_preds.shape,val_lesion_mask.shape)
            val_preds_all.append(val_preds)
            val_lesion_mask_all.append(val_lesion_mask)
    # print(all_preds)
    val_preds_all = torch.cat(val_preds_all, dim=0)  ## Flat tensor
    val_lesion_mask_all = torch.cat(val_lesion_mask_all, dim=0)
    # print(val_preds_all.shape, val_lesion_mask_all.shape)
    _, trg_dice_arr, trg_class_number, trg_IOU_arr = dice_eval(pred=val_preds_all, label=val_lesion_mask_all,
                                                  n_class=6)
    trg_dice_arr = np.hstack(trg_dice_arr)

    # print('Dice')
    #
    # print('######## Target Train Set ##########')
    print('Each Class Number {}'.format(trg_class_number))
    dsc_drusen = trg_dice_arr[1]
    dsc_exudate = trg_dice_arr[2]
    dsc_hemorrhage = trg_dice_arr[3]
    dsc_others = trg_dice_arr[4]
    dsc_scar = trg_dice_arr[5]

    trg_IOU_arr = np.hstack(trg_IOU_arr)
    IOU_drusen = trg_IOU_arr[1]
    IOU_exudate = trg_IOU_arr[2]
    IOU_hemorrhage = trg_IOU_arr[3]
    IOU_others = trg_IOU_arr[4]
    IOU_scar = trg_IOU_arr[5]

    return total_loss/len(test_loader),dsc_drusen,dsc_exudate,dsc_hemorrhage,\
           dsc_others,dsc_scar, \
           IOU_drusen,IOU_exudate,IOU_hemorrhage,IOU_others,IOU_scar
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', default='./checkpoints/')
parser.add_argument('--recover', default=False, type=bool, help='recover from a checkpoint')
parser.add_argument('--pretrained', default=False, type=bool)
parser.add_argument('--reco_name', default=None, type=str, help='model to recover from')
parser.add_argument('--reco_type', default='last_checkpoint', type=str,
                    help='which type of recovery (best_error or iter_XXX)')

parser.add_argument('--workers', default=6, type=int)
parser.add_argument('--batch_size', default=8, type=int)
parser.add_argument('--optimizer', default='adam', type=str)
parser.add_argument('--arc', default='efficient', type=str)

parser.add_argument('--one_optim_per_task', default=True, type=bool, help='use one optimizer for all tasks or one per task')

parser.add_argument('--per_batch_step', default=False, type=bool, help='optimize tasks altogether or one by one')
parser.add_argument('--learning_rate', default=1e-4, type=float)
parser.add_argument('--eval_interval', default=3, type=int, help='interval between two validations')
parser.add_argument('--total_epoch', default=30, type=int, help='number of training epochs to proceed')
parser.add_argument('--seed', default=None, type=int)

opt = parser.parse_args()

# Seed
if opt.seed:
    TRAIN_RANDOM_SEED = opt.seed
    print("training seed used:", TRAIN_RANDOM_SEED)
    cudnn.enabled = True
    torch.manual_seed(TRAIN_RANDOM_SEED)
    torch.cuda.manual_seed(TRAIN_RANDOM_SEED)
    torch.cuda.manual_seed_all(TRAIN_RANDOM_SEED)  # 为所有GPU设置随机种子
    np.random.seed(TRAIN_RANDOM_SEED)
    random.seed(TRAIN_RANDOM_SEED)

# Saving settings
model_dir = opt.checkpoint_path
os.mkdir(model_dir) if not os.path.isdir(model_dir) else None


log_dir = os.path.join('log',"STL_lesion", f'{opt.arc}')
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

logger.add(os.path.join(log_dir, 'log.txt'))
opt.logger = logger
opt.log_dir = log_dir

# Define model and optimiser
# gpu = utils.check_gpu()
# device = torch.device("cuda:{}".format(gpu) if torch.cuda.is_available() else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载数据集
df = pd.read_excel('/mnt/sdb/fengwei/AMD_code/Training400/adam_data.xlsx', index_col='ID')
train_df, test_df = train_test_split(df, test_size=0.2, random_state=2024)
train_feature_list = []
for i in range(len(train_df)):
    train_feature_list.append(train_df.iloc[i:i + 1].values.tolist()[0])
# print(len(train_feature_list[0]))
test_feature_list = []
for i in range(len(test_df)):
    test_feature_list.append(test_df.iloc[i:i + 1].values.tolist()[0])
# Create datasets and loaders
# transforms.Compose([transforms.RandomCrop((224, 224)),
#                                  #                                      transforms.RandomResizedCrop(512),
#                                  transforms.RandomHorizontalFlip(),
#                                  transforms.ToTensor(),
#                                  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
train_data = AMDDataset_v2(train_feature_list)
val_data = AMDDataset_v2(test_feature_list)

from torch.utils.data.sampler import WeightedRandomSampler
print(Counter(train_data.label_x))
print(Counter(val_data.label_x))
# print(train_data.label_x.shape)
class_sample_count = np.array([len(np.where(train_data.label_x == t)[0]) for t in np.unique(train_data.label_x)])
weight = 1. / class_sample_count
# print(weight)
samples_weight = np.array([weight[int(t)] for t in train_data.label_x])
samples_weight = torch.from_numpy(samples_weight)
samples_weight = samples_weight.double()
sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

train_loader = torch.utils.data.DataLoader(train_data, batch_size=opt.batch_size,
                                          # shuffle=True,
                                          sampler=sampler
                                          , num_workers=opt.workers)

# train_loader = torch.utils.data.DataLoader(
#     dataset=train_data,
#     batch_size=opt.batch_size,
#     num_workers=opt.workers,
#     shuffle=True,
#     drop_last=True)

val_loader = torch.utils.data.DataLoader(
    dataset=val_data,
    batch_size=opt.batch_size,
    num_workers=opt.workers,
    shuffle=False)


# model = MTL_UNet_v2(opt).to(device)
if opt.arc=="B3":
   model = get_efficientunet_b3_seg(out_channels=2, concat_input=True, pretrained=True).cuda()
elif opt.arc=="resnet":
   model = DeepLab_seg(backbone='resnet', output_stride=16).cuda()
elif opt.arc=="B0":
   #model = DeepLab(backbone='mobilenet', output_stride=16).cuda()
   model = get_efficientunet_b0_seg(out_channels=2, concat_input=True, pretrained=True).cuda()
# Few parameters
total_epoch = opt.total_epoch
nb_train_batches = len(train_loader)
nb_val_batches = len(val_loader)

CLASS_WEIGHT = torch.Tensor([0.4,1]).cuda()
loss_fun = torch.nn.CrossEntropyLoss(weight=CLASS_WEIGHT)

# Optimizer

if opt.optimizer.upper() == 'ADAM':
    optimizer_one = optim.Adam(model.parameters(), lr=opt.learning_rate)
elif opt.optimizer.upper() == 'SGD':
    optimizer_one = optim.SGD(model.parameters(), lr=opt.learning_rate)
optimizer_all = optimizer_one

# Iterations
train_avg_losses = 0.0
best_test_AUC = 0
n_iter = 0
n_epoch = 0
while n_epoch < total_epoch:
    ####################
    ##### Training #####
    ####################
    model.train()
    train_dataset = iter(train_loader)
    for data in tqdm.tqdm(train_loader):
        train_data, train_gts = data
        train_data = train_data.to(device)
        # print(train_gts[0].shape)

        train_lesion_mask = train_gts[2].to(device)
        # print(train_scar_mask.shape, torch.unique(train_scar_mask))
        # Forward pass
        logits = model(train_data, task=2)
        train_lesion_mask = train_lesion_mask.squeeze().type(torch.LongTensor).cuda()
        #print(loss_calc(logits, train_lesion_mask), dice_loss(logits, train_lesion_mask))
        task_losses_2 = loss_calc(logits, train_lesion_mask) + dice_loss(logits, train_lesion_mask)

        task_loss = task_losses_2
        # Backward pass

        optimizer_all.zero_grad()
        task_loss.backward()
        optimizer_all.step()
        # Incr iter nb
        n_iter += 1

    ######################
    ##### Validation #####
    ######################

    if (n_epoch + 1) % opt.eval_interval == 0:
        train_loss, train_dsc_drusen,train_dsc_exudate,train_dsc_hemorrhage,\
           train_dsc_others,train_dsc_scar,train_IOU_drusen,train_IOU_exudate,train_IOU_hemorrhage,train_IOU_others,train_IOU_scar = test_model_thres_all(model, train_loader)

        opt.logger.info("epoch {:3d}".format(n_epoch)+"\ttrain_loss: {0:.2f}".format(train_loss)+
            '\ttrain_dsc_drusen: {0:.2f}'.format(train_dsc_drusen)+
        "\ttrain_dsc_exudate: {0:.2f}".format(train_dsc_exudate)+
        "\ttrain_dsc_hemorrhage: {0:.2f}".format(train_dsc_hemorrhage)+
        "\ttrain_dsc_others: {0:.2f}".format( train_dsc_others)+
        "\ttrain_dsc_scar: {0:.2f}".format(train_dsc_scar))
        
        opt.logger.info("\ttrain_IOU_drusen: {0:.2f}".format(train_IOU_drusen)+
        "\ttrain_IOU_exudate: {0:.2f}".format(train_IOU_exudate)+
        "\ttrain_IOU_hemorrhage: {0:.2f}".format(train_IOU_hemorrhage)+
        "\ttrain_IOU_others: {0:.2f}".format( train_IOU_others)+
        "\ttrain_IOU_scar: {0:.2f}".format(train_IOU_scar))


        test_loss,test_dsc_drusen,test_dsc_exudate,test_dsc_hemorrhage,\
           test_dsc_others,test_dsc_scar,test_IOU_drusen,test_IOU_exudate,test_IOU_hemorrhage,\
           test_IOU_others,test_IOU_scar = test_model_thres_all(model, val_loader)

        if (test_dsc_drusen+test_dsc_exudate+test_dsc_hemorrhage+test_dsc_others+test_dsc_scar) > best_test_AUC:
            best_test_AUC = (test_dsc_drusen+test_dsc_exudate+test_dsc_hemorrhage+test_dsc_others+test_dsc_scar)
            best_model = copy.deepcopy(model)
            # if (perth_val):
            print("====================================")
            print("in best model, validating on testing subset")
            #torch.save(model.state_dict(), os.path.join(model_dir,'AMD_seg_lesion_{}.pkl'.format(opt.arc)))
            opt.logger.info("save model to"+os.path.join(model_dir,'AMD_seg_lesion_{}.pkl'.format(opt.arc)))
            model.train()
            opt.logger.info("epoch {:3d}".format(n_epoch)+
                  "\ttest_loss: {0:.2f}".format(test_loss)+
                  '\ttest_dsc_drusen: {0:.2f}'.format(test_dsc_drusen)+
                  "\ttest_dsc_exudate: {0:.2f}".format(test_dsc_exudate)+
                  "\ttest_dsc_hemorrhage: {0:.2f}".format(test_dsc_hemorrhage)+
                  "\ttest_dsc_others: {0:.2f}".format(test_dsc_others)+
                  "\ttest_dsc_scar: {0:.2f}".format(test_dsc_scar))
            opt.logger.info("\ttest_IOU_drusen: {0:.2f}".format(test_IOU_drusen)+
                  "\ttest_IOU_exudate: {0:.2f}".format(test_IOU_exudate)+
                  "\ttest_IOU_hemorrhage: {0:.2f}".format(test_IOU_hemorrhage)+
                  "\ttest_IOU_others: {0:.2f}".format(test_IOU_others)+
                  "\ttest_IOU_scar: {0:.2f}".format(test_IOU_scar))


    # Update epoch and LR
    n_epoch += 1


