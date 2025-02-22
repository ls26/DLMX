import os
import torch
import numpy as np
import argparse
import utils
import copy
from dataset import AMDDataset
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
from loguru import logger
def focal_loss(pred, gt, alpha=1, gamma=2):
    BCE_loss = F.binary_cross_entropy(pred, gt, reduction='none')
    pt = torch.exp(-BCE_loss)
    F_loss = alpha * (1-pt)**gamma * BCE_loss

    return torch.mean(F_loss)
def focal_loss_2(pred, gt, alpha=1, gamma=2):
    BCE_loss = F.binary_cross_entropy(pred, gt, reduction='none')

    return torch.mean(BCE_loss)
def test_model_thres(model, test_loader, thres_val=0.5):
    total_loss = 0
    model.eval()
    all_preds = []
    all_labels = []
    loss_fun = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for i, (val_data, val_gts) in enumerate(test_loader):
            val_data = val_data.type(torch.float32).to(device)  # Turn into a batch

            val_gts = val_gts[0].to(device)

            # Test step
            val_preds = model(val_data)
            task_losses = loss_fun(val_preds, val_gts.long())

            preds_after = F.softmax(val_preds, dim=1)[:, 1]
            all_preds.append(preds_after)
            all_labels.append(val_gts)
            total_loss += task_losses.data
    # print(all_preds)
    real_results = torch.cat(all_labels, dim=0).flatten().type(torch.float32).cpu().numpy()  ## Flat tensor
    pred_results = torch.cat(all_preds, dim=0).flatten().type(torch.float32).cpu().detach().numpy()
    print(real_results.shape, pred_results.shape)
    test_AUC_ROC = roc_auc_score(real_results, pred_results)
    pred_res = pred_results>=thres_val

    TP = sum((pred_res == 1) & (real_results == 1))
    FN = sum((pred_res == 0) & (real_results == 1))
    TN = sum((pred_res == 0) & (real_results == 0))
    FP = sum((pred_res == 1) & (real_results == 0))
    test_sens = TP / (TP + FN)
    test_spec = TN / (TN + FP)
    test_accuracy_calculated = (TN + TP) / (TN + TP + FP + FN)
    test_weighted_acc = (TN / (FP + TN)) * 0.5 + (TP / (TP + FN)) * 0.5
    # auc_confint_cal(real_results, pred_results, pred_res)
    precision_s = precision_score(real_results, pred_res)
    f1 = f1_score(real_results, pred_res)
    precision, recall, _ = precision_recall_curve(real_results, pred_res)
    aupr = auc(recall, precision)
    return test_sens, test_spec, test_accuracy_calculated,test_weighted_acc,test_AUC_ROC, \
           total_loss/len(test_loader),real_results, pred_results, precision_s, f1, aupr
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
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--optimizer', default='adam', type=str)

parser.add_argument('--per_batch_step', default=False, type=bool, help='optimize tasks altogether or one by one')
parser.add_argument('--learning_rate', default=1e-4, type=float)
parser.add_argument('--eval_interval', default=1, type=int, help='interval between two validations')
parser.add_argument('--total_epoch', default=100, type=int, help='number of training epochs to proceed')
parser.add_argument('--seed', default=None, type=int)
parser.add_argument('--arc', default='B3', type=str)
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


log_dir = os.path.join('log',"STL_cls", f'{opt.arc}')
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

logger.add(os.path.join(log_dir, 'log.txt'))
opt.logger = logger
opt.log_dir = log_dir

# Define model and optimiser
# gpu = utils.check_gpu()
# device = torch.device("cuda:{}".format(gpu) if torch.cuda.is_available() else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
task_groups = [{'ids': 0,
                'type': 'classif',
                'size': 1}]
# 加载数据集
df = pd.read_excel('adam_data.xlsx', index_col='ID')

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
train_data = AMDDataset(train_feature_list)
val_data = AMDDataset(test_feature_list)

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
    batch_size=64,
    num_workers=opt.workers,
    shuffle=False)

n_tasks = len(task_groups)
if opt.arc=="B3":
    model = get_efficientunet_b3_cls(out_channels=2, concat_input=True, pretrained=True).cuda()
elif opt.arc=="resnet":
#model = MTL_UNet_v2(opt).to(device)
    model = DeepLab_cls(backbone='resnet', output_stride=16).cuda()
elif opt.arc=="B0":
    #model = ResUnet(channel=3).cuda()
    model = get_efficientunet_b0_cls(out_channels=2, concat_input=True, pretrained=True).cuda()
    #model = DeepLab(backbone='mobilenet', output_stride=16).cuda()
#model = cls_model(task_groups, opt).to(device)

# Few parameters
total_epoch = opt.total_epoch
nb_train_batches = len(train_loader)
nb_val_batches = len(val_loader)



CLASS_WEIGHT = torch.Tensor([0.4,1]).cuda()
loss_fun = torch.nn.CrossEntropyLoss(weight=CLASS_WEIGHT)

# Optimizer
if opt.optimizer.upper() == 'ADAM':
    optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate)
elif opt.optimizer.upper() == 'SGD':
    optimizer = optim.SGD(model.parameters(), lr=opt.learning_rate)
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
        # print(train_data.shape)
        train_data = train_data.to(device)
        # print(train_gts[0].shape)
        train_gts = train_gts[0].to(device)
        # Train step
        logits = model(train_data)

        task_losses = loss_fun(logits,train_gts.long())

        # logits = model(train_data)
        # print(logits.shape)
        # task_losses = focal_loss(logits,train_gts)
        # task_losses = loss_fun(logits,train_gts.long())
        # Backward pass
        optimizer.zero_grad()
        task_losses.backward()
        optimizer.step()

        # Incr iter nb
        n_iter += 1
    ######################
    ##### Validation #####
    ######################

    if (n_epoch + 1) % opt.eval_interval == 0:
        train_sens, train_spec, train_accuracy_calculated, train_weighted_acc, train_AUC_ROC, train_loss, _, _, train_precision,\
        train_f1,train_aupr = test_model_thres(model, train_loader)

        opt.logger.info("epoch {:3d}".format(n_epoch)+"\ttrain_loss: {0:.2f}".format(train_loss)+
              "\ttrain_Sens: {0:.2f}".format(train_sens)+ "\ttrain_Spec: {0:.2f}".format(train_spec)+
              "\ttrain_Acc: {0:.2f}".format(train_accuracy_calculated)+
              '\ttrain_Weighted Acc: {0:.2f}'.format(train_weighted_acc)+
              "\ttrain_AUC: {0:.2f}".format(train_AUC_ROC)+
              "\ttrain_precision: {0:.2f}".format(train_precision)+
              "\ttrain_f1: {0:.2f}".format(train_f1)+
              "\ttrain_aupr: {0:.2f}".format(train_aupr))

        test_sens, test_spec, test_acc,test_weighted_acc,test_AUC_ROC,test_loss,_, _ ,test_precision,\
        test_f1,test_aupr = test_model_thres(model, val_loader)

        if ((test_AUC_ROC) > best_test_AUC):
            best_test_AUC = (test_AUC_ROC)
            best_model = copy.deepcopy(model)
            # if (perth_val):
            print("====================================")
            print("in best model, validating on testing subset")
            torch.save(model.state_dict(), os.path.join(model_dir,'AMD_CLS_{}.pkl'.format(opt.arc)))
            model.train()
            opt.logger.info("epoch {:3d}".format(n_epoch)+
                  "\ttest_loss: {0:.2f}".format(test_loss)+
                  "\ttest_Sens: {0:.2f}".format(test_sens)+
                  "\ttest_Spec: {0:.2f}".format(test_spec)+
                  "\ttest_Acc: {0:.2f}".format(test_acc)+
                  '\ttest_Weighted Acc: {0:.2f}'.format(test_weighted_acc)+
                  "\ttest_AUC: {0:.2f}".format(test_AUC_ROC)+
                  "\ttest_precision: {0:.2f}".format(test_precision)+
                  "\ttest_f1: {0:.2f}".format(test_f1)+
                  "\ttest_aupr: {0:.2f}".format(test_aupr))

    # Update epoch and LR
    n_epoch += 1


