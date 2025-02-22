import os
import torch
import numpy as np
import argparse
import utils
from losses import RefugeLoss
from dataset import RefugeDataset_test
from metrics import *
from saver import Saver
import os
import os.path as osp
import torch.nn.functional as F
from PIL import Image
import torch
import matplotlib.pyplot as plt
import tqdm
from torch.utils.data import DataLoader
import pytz
from sklearn.metrics import roc_auc_score,roc_curve,auc
import tqdm
from torch.utils.data import DataLoader
import pytz

import cv2
def compute_vCDR_error(pred_vCDR, gt_vCDR):
    '''
    Compute vCDR prediction error, along with predicted vCDR and ground truth vCDR.
    '''
    vCDR_err = np.mean(np.abs(gt_vCDR - pred_vCDR))
    return vCDR_err
parser = argparse.ArgumentParser()
parser.add_argument('--name', default='test', type=str, help='model name')
parser.add_argument('--method', required=True, type=str, help='which model to use (MTL, MGDA, PCGrad, gradnorm, MR)')
parser.add_argument('--active_task', default=0, type=int, help='which task to train for STL model (0: classification, 1: OD segmentation, 2: OC segmentation, 3: fovea localization)')

parser.add_argument('--dataroot', default='/mnt/workdir/fengwei/Glaucoma_Fundus_Imaging_Datasets/archive/REFUGE')
parser.add_argument('--split', default='val')
parser.add_argument('--checkpoint_path', default='/mnt/workdir/fengwei/Glaucoma_Fundus_Imaging_Datasets'
                                                 '/ga/glaucoma_mtl-main/glaucoma_mtl-main/checkpoints/test/')
parser.add_argument('--recover', default=True, type=bool, help='recover from a checkpoint')
parser.add_argument('--pretrained', default=True, type=bool)
parser.add_argument('--reco_name', default=None, type=str, help='model to recover from')

parser.add_argument('--workers', default=6, type=int)
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--optimizer', default='adam', type=str)
parser.add_argument('--per_batch_step', default=False, type=bool, help='optimize tasks altogether or one by one')
parser.add_argument('--one_optim_per_task', default=False, type=bool, help='use one optimizer for all tasks or one per task')
parser.add_argument('--learning_rate', default=1e-4, type=float)
parser.add_argument('--eval_interval', default=400, type=int, help='interval between two validations')
parser.add_argument('--total_epoch', default=500, type=int, help='number of training epochs to proceed')
parser.add_argument('--seed', default=None, type=int)

opt = parser.parse_args()
# Seed
if opt.seed:
    torch.manual_seed(opt.seed)
# Saving settings
model_dir = opt.checkpoint_path
# os.mkdir(model_dir) if not os.path.isdir(model_dir) else None
saver = Saver(model_dir, args=opt)
# test/best_error_weights.pth
# Define model and optimiser
# gpu = utils.check_gpu()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
task_groups = [{'ids':0,
                'type':'classif',
                'size': 1},
               {'ids':1,
                'type':'od',
                'size': 1},
               {'ids':2,
                'type':'oc',
                'size': 1},
               {'ids':3,
                'type':'fov',
                'size': 1}]

n_tasks = len(task_groups)
model = utils.select_model(opt, task_groups).to(device)

# Loss and metrics
loss_func = RefugeLoss(task_groups, opt)
test_metrics = RefugeMetrics(task_groups, opt)
logger = utils.logger(test_metrics.metrics_names)
saver.add_metrics(test_metrics.metrics_names)

# Recover weights, if required
# model.initialize(opt, device, model_dir, saver)
if opt.method == "STL":
    if opt.active_task==0:
        ckpt_file = opt.checkpoint_path +"best_error_weights"+ "_Glaucoma_clf"+".pth"
    elif opt.active_task==1:
        ckpt_file = opt.checkpoint_path +"best_error_weights"+ "_Glaucoma_od"+".pth"
    elif opt.active_task==2:
        ckpt_file = opt.checkpoint_path +"best_error_weights"+ "_Glaucoma_oc"+".pth"
    elif opt.active_task==4:
        ckpt_file = opt.checkpoint_path +"best_error_weights"+ "_Glaucoma_ppa"+".pth"
else:
    ckpt_file = opt.checkpoint_path +"best_error_weights"+ "_Glaucoma_MTL" + ".pth"


ckpt = torch.load(ckpt_file, map_location=device)

# Gets what needed in the checkpoint
pretrained_dict = {k: v for k, v in ckpt['model_state_dict'].items() if
                   'CONV' in k or 'BN' in k or 'FC' in k or 'outcs' in k or 'outfc' in k}

# Loads the weights
model.load_state_dict(pretrained_dict, strict=False)
model.clf = ckpt['classifier']
print('Weights and classifier recovered from {}.'.format(ckpt_file))

# For recovering only
model.n_epoch = ckpt['epoch']
model.n_iter = ckpt['n_iter'] + 1
model.eval()
# Create datasets and loaders
dataset_path = opt.dataroot
test_data = RefugeDataset_test(dataset_path,
                         split=opt.split)

test_loader = torch.utils.data.DataLoader(
    dataset=test_data,
    batch_size=1,
    num_workers=opt.workers,
    shuffle=False)
# Few parameters
total_epoch = opt.total_epoch
nb_test_batches = len(test_loader)

################
##### Test #####
################
test_avg_losses = np.zeros(n_tasks, dtype=np.float32)
model.eval()
test_dataset = iter(test_loader)

if not osp.exists(osp.join("./results_1", opt.method)):
    os.makedirs(osp.join("./results_1", opt.method))
with torch.no_grad():
    with open(osp.join("./results_1",opt.method, 'test_log.csv'), 'a') as f:
        log = [["img_index"] + ["img_name"] +['Glaucoma classification probability'] +["Glaucoma classification prediction "]+
               ["Glaucoma Classification GT"] + ['PPA probability'] +["PPA prediction"]+
               ["PPA GT"]+["test_pred_vCDR"]+ ["test_gt_vCDR"]+["CDR error"]]
        log = map(str, log)
        f.write(','.join(log) + '\n')
    all_preds = []
    all_labels = []
    for batch_idx, (sample) in tqdm.tqdm(enumerate(test_loader),
                                         total=len(test_loader),
                                         ncols=80, leave=False):
        test_data, test_gts, img_name = sample
        test_data = test_data.to(device)
        test_gts = [test_gts[0].to(device), test_gts[1].to(device), test_gts[2].to(device), [test_gts[3][0].to(device), test_gts[3][1].to(device)], test_gts[4].to(device)]

        # Logging
        print('Test {}/{} epoch {} iter {}'.format(batch_idx, nb_test_batches, model.n_epoch, model.n_iter), end=' '*50+'\r')

        # Test step
        task_losses, test_preds = model.test_step(test_data, test_gts, loss_func)
        test_pred_vCDR, test_gt_vCDR = utils.get_vCDRs(test_preds, test_gts)
        clf_preds = torch.from_numpy(model.clf.predict_proba(np.array(test_pred_vCDR).reshape(-1,1))[:,1]).type(torch.float32).to(device)
        test_preds.append(clf_preds)
        # print(len(test_preds))
        # print(test_preds[0],test_preds[1].min(),test_preds[2].shape,test_preds[-2])
        # print(task_losses)
        # Scoring
        # test_avg_losses += task_losses.cpu().numpy() / nb_test_batches
        test_metrics.incr(test_preds, test_gts, test_pred_vCDR, test_gt_vCDR)
        prediction = torch.cat((test_preds[1],test_preds[2]),1)
        # print(prediction.shape)
        # predictions = prediction
        prediction = torch.sigmoid(prediction)

        # # prediction = utils.postprocessing(prediction.data.cpu()[0])
        pred_od = utils.refine_seg((prediction[:, 0]>=0.6).type(torch.int8).cpu()).to(device)
        pred_oc = utils.refine_seg((prediction[:, 1] >= 0.6).type(torch.int8).cpu()).to(device)
        # print(pred_od.shape, pred_oc.shape)
        prediction = torch.cat((pred_od, pred_oc), 0).cpu().data.numpy()
        od = test_gts[1].to(device)

        oc = test_gts[2].to(device)

        # if batch_idx == 0:
        #     im = Image.fromarray(od[0, 0].cpu().data.numpy() * 255).convert('RGB')
        #     im.save("./results/od.jpg")
        #
        #     im = Image.fromarray(oc[0, 0].cpu().data.numpy() * 255).convert('RGB')
        #     im.save("./results/oc.jpg")
        #
        #     print(prediction[0,0].max(),prediction[0,0].min())
        #     prediction = (prediction.cpu().data.numpy() > 0.6)  # return binary mask
        #     prediction = prediction.astype(np.uint8)
        #     im = Image.fromarray(prediction[0,0] * 255).convert('RGB')
        #     im.save("./results/pod.jpg")
        #
        #     im = Image.fromarray(prediction[0,1] * 255).convert('RGB')
        #     im.save("./results/poc.jpg")
        # # print(od.shape,oc.shape)
        target_numpy = torch.cat((od,oc),1)
        target_numpy = target_numpy.data.cpu().data.numpy()[0]
        vCDR_error = compute_vCDR_error(test_pred_vCDR, test_gt_vCDR)
        with open(osp.join("./results_1",opt.method, 'test_log.csv'), 'a') as f:
            # print(vCDR_error)
            log = [batch_idx,img_name[0],round(test_preds[0].cpu().data.numpy()[0],4),int((test_preds[0].cpu().data.numpy()[0] > 0.5)),
                   round(test_gts[0].cpu().data.numpy()[0],4),
                   round(test_preds[-2].cpu().data.numpy()[0],4),
                   int((test_preds[-2].cpu().data.numpy()[0] > 0.5)),round(test_gts[4].cpu().data.numpy()[0],4),test_pred_vCDR[0], test_gt_vCDR[0], vCDR_error]
            log = map(str, log)
            f.write(','.join(log) + '\n')

        # # print(batch_idx,cup_dice,disc_dice)
        # imgs = test_data.data.cpu()
        # # print(imgs.max(),imgs.min())
        # for img, lt, lp in zip(imgs, [target_numpy], [prediction]):
        #     # print(img.shape, lt.shape, lp.shape)
        #     img, lt = utils.untransform(img, lt)
        #     utils.save_per_img(img.numpy().transpose(1, 2, 0), os.path.join("./results_1",opt.method),
        #                  img_name[0],
        #                  lp, mask_path=None, ext="bmp")
        all_preds.append(test_preds[-2])
        all_labels.append(test_gts[4])

    ## Rest should be self explanatory
    real_results = torch.cat(all_labels, dim=0).flatten().type(torch.float32).cpu().numpy()  ## Flat tensor
    pred_results = torch.cat(all_preds, dim=0).flatten().type(torch.float32).cpu().detach().numpy()
    train_AUC_ROC = roc_auc_score(real_results, pred_results)
    pred_res = pred_results>=0.1

    TP = sum((pred_res == 1) & (real_results == 1))
    FN = sum((pred_res == 0) & (real_results == 1))
    TN = sum((pred_res == 0) & (real_results == 0))
    FP = sum((pred_res == 1) & (real_results == 0))
    train_sens = TP / (TP + FN)
    train_spec = TN / (TN + FP)
    train_accuracy_calculated = (TN + TP) / (TN + TP + FP + FN)
    train_weighted_acc = (TN / (FP + TN)) * 0.5 + (TP / (TP + FN)) * 0.5

    print("\tSens: {0:.2f}".format(train_sens), "\tSpec: {0:.2f}".format(train_spec),
          "\tAcc: {0:.2f}".format(train_accuracy_calculated),
          '\tWeighted Acc: {0:.2f}'.format(train_weighted_acc),
          "\tAUC: {0:.2f}".format(train_AUC_ROC))

    fig = plt.figure(figsize=(20, 20), dpi=300)
    ax = fig.add_subplot(111)
    colors = ['crimson',
              'orange',
              'mediumpurple',
              'mediumseagreen',
              'steelblue',
              'gold']
    gt_np = real_results

    pred_np = pred_results
    THRESH = 0.1
    # print(gt_np.shape,gt_np,pred_np.shape,pred_np)
    fpr, tpr, thresholds = roc_curve(gt_np, pred_np, pos_label=1)
    colors = ['crimson',
              'orange',
              'mediumpurple',
              'mediumseagreen',
              'steelblue',
              'gold']
    ax.plot(fpr, tpr, lw=5, label='AUC={:.3f}'.format(auc(fpr, tpr)), color=colors[1])

    ax.plot([0, 1], [0, 1], '--', lw=5, color='grey')
    # plt.axis('square')
    plt.xlim([-0.01, 1])
    plt.ylim([-0.01, 1])
    plt.grid(False)
    plt.tick_params(labelsize=25)  # 刻度字体大小13
    plt.xlabel('False Positive Rate', fontsize=30)
    plt.ylabel('True Positive Rate', fontsize=30)
    plt.title('ROC Curve', fontsize=30)
    plt.legend(loc='lower right', fontsize=30)

    # ax.patch.set_facecolor('white')
    # ax.spines['top'].set_visible(True)
    # ax.spines['right'].set_visible(True)
    # ax.spines['bottom'].set_visible(True)
    # ax.spines['left'].set_visible(True)
    plt.savefig('./results_1/MTL_roc_ppa.png')
# Logging
test_results = test_metrics.result()
print('auc_unet', test_results[0])
print('auc_ppa', test_results[1])
print('auc_vcdr', test_results[2])
print('auc', test_results[3])
print('dsc_od', test_results[4])
print('dsc_oc', test_results[5])
print('vCDR_error', test_results[6])
print('fov_error', test_results[7])
