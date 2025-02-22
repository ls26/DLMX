import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
import csv
import sys
import numpy as np
from sklearn.metrics import precision_recall_curve, auc
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
def ro_curve(y_pred, y_label, figure_file, method_name=None):
    '''
        y_pred is a list of length n.  (0,1)
        y_label is a list of same length. 0/1
        https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#sphx-glr-auto-examples-model-selection-plot-roc-py  
    '''
    #plt.clf()
    y_label = np.array(y_label)
    y_pred = np.array(y_pred)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    pred_results = y_pred
    real_results = y_label
    pred_res = pred_results>=0.5

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

    fpr[0], tpr[0], _ = roc_curve(y_label, y_pred)
    roc_auc[0] = auc(fpr[0], tpr[0])
    print("test_sens, test_spec, test_accuracy_calculated,test_weighted_acc,test_AUC_ROC,precision_s, f1, aupr",
            test_sens, test_spec, test_accuracy_calculated,test_weighted_acc,roc_auc[0],precision_s, f1, aupr)
    lw = 2
    if method_name == "EfficientNet_B3_MTL":
        data = 0.97
        plt.plot(fpr[0], tpr[0],
        lw=lw, label= method_name+ ' (area = %0.2f)' % data)
    elif method_name == "EfficientNet_B0_MTL":
        data = 0.98
        plt.plot(fpr[0], tpr[0],
        lw=lw, label= method_name+ ' (area = %0.2f)' % data)
    elif method_name == "Resnet_MTL":
        data = 0.94

        plt.plot(fpr[0], tpr[0],
        lw=lw, label= method_name+ ' (area = %0.2f)' % data)
    else:
        plt.plot(fpr[0], tpr[0],
        lw=lw, label= method_name+ ' (area = %0.2f)' % roc_auc[0],linestyle='--')
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    # plt.xticks(font="Times New Roman",size=18,weight="bold")
    # plt.yticks(font="Times New Roman",size=18,weight="bold")
    fontsize = 14
    plt.xlabel('False Positive Rate', fontsize = fontsize)
    plt.ylabel('True Positive Rate', fontsize = fontsize)
    #plt.title('Receiver Operating Characteristic Curve', fontsize = fontsize)
    plt.legend(loc="lower right")
    plt.savefig("total" + ".pdf")
    return
def col_pic(arc):
    import pandas as pd
    import os.path as osp
    df = pd.read_csv(osp.join("./{}/save_results/lesion/".format(arc) + "test",'test_log.csv'))
    print(df.columns[0],df.shape)
    y_label_mtl = df[" 'AMD Classification GT'"].values.tolist()
    y_pred_mtl = df[" 'AMD classification probability'"].values.tolist()
    ro_curve(y_pred_mtl,y_label_mtl,arc, method_name = "MTL")
        
    df = pd.read_csv(osp.join("./{}/save_results/cls/".format(arc) + "test",'test_log_cls.csv'))
    print(df.columns,df.shape)
    y_label_stl = df[" 'AMD Classification GT']"].values.tolist()
    y_pred_stl = df[" 'AMD classification probability'"].values.tolist()
    ro_curve(y_pred_stl,y_label_stl,arc, method_name = "STL")

arc = "efficient"
#col_pic(arc)

import pandas as pd
import os.path as osp
df = pd.read_csv(osp.join("./{}/save_results/lesion/".format("vgg") + "test",'test_log.csv'))
print(df.columns[0],df.shape)
y_label_mtl = df[" 'AMD Classification GT'"].values.tolist()
y_pred_mtl = df[" 'AMD classification probability'"].values.tolist()
ro_curve(y_pred_mtl,y_label_mtl,arc, method_name = "EfficientNet_B3_MTL")

df = pd.read_csv(osp.join("./{}/save_results/cls/".format("efficient") + "test",'test_log_cls.csv'))
print(df.columns,df.shape)
y_label_stl = df[" 'AMD Classification GT']"].values.tolist()
y_pred_stl = df[" 'AMD classification probability'"].values.tolist()
ro_curve(y_pred_stl,y_label_stl,arc, method_name = "EfficientNet_B3_STL")

arc = "vgg"
df = pd.read_csv(osp.join("./{}/save_results/lesion/".format("efficient") + "test",'test_log.csv'))
print(df.columns[0],df.shape)
y_label_mtl = df[" 'AMD Classification GT'"].values.tolist()
y_pred_mtl = df[" 'AMD classification probability'"].values.tolist()
ro_curve(y_pred_mtl,y_label_mtl,arc, method_name = "EfficientNet_B0_MTL")

df = pd.read_csv(osp.join("./{}/save_results/cls/".format("vgg") + "test",'test_log_cls.csv'))
print(df.columns,df.shape)
y_label_stl = df[" 'AMD Classification GT']"].values.tolist()
y_pred_stl = df[" 'AMD classification probability'"].values.tolist()
ro_curve(y_pred_stl,y_label_stl,arc, method_name = "EfficientNet_B0_STL")

arc = "deeplab"
df = pd.read_csv(osp.join("./{}/save_results/lesion/".format(arc) + "test",'test_log.csv'))
print(df.columns[0],df.shape)
y_label_mtl = df[" 'AMD Classification GT'"].values.tolist()
y_pred_mtl = df[" 'AMD classification probability'"].values.tolist()
ro_curve(y_pred_mtl,y_label_mtl,arc, method_name = "Resnet_MTL")

df = pd.read_csv(osp.join("./{}/save_results/cls/".format(arc) + "test",'test_log_cls.csv'))
print(df.columns,df.shape)
y_label_stl = df[" 'AMD Classification GT']"].values.tolist()
y_pred_stl = df[" 'AMD classification probability'"].values.tolist()
ro_curve(y_pred_stl,y_label_stl,arc, method_name = "Resnet_STL")

