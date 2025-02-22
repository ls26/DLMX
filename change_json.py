import json
# 导入json头文件

import os, sys

# 调用两个函数，更新内容
import pandas as pd
# root_dir = '/mnt/workdir/fengwei/Glaucoma_Fundus_Imaging_Datasets/archive/REFUGE'
#
# ppa_path = '/mnt/workdir/fengwei/Glaucoma_Fundus_Imaging_Datasets/ga/PPA_Train_Dataset.xlsx'
# ppa_label = pd.read_excel(ppa_path,sheet_name="Sheet2")
# ppa_label = ppa_label.fillna(0)
# print(ppa_label.head())
#
# split = "train"
# json_path = os.path.join(root_dir, split, 'index.json')
# # json原文件
# json_path1 = os.path.join(root_dir, split, 'index_c.json')
# # 修改json文件后保存的路径
# image_aug = {} # 先创建一个zidian
# # print(path)
# with open(json_path, 'r') as f:
#     load_dict = json.load(f)
#     num_images = len(load_dict)
#     # "ImgName": "g0001.jpg", "Fovea_X": 1057.95, "Fovea_Y": 1076.52, "Size_X": 2124, "Size_Y": 2056, "Label": 1
#     for image in range(num_images):
#         ImgName = load_dict[str(image)]["ImgName"]
#         ppa = ppa_label[ppa_label["Train"]==ImgName]["PPA"].values
#         # print(ppa[0])
#         Fovea_X = load_dict[str(image)]["Fovea_X"]
#         Fovea_Y = load_dict[str(image)]["Fovea_Y"]
#         Size_X = load_dict[str(image)]["Size_X"]
#         Size_Y = load_dict[str(image)]["Size_Y"]
#         Label = load_dict[str(image)]["Label"]
#         image_dict = {"ImgName": ImgName, "Fovea_X": Fovea_X, "Fovea_Y": Fovea_Y, "Size_X": Size_X,
#                       "Size_Y": Size_Y, "Label": Label,"PPA_label":ppa[0]}
#         print(str(image))
#         image_aug[str(image)] = image_dict
#         # image_aug.append(image_dict) # 依据列表的append对文件进行追加
#
# with open(json_path1, 'w', encoding='utf-8') as file:
#     json.dump(image_aug, file, ensure_ascii=False)
#     # 最后根据json的dump将上面的列表写入文件，得到最终的json文件

# amd_root_dir ='/mnt/workdir/fengwei/AMD_code/iChallenge-AMD-Training400/Training400/AMD/'
# non_amd_root_dir ='/mnt/workdir/fengwei/AMD_code/iChallenge-AMD-Training400/Training400/Non-AMD/'
import os
path2label = "/mnt/sdb/fengwei/AMD_code/Training400/Fovea_location.xlsx"
labels_df = pd.read_excel(path2label)
imgName = labels_df["imgName"]
ids = labels_df.index
### drusen Training400-Lesion\Training400-Lesion\Lesion_Masks\exudate \hemorrhage \others \scar
fullPath2img = []
labels_df["img_label"] = pd.Series(0,index=labels_df.index).astype(float)
labels_df["disc_path"] = pd.Series(0,index=labels_df.index)
labels_df["img_path"] = pd.Series(0,index=labels_df.index)

labels_df["drusen_path"] = pd.Series(0,index=labels_df.index)
labels_df["exudate_path"] = pd.Series(0,index=labels_df.index)
labels_df["hemorrhage_path"] = pd.Series(0,index=labels_df.index)
labels_df["others_path"] = pd.Series(0,index=labels_df.index)
labels_df["scar_path"] = pd.Series(0,index=labels_df.index)

for id_ in ids:
    if imgName[id_][0] == 'A':
        # prefix = 'AMD'
        labels_df.loc[id_,"img_label"] = 1
        img_path = '/mnt/sdb/fengwei/AMD_code/Training400/AMD/'+imgName[id_]
        labels_df.loc[id_, "img_path"] = img_path
    else:
        # prefix = 'Non-AMD'
        labels_df.loc[id_,"img_label"] = 0
        img_path = '/mnt/sdb/fengwei/AMD_code/Training400/Non-AMD/' + imgName[id_]
        labels_df.loc[id_, "img_path"] = img_path

    disc_path = '/mnt/sdb/fengwei/AMD_code/Training400/Disc_Masks/'+ imgName[id_]
    disc_path = disc_path.replace(".jpg",".bmp")
    labels_df.loc[id_,"disc_path"] = disc_path

    drusen_path = ('/mnt/sdb/fengwei/AMD_code/Training400-Lesion/Lesion_Masks/drusen/'+imgName[id_]).replace(".jpg",".bmp")
    exudate_path = ('/mnt/sdb/fengwei/AMD_code/Training400-Lesion/Lesion_Masks/exudate/' + imgName[
        id_]).replace(".jpg", ".bmp")
    hemorrhage_path = ('/mnt/sdb/fengwei/AMD_code/Training400-Lesion/Lesion_Masks/hemorrhage/' + imgName[
        id_]).replace(".jpg", ".bmp")
    others_path = ('/mnt/sdb/fengwei/AMD_code/Training400-Lesion/Lesion_Masks/others/' + imgName[
        id_]).replace(".jpg", ".bmp")
    scar_path = ('/mnt/sdb/fengwei/AMD_code/Training400-Lesion/Lesion_Masks/scar/' + imgName[
        id_]).replace(".jpg", ".bmp")

    if os.path.exists(drusen_path):
        labels_df.loc[id_, "drusen_path"] = drusen_path
    else:
        labels_df.loc[id_, "drusen_path"] = "none"
    if os.path.exists(exudate_path):
        labels_df.loc[id_, "exudate_path"] = exudate_path
    else:
        labels_df.loc[id_, "exudate_path"] = "none"

    if os.path.exists(hemorrhage_path):
        labels_df.loc[id_, "hemorrhage_path"] = hemorrhage_path
    else:
        labels_df.loc[id_, "hemorrhage_path"] = "none"

    if os.path.exists(others_path):
        labels_df.loc[id_, "others_path"] = others_path
    else:
        labels_df.loc[id_, "others_path"] = "none"

    if os.path.exists(scar_path):
        labels_df.loc[id_, "scar_path"] = scar_path
    else:
        labels_df.loc[id_, "scar_path"] = "none"

    # self.fullPath2img.append(os.path.join(path2data, "Training400", prefix, self.imgName[id_]))
labels_df.to_excel('/mnt/sdb/fengwei/AMD_code/Training400/adam_data.xlsx')

# root_dir = '/mnt/workdir/fengwei/Glaucoma_Fundus_Imaging_Datasets/archive/REFUGE'
#
# ppa_path = '/mnt/workdir/fengwei/Glaucoma_Fundus_Imaging_Datasets/ga/PPA_Train_Dataset.xlsx'
# ppa_label = pd.read_excel(ppa_path,sheet_name="Sheet2")
# ppa_label = ppa_label.fillna(0)
# print(ppa_label.head())
#
# split = "train"
# json_path = os.path.join(root_dir, split, 'index.json')
# # json原文件
# json_path1 = os.path.join(root_dir, split, 'index_c.json')
# # 修改json文件后保存的路径
# image_aug = {} # 先创建一个zidian
# # print(path)
# with open(json_path, 'r') as f:
#     load_dict = json.load(f)
#     num_images = len(load_dict)
#     # "ImgName": "g0001.jpg", "Fovea_X": 1057.95, "Fovea_Y": 1076.52, "Size_X": 2124, "Size_Y": 2056, "Label": 1
#     for image in range(num_images):
#         ImgName = load_dict[str(image)]["ImgName"]
#         ppa = ppa_label[ppa_label["Train"]==ImgName]["PPA"].values
#         # print(ppa[0])
#         Fovea_X = load_dict[str(image)]["Fovea_X"]
#         Fovea_Y = load_dict[str(image)]["Fovea_Y"]
#         Size_X = load_dict[str(image)]["Size_X"]
#         Size_Y = load_dict[str(image)]["Size_Y"]
#         Label = load_dict[str(image)]["Label"]
#         image_dict = {"ImgName": ImgName, "Fovea_X": Fovea_X, "Fovea_Y": Fovea_Y, "Size_X": Size_X,
#                       "Size_Y": Size_Y, "Label": Label,"PPA_label":ppa[0]}
#         print(str(image))
#         image_aug[str(image)] = image_dict
#         # image_aug.append(image_dict) # 依据列表的append对文件进行追加
#
# with open(json_path1, 'w', encoding='utf-8') as file:
#     json.dump(image_aug, file, ensure_ascii=False)
#     # 最后根据json的dump将上面的列表写入文件，得到最终的json文件
