import os
import json
import scipy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from scipy.stats import multivariate_normal
from torchvision.transforms import InterpolationMode
class RefugeDataset(Dataset):
    '''
    Loads all data samples once and for all into memory. Can speed up
    the training depending on the computer setup.
    '''
    def __init__(self, root_dir, split='train', output_size=(256,256)):
        # Define attributes
        self.output_size = output_size
        self.root_dir = root_dir
        self.split = split
        
        # Load data index
        with open(os.path.join(self.root_dir, self.split, 'index.json')) as f:
            self.index = json.load(f)
            
        # Sample lists
        self.images = []
        self.labs = []
        self.ods = []
        self.ocs = []
        self.fovs = []
        self.pdfs = []
        
        # Loading
        for k in range(len(self.index)):
            print('Loading {} sample {}/{}...'.format(split, k, len(self.index)), end='\r')
            base_height = self.index[str(k)]['Size_Y']
            base_width = self.index[str(k)]['Size_X']
            
            # Image
            img_name = os.path.join(self.root_dir, self.split, 'images', self.index[str(k)]['ImgName'])
            img = Image.open(img_name).convert('RGB')
            w,h = img.size
            img = transforms.functional.resize(img, self.output_size, interpolation=InterpolationMode.BILINEAR)
            img = transforms.functional.to_tensor(img)
            self.images.append(img)
 

            # Label
            lab = torch.tensor(self.index[str(k)]['Label'], dtype=torch.float32)
            self.labs.append(lab)

            # Seg
            seg_name = os.path.join(self.root_dir, self.split, 'seg', self.index[str(k)]['ImgName'].split('.')[0]+'.bmp')
            seg = np.array(Image.open(seg_name)).copy()
            seg = 255. - seg
            od = Image.fromarray((seg>=127.).astype(np.float32))
            oc = Image.fromarray((seg>=250.).astype(np.float32))
            od = transforms.functional.resize(od, self.output_size, interpolation=InterpolationMode.NEAREST)
            oc = transforms.functional.resize(oc, self.output_size, interpolation=InterpolationMode.NEAREST)
            od = transforms.functional.to_tensor(od)
            oc = transforms.functional.to_tensor(oc)
            self.ods.append(od)
            self.ocs.append(oc)

            
            # Fovea
            f_x = self.index[str(k)]['Fovea_X']/base_width*self.output_size[1]
            f_y = self.index[str(k)]['Fovea_Y']/base_height*self.output_size[0]
            x, y = np.mgrid[0:self.output_size[1]:1, 0:self.output_size[0]:1]
            pos = np.dstack((x, y))
            cov = 50
            rv = multivariate_normal([f_y, f_x], [[cov,0],[0,cov]])
            pdf = rv.pdf(pos)
            pdf = pdf/np.max(pdf)
            pdf = transforms.functional.to_tensor(Image.fromarray(pdf))[0,:,:]
            (f_y, f_x) = scipy.ndimage.measurements.center_of_mass(pdf.numpy())
            fov = torch.FloatTensor([f_x, f_y])
            self.fovs.append(fov)
            self.pdfs.append(pdf)
            
        print('Succesfully loaded {} dataset.'.format(split) + ' '*50)
            
            
    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        # Image
        img = self.images[idx]
    
        # Label
        lab = self.labs[idx]

        # Segmentation masks
        od = self.ods[idx]
        oc = self.ocs[idx]

        # Fovea localization
        fov = self.fovs[idx]
        pdf = self.pdfs[idx]

        return img, [lab, od, oc, (fov, pdf)]
    
class RefugeDataset2(Dataset):
    '''
    Usual on-line loading dataset. More memory efficient.
    '''
    def __init__(self, root_dir, split='train', output_size=(256,256)):
        # Define attributes
        self.output_size = output_size
        self.root_dir = root_dir
        self.split = split
        
        # Load data index
        with open(os.path.join(self.root_dir, self.split, 'index_c.json')) as f:
            self.index = json.load(f)
            
            
    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        base_height = self.index[str(idx)]['Size_Y']
        base_width = self.index[str(idx)]['Size_X']
        
        # Image
        img_name = os.path.join(self.root_dir, self.split, 'Images', self.index[str(idx)]['ImgName'])
        img = Image.open(img_name).convert('RGB')
        w,h = img.size
        img = transforms.functional.resize(img, self.output_size, interpolation=InterpolationMode.BILINEAR)
        img = transforms.functional.to_tensor(np.array(img))
    
        # Label
        lab = torch.tensor(self.index[str(idx)]['Label'], dtype=torch.float32)
        ppa_lab = torch.tensor(self.index[str(idx)]['PPA_label'], dtype=torch.float32)
        # Segmentation masks
        seg_name = os.path.join(self.root_dir, self.split, 'gts', self.index[str(idx)]['ImgName'].split('.')[0]+'.bmp')
        seg = np.array(Image.open(seg_name)).copy()
        seg = 255. - seg
        od = Image.fromarray((seg>=127.).astype(np.float32))
        oc = Image.fromarray((seg>=250.).astype(np.float32))
        od = transforms.functional.resize(od, self.output_size, interpolation=InterpolationMode.NEAREST)
        oc = transforms.functional.resize(oc, self.output_size, interpolation=InterpolationMode.NEAREST)
        od = transforms.functional.to_tensor(np.array(od))
        oc = transforms.functional.to_tensor(np.array(oc))
        
        # Fovea localization
        f_x = self.index[str(idx)]['Fovea_X']/base_width*self.output_size[1]
        f_y = self.index[str(idx)]['Fovea_Y']/base_height*self.output_size[0]
        x, y = np.mgrid[0:self.output_size[1]:1, 0:self.output_size[0]:1]
        pos = np.dstack((x, y))
        cov = 50
        rv = multivariate_normal([f_y, f_x], [[cov,0],[0,cov]])
        pdf = rv.pdf(pos)
        pdf = pdf/np.max(pdf)
        pdf = transforms.functional.to_tensor(np.array(Image.fromarray(pdf)))[0,:,:]
        (f_y, f_x) = scipy.ndimage.measurements.center_of_mass(pdf.numpy())
        fov = torch.FloatTensor([f_x, f_y])

        return img, [lab, od, oc, (fov, pdf), ppa_lab]


class RefugeDataset_test(Dataset):
    '''
    Usual on-line loading dataset. More memory efficient.
    '''

    def __init__(self, root_dir, split='train', output_size=(256, 256)):
        # Define attributes
        self.output_size = output_size
        self.root_dir = root_dir
        self.split = split

        # Load data index
        with open(os.path.join(self.root_dir, self.split, 'index_c.json')) as f:
            self.index = json.load(f)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        base_height = self.index[str(idx)]['Size_Y']
        base_width = self.index[str(idx)]['Size_X']

        # Image
        img_name = os.path.join(self.root_dir, self.split, 'Images', self.index[str(idx)]['ImgName'])
        img = Image.open(img_name).convert('RGB')
        w, h = img.size
        img = transforms.functional.resize(img, self.output_size, interpolation=InterpolationMode.BILINEAR)
        img = transforms.functional.to_tensor(np.array(img))

        # Label
        lab = torch.tensor(self.index[str(idx)]['Label'], dtype=torch.float32)
        ppa_lab = torch.tensor(self.index[str(idx)]['PPA_label'], dtype=torch.float32)
        # Segmentation masks
        seg_name = os.path.join(self.root_dir, self.split, 'gts',
                                self.index[str(idx)]['ImgName'].split('.')[0] + '.bmp')
        seg = np.array(Image.open(seg_name)).copy()
        seg = 255. - seg
        od = Image.fromarray((seg >= 127.).astype(np.float32))
        oc = Image.fromarray((seg >= 250.).astype(np.float32))
        od = transforms.functional.resize(od, self.output_size, interpolation=InterpolationMode.NEAREST)
        oc = transforms.functional.resize(oc, self.output_size, interpolation=InterpolationMode.NEAREST)
        od = transforms.functional.to_tensor(np.array(od))
        oc = transforms.functional.to_tensor(np.array(oc))

        # Fovea localization
        f_x = self.index[str(idx)]['Fovea_X'] / base_width * self.output_size[1]
        f_y = self.index[str(idx)]['Fovea_Y'] / base_height * self.output_size[0]
        x, y = np.mgrid[0:self.output_size[1]:1, 0:self.output_size[0]:1]
        pos = np.dstack((x, y))
        cov = 50
        rv = multivariate_normal([f_y, f_x], [[cov, 0], [0, cov]])
        pdf = rv.pdf(pos)
        pdf = pdf / np.max(pdf)
        pdf = transforms.functional.to_tensor(np.array(Image.fromarray(pdf)))[0, :, :]
        (f_y, f_x) = scipy.ndimage.measurements.center_of_mass(pdf.numpy())
        fov = torch.FloatTensor([f_x, f_y])

        return img, [lab, od, oc, (fov, pdf), ppa_lab], self.index[str(idx)]['ImgName']

class AMDDataset(Dataset):
    '''
    Usual on-line loading dataset. More memory efficient.
    '''

    def __init__(self, feature_list, output_size=(256, 256)):
        # Define attributes
        self.output_size = output_size
        self.feature_list = feature_list

        self.label_x = np.array(self.feature_list)[:,4]
    def __len__(self):
        return len(self.feature_list)

    def __getitem__(self, idx):
        img_name = self.feature_list[idx][1]
        # print(self.feature_list[idx][0],self.feature_list[idx][1],self.feature_list[idx][2],self.feature_list[idx][3])
        img_label = self.feature_list[idx][4]

        base_height = self.feature_list[idx][2]
        base_width = self.feature_list[idx][3]

        disc_path = self.feature_list[idx][5]
        img_path = self.feature_list[idx][6]
        drusen_path = self.feature_list[idx][7]
        exudate_path = self.feature_list[idx][8]

        hemorrhage_path = self.feature_list[idx][9]
        others_path = self.feature_list[idx][10]
        scar_path = self.feature_list[idx][11]
        # print(self.feature_list[idx])
        # print(img_label, base_height, base_width,disc_path,img_path,drusen_path,exudate_path,hemorrhage_path,others_path,scar_path)
        # Image
        img = Image.open(img_path).convert('RGB')
        # print("img_size:",np.array(img).shape)
        # w, h = img.size
        img = transforms.functional.resize(img, self.output_size, interpolation=InterpolationMode.BILINEAR)
        img = transforms.functional.to_tensor(np.array(img))

        disc_seg = np.array(Image.open(disc_path)).copy()
        # print("ori disc size:", np.unique(disc_seg), img_name)
        disc_seg = 255. - disc_seg
        disc_mask = Image.fromarray((disc_seg >= 250.).astype(np.float32))
        disc_mask = transforms.functional.resize(disc_mask, self.output_size,
                                                   interpolation=InterpolationMode.NEAREST)
        disc_mask = transforms.functional.to_tensor(np.array(disc_mask))
        # print("disc_ size:", torch.unique(disc_mask))

        if drusen_path!= "none":
            drusen_seg = np.array(Image.open(drusen_path)).copy()

            drusen_seg = 255. - drusen_seg
            drusen_mask = (drusen_seg >= 250.).astype(np.float32)
            drusen_mask = Image.fromarray(drusen_mask)
            drusen_mask = transforms.functional.resize(drusen_mask, self.output_size, interpolation=InterpolationMode.NEAREST)

            drusen_mask = transforms.functional.to_tensor(np.array(drusen_mask))
            # # print("drusen size:",torch.unique(drusen_mask))
        else:
            drusen_mask = torch.zeros(1, 256, 256)

        if exudate_path != "none":
            exudate_seg = np.array(Image.open(exudate_path)).copy()
            exudate_seg = 255. - exudate_seg
            exudate_mask = (exudate_seg >= 250.).astype(np.float32)
            exudate_mask = Image.fromarray(exudate_mask)
            exudate_mask = transforms.functional.resize(exudate_mask, self.output_size,
                                                       interpolation=InterpolationMode.NEAREST)

            exudate_mask = transforms.functional.to_tensor(np.array(exudate_mask))
        else:
            exudate_mask = torch.zeros(1, 256, 256)

        if hemorrhage_path != "none":
            hemorrhage_seg = np.array(Image.open(hemorrhage_path)).copy()
            hemorrhage_seg = 255. - hemorrhage_seg
            hemorrhage_mask = (hemorrhage_seg >= 250.).astype(np.float32)
            hemorrhage_mask = Image.fromarray(hemorrhage_mask)
            hemorrhage_mask = transforms.functional.resize(hemorrhage_mask, self.output_size,
                                                       interpolation=InterpolationMode.NEAREST)

            hemorrhage_mask = transforms.functional.to_tensor(np.array(hemorrhage_mask))
        else:
            hemorrhage_mask = torch.zeros(1, 256, 256)

        if others_path != "none":
            others_seg = np.array(Image.open(others_path)).copy()
            others_seg = 255. - others_seg
            others_mask = (others_seg >= 250.).astype(np.float32)
            others_mask = Image.fromarray(others_mask)
            others_mask = transforms.functional.resize(others_mask, self.output_size,
                                                        interpolation=InterpolationMode.NEAREST)

            others_mask = transforms.functional.to_tensor(np.array(others_mask))
        else:
            others_mask = torch.zeros(1, 256, 256)

        if scar_path != "none":
            scar_seg = np.array(Image.open(scar_path)).copy()
            scar_seg = 255. - scar_seg
            scar_mask = (scar_seg >= 250.).astype(np.float32)
            scar_mask = Image.fromarray(scar_mask)
            scar_mask = transforms.functional.resize(scar_mask, self.output_size,
                                                       interpolation=InterpolationMode.NEAREST)
            scar_mask = transforms.functional.to_tensor(np.array(scar_mask))
        else:
            scar_mask = torch.zeros(1, 256, 256)


        lab = torch.tensor(img_label, dtype=torch.float32)
        # print(img.shape,disc_mask.shape, drusen_mask.shape, exudate_mask.shape, hemorrhage_mask.shape, others_mask.shape, scar_mask.shape)
        return img, [lab, disc_mask, drusen_mask, exudate_mask, hemorrhage_mask, others_mask, scar_mask]


class AMDDataset_v2(Dataset):
    '''
    Usual on-line loading dataset. More memory efficient.
    '''

    def __init__(self, feature_list, output_size=(256, 256)):
        # Define attributes
        self.output_size = output_size
        self.feature_list = feature_list

        self.label_x = np.array(self.feature_list)[:,4]
    def __len__(self):
        return len(self.feature_list)

    def __getitem__(self, idx):
        img_name = self.feature_list[idx][1]
        # print(self.feature_list[idx][0],self.feature_list[idx][1],self.feature_list[idx][2],self.feature_list[idx][3])
        img_label = self.feature_list[idx][4]

        base_height = self.feature_list[idx][2]
        base_width = self.feature_list[idx][3]

        disc_path = self.feature_list[idx][5]
        img_path = self.feature_list[idx][6]
        drusen_path = self.feature_list[idx][7]
        exudate_path = self.feature_list[idx][8]

        hemorrhage_path = self.feature_list[idx][9]
        others_path = self.feature_list[idx][10]
        scar_path = self.feature_list[idx][11]
        # print(self.feature_list[idx])
        # print(img_label, base_height, base_width,disc_path,img_path,drusen_path,exudate_path,hemorrhage_path,others_path,scar_path)
        # Image
        img = Image.open(img_path).convert('RGB')
        # print("img_size:",np.array(img).shape)
        # w, h = img.size
        img = transforms.functional.resize(img, self.output_size, interpolation=InterpolationMode.BILINEAR)
        img = transforms.functional.to_tensor(np.array(img))

        disc_seg = np.array(Image.open(disc_path)).copy()
        # print("ori disc size:", np.unique(disc_seg), img_name)
        disc_seg = 255. - disc_seg
        disc_mask = Image.fromarray((disc_seg >= 250.).astype(np.float32))
        disc_mask = transforms.functional.resize(disc_mask, self.output_size,
                                                   interpolation=InterpolationMode.NEAREST)
        disc_mask = transforms.functional.to_tensor(np.array(disc_mask))
        # print("disc_ size:", torch.unique(disc_mask))

        if drusen_path!= "none":
            drusen_seg = np.array(Image.open(drusen_path)).copy()

            drusen_seg = 255. - drusen_seg
            drusen_mask = (drusen_seg >= 250.).astype(np.float32)
            drusen_mask = Image.fromarray(drusen_mask)
            drusen_mask = transforms.functional.resize(drusen_mask, self.output_size, interpolation=InterpolationMode.NEAREST)
            drusen_mask =  np.array(drusen_mask)
            # drusen_mask = transforms.functional.to_tensor(np.array(drusen_mask))
            # # print("drusen size:",torch.unique(drusen_mask))
            # print(drusen_mask.shape)
        else:
            drusen_mask = np.zeros((256, 256))

        if exudate_path != "none":
            exudate_seg = np.array(Image.open(exudate_path)).copy()
            exudate_seg = 255. - exudate_seg
            exudate_mask = (exudate_seg >= 250.).astype(np.float32)
            exudate_mask = Image.fromarray(exudate_mask)
            exudate_mask = transforms.functional.resize(exudate_mask, self.output_size,
                                                       interpolation=InterpolationMode.NEAREST)
            exudate_mask = np.array(exudate_mask)
            # print(exudate_mask.shape)
            # exudate_mask = transforms.functional.to_tensor(np.array(exudate_mask))
        else:
            exudate_mask = np.zeros((256, 256))

        if hemorrhage_path != "none":
            hemorrhage_seg = np.array(Image.open(hemorrhage_path)).copy()
            hemorrhage_seg = 255. - hemorrhage_seg
            hemorrhage_mask = (hemorrhage_seg >= 250.).astype(np.float32)
            hemorrhage_mask = Image.fromarray(hemorrhage_mask)
            hemorrhage_mask = transforms.functional.resize(hemorrhage_mask, self.output_size,
                                                       interpolation=InterpolationMode.NEAREST)
            hemorrhage_mask = np.array(hemorrhage_mask)
            # print(hemorrhage_mask.shape)
            # hemorrhage_mask = transforms.functional.to_tensor(np.array(hemorrhage_mask))
        else:
            hemorrhage_mask = np.zeros((256, 256))

        if others_path != "none":
            others_seg = np.array(Image.open(others_path)).copy()
            others_seg = 255. - others_seg
            others_mask = (others_seg >= 250.).astype(np.float32)
            others_mask = Image.fromarray(others_mask)
            others_mask = transforms.functional.resize(others_mask, self.output_size,
                                                        interpolation=InterpolationMode.NEAREST)
            others_mask = np.array(others_mask)
            # others_mask = transforms.functional.to_tensor(np.array(others_mask))
        else:
            others_mask = np.zeros((256, 256))

        if scar_path != "none":
            scar_seg = np.array(Image.open(scar_path)).copy()
            scar_seg = 255. - scar_seg
            scar_mask = (scar_seg >= 250.).astype(np.float32)
            scar_mask = Image.fromarray(scar_mask)
            scar_mask = transforms.functional.resize(scar_mask, self.output_size,
                                                       interpolation=InterpolationMode.NEAREST)
            scar_mask = np.array(scar_mask)
            # scar_mask = transforms.functional.to_tensor(np.array(scar_mask))
        else:
            scar_mask = np.zeros((256, 256))


        mask_all = np.zeros((256, 256))
        mask_all[drusen_mask == 1] = 1
        mask_all[exudate_mask == 1] = 2
        mask_all[hemorrhage_mask == 1] = 3
        mask_all[others_mask == 1] = 4
        mask_all[scar_mask == 1] = 5
        mask_all = transforms.functional.to_tensor(mask_all)
        lab = torch.tensor(img_label, dtype=torch.float32)
        # print(img.shape,disc_mask.shape, drusen_mask.shape, exudate_mask.shape, hemorrhage_mask.shape, others_mask.shape, scar_mask.shape)
        # return img, [lab, disc_mask, drusen_mask, exudate_mask, hemorrhage_mask, others_mask, scar_mask]
        return img, [lab, disc_mask, mask_all]