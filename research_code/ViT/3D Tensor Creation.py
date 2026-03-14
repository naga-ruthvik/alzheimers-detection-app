#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import torch
import time
import pandas as pd
import nibabel as nib
import numpy as np
import random
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image

import gc
gc.collect()
torch.cuda.empty_cache()

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'ADNI')
config = {
    'img_size': 192,
    'depth' : 192
}
labels_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'ADNI1_Complete_3Yr_1.5T_9_12_2023.csv')
random.seed(37)

class DataPaths():
    def __init__(self, data_path=None, csv_path=None):
        if data_path==None:
            self.data_path = DATA_PATH
        else:
            self.data_path = data_path

        if csv_path==None:
            self.csv_path = labels_path
        else:
            self.csv_path = csv_path

    def patient_id_loading(self):
        df = pd.read_csv(self.csv_path)
        print("Total number of images: ", len(df))
        cn_mri_scan_list, mci_mri_scan_list, ad_mri_scan_list = [], [], []

        for index, row in df.iterrows():
            image_dict = {}
            # The CSV path is like "data/Extracted\REG-ADNI-AD1.nii", we need to fix the slashes
            # and make it relative to the scripts location
            rel_path = row['mri_path'].replace('\\', '/')
            # path from csv assumes we are in a directory above data
            image_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), rel_path)
            
            image_dict['image_path'] = image_path
            image_dict['patient_id'] = row['subject'].split('.')[0]
            image_dict['image_id'] = row['subject'].split('.')[0]
            
            label = row['diagnosis']
            image_dict['label'] = label

            if label == 'CN':
                cn_mri_scan_list.append(image_dict)
            elif label == 'MCI':
                mci_mri_scan_list.append(image_dict)
            elif label == 'AD':
                ad_mri_scan_list.append(image_dict)

            
        
        random.shuffle(cn_mri_scan_list)
        random.shuffle(mci_mri_scan_list)
        random.shuffle(ad_mri_scan_list)
        no_of_images = {
            'train_cn' : int(len(cn_mri_scan_list)*0.7),
            'train_mci' : int(len(mci_mri_scan_list)*0.7),
            'train_ad' : int(len(ad_mri_scan_list)*0.7),
            'val_cn' : int(len(cn_mri_scan_list)*0.15),
            'val_mci' : int(len(mci_mri_scan_list)*0.15),
            'val_ad' : int(len(ad_mri_scan_list)*0.15),
            'test_cn' : len(cn_mri_scan_list) - int(len(cn_mri_scan_list)*0.7) - int(len(cn_mri_scan_list)*0.15),
            'test_mci' : len(mci_mri_scan_list) - int(len(mci_mri_scan_list)*0.7) - int(len(mci_mri_scan_list)*0.15),
            'test_ad' : len(ad_mri_scan_list) - int(len(ad_mri_scan_list)*0.7) - int(len(ad_mri_scan_list)*0.15)
        }
        print(no_of_images)
        len_train = no_of_images['train_cn'] + no_of_images['train_mci'] + no_of_images['train_ad']
        len_val = no_of_images['val_cn'] + no_of_images['val_mci'] + no_of_images['val_ad']
        len_test = no_of_images['test_cn'] + no_of_images['test_mci'] + no_of_images['test_ad']
        print("Total number of train, validation and test images are {}, {} and {} respectively.".format(len_train, len_val, len_test))
        
        save_path = os.path.join(os.getcwd(), 'data')
        if os.path.exists(save_path)==False:
            os.mkdir(save_path)

        trin_img_df = pd.DataFrame(cn_mri_scan_list[:no_of_images['train_cn']]+\
                                   mci_mri_scan_list[:no_of_images['train_mci']]+\
                                   ad_mri_scan_list[:no_of_images['train_ad']])
        trin_img_df_path = os.path.join(save_path, 'train_mri_scan_list.csv')
        trin_img_df.to_csv(trin_img_df_path, index=False)

        val_img_df = pd.DataFrame(cn_mri_scan_list[no_of_images['train_cn']:no_of_images['train_cn']+no_of_images['val_cn']]+\
                                   mci_mri_scan_list[no_of_images['train_mci']:no_of_images['train_mci']+no_of_images['val_mci']]+\
                                   ad_mri_scan_list[no_of_images['train_ad']:no_of_images['train_ad']+no_of_images['val_ad']])
        val_img_df_path = os.path.join(save_path, 'val_mri_scan_list.csv')
        val_img_df.to_csv(val_img_df_path, index=False)

        test_img_df = pd.DataFrame(cn_mri_scan_list[no_of_images['train_cn']+no_of_images['val_cn']:]+\
                                   mci_mri_scan_list[no_of_images['train_mci']+no_of_images['val_mci']:]+\
                                   ad_mri_scan_list[no_of_images['train_ad']+no_of_images['val_ad']:])
        test_img_df_path = os.path.join(save_path, 'test_mri_scan_list.csv')
        test_img_df.to_csv(test_img_df_path, index=False)

        return trin_img_df_path, val_img_df_path, test_img_df_path
    
    


class ADNIAlzheimerDataset(Dataset):
    def __init__(self, image_df_paths, transform=None):
        self.image_df_paths = image_df_paths
        self.transform = transform
        self.df = pd.read_csv(self.image_df_paths)
        self.desired_width = config['img_size']
        self.desired_height = config['img_size']
        self.desired_depth = config['depth']
        self.transform = transforms.Compose([
                                transforms.RandomHorizontalFlip(0.5),
                                transforms.RandomVerticalFlip(p=0.5),
                                transforms.RandomAffine(15),
                                transforms.ToTensor()
                                #transforms.functional.equalize
                                #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                            ])
        #labels = self.df['label'].values

    def __label_extract(self, group):
        if group=='CN':
            return 0
        elif group=='MCI':
            return 1
        elif group=='AD':
            return 2
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        data = {}
        image_filepath = self.df['image_path'][idx]
        image = nib.as_closest_canonical(nib.load(image_filepath))
        image = image.get_fdata()
        xdim, ydim, zdim = image.shape
        image = np.pad(image, [((256-xdim)//2, (256-xdim)//2), ((256-ydim)//2, (256-ydim)//2), ((256-zdim)//2, (256-zdim)//2)], 'constant', constant_values=0)
        #image = image.reshape(image.shape[2], image.shape[1], image.shape[0])

        width_factor = self.desired_width / image.shape[0]
        height_factor = self.desired_height / image.shape[1]
        depth_factor = self.desired_depth / image.shape[-1]

        image = zoom(image, (width_factor, height_factor, depth_factor), order=1)
        image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2])
        image = image.astype('float32')
        image = torch.from_numpy(image)
        
        label = self.df['label'][idx]
        label = self.__label_extract(label)
        
        return image, label
    


# In[2]:


dataPath = DataPaths()
trin_img_df_path, val_img_df_path, test_img_df_path = dataPath.patient_id_loading()

train_dataset = ADNIAlzheimerDataset(trin_img_df_path)
val_dataset = ADNIAlzheimerDataset(val_img_df_path)
test_dataset = ADNIAlzheimerDataset(test_img_df_path)

def saveTensors(dataset, data_type):
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', '3D_tensors')
    data_path = os.path.join(path, data_type)
    os.makedirs(data_path, exist_ok=True)
    
    labels = {
        0 : 'CN',
        1 : 'MCI',
        2 : 'AD'
    }
    
    for label in labels.keys():
        os.makedirs(os.path.join(data_path, labels[label]), exist_ok=True)
    
    print(f"Processing for {data_type} data is starting. Data will be saved at {data_path}")
    print(f"Total number of images are: {len(dataset)}")
    
    start = time.time()
    for idx in range(len(dataset)):
        tensor, label = dataset.__getitem__(idx)
        tensor_path = f"{data_path}/{labels[label]}/{idx}.pt"
        torch.save(tensor, tensor_path)
        
        if (idx+1)%100==0:
            print(f"{idx+1} images done.")
    
    req_time = time.time() - start
    print(f"Total time required for processing the data is {req_time// 60} minutes {req_time%60} sec.")
    print(f"Processing of a single image took {req_time/(1.0*len(dataset))} sec.")


# In[3]:


saveTensors(test_dataset, 'Test')


# In[4]:


saveTensors(val_dataset, 'Val')


# In[5]:


saveTensors(train_dataset, 'Train')

