import torch
import numpy as np
import os
from torch.utils.data import Dataset
import cv2 


class mct_dataset(Dataset):
    def __init__(self, dir, im_list):
        super().__init__()
        self.im_dir = os.path.join(dir, "im")
        self.gt_dir = os.path.join(dir, "gt")
        self.im_list = []
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        with open(im_list) as fh:
            for line in fh:
                self.im_list.append(line)

    def __len__(self):
        return len(self.im_list)


    def __getitem__(self, index):
        im = np.load(os.path.join(self.im_dir, self.im_list[index][:-1] + ".npy"))
        gt = np.load(os.path.join(self.gt_dir, self.im_list[index][:-1] + ".npy"))
        im = cv2.resize(im, dsize=(im.shape[1] // 2, im.shape[0] //2), interpolation=cv2.INTER_CUBIC)
        im = torch.from_numpy(im).unsqueeze(0).to(self.device)
        im = im.type(torch.cuda.FloatTensor)
        gt = torch.from_numpy(gt).unsqueeze(0).to(self.device)        
        gt = gt.type(torch.cuda.FloatTensor)

        return im, gt

train_data = mct_dataset("/media/ncclab/database4/database4/zhichao/koopman_2/real_data_case_ieeg/CT/CT-predict/Dataset/Processed_data/train", "/media/ncclab/database4/database4/zhichao/koopman_2/real_data_case_ieeg/CT/CT-predict/data/train.txt")
val_data = mct_dataset("/media/ncclab/database4/database4/zhichao/koopman_2/real_data_case_ieeg/CT/CT-predict/Dataset/Processed_data/val", "/media/ncclab/database4/database4/zhichao/koopman_2/real_data_case_ieeg/CT/CT-predict/data/val.txt")

class test_dataset(Dataset):
    def __init__(self, dir, im_list):
        super().__init__()
        self.im_dir = os.path.join(dir, "im")
        self.im_list = []
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        with open(im_list) as fh:
            for line in fh:
                self.im_list.append(line)

    def __len__(self):
        return len(self.im_list)


    def __getitem__(self, index):
        # print(os.path.join(self.im_dir, self.im_list[index][:-1] + ".npy"))
        im = np.load(os.path.join(self.im_dir, self.im_list[index][:-1] + ".npy"))
        im = cv2.resize(im, dsize=(im.shape[1] // 2, im.shape[0] //2), interpolation=cv2.INTER_CUBIC)
        im = torch.from_numpy(im).unsqueeze(0).to(self.device)
        im = im.type(torch.cuda.FloatTensor)
        return im

test_data=test_dataset("/media/ncclab/database4/database4/zhichao/koopman_2/real_data_case_ieeg/CT/CT-predict/Dataset/Processed_data/test",'/media/ncclab/database4/database4/zhichao/koopman_2/real_data_case_ieeg/CT/CT-predict/data/test_list.txt')

# print('finish')
