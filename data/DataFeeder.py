import os
from posixpath import split
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import random


class DataFeeder:
    
    def __init__(self):
        self.train_dir = "/media/ncclab/database4/database4/zhichao/koopman_2/real_data_case_ieeg/CT/CT-predict/Dataset/Dataset/train/image/" #
        self.label_dir="/media/ncclab/database4/database4/zhichao/koopman_2/real_data_case_ieeg/CT/CT-predict/Dataset/Dataset/train/groundtruth/"
        self.test_dir="/media/ncclab/database4/database4/zhichao/koopman_2/real_data_case_ieeg/CT/CT-predict/Dataset/Dataset/test/"
        self.train_start=0
        self.train_end=79
        self.train_out = "/media/ncclab/database4/database4/zhichao/koopman_2/real_data_case_ieeg/CT/CT-predict/Dataset/Processed_data/train/"
        self.val_out = "/media/ncclab/database4/database4/zhichao/koopman_2/real_data_case_ieeg/CT/CT-predict/Dataset/Processed_data/val/"
        self.test_out = "/media/ncclab/database4/database4/zhichao/koopman_2/real_data_case_ieeg/CT/CT-predict/Dataset/Processed_data/test/"
        


    
    def train_reading(self):
        train_names=os.listdir(self.train_dir)
        out_dir = os.path.join(self.train_out, 'im/')
        fh = open("train.txt", "w")
        train_list=[]
        for f in train_names[self.train_start : self.train_end-10]: 
            print(f)
            train_path=os.path.join(self.train_dir,f)
            temp1=nib.load(train_path)
            temp2=temp1.get_fdata()
            # temp2=temp2.astype(np.int16)
            temp2=np.rollaxis(temp2,2,0)  
            temp2=temp2[40:90,:,:]
            if temp2.shape[1]!=1024:
                pad=np.zeros((50,1024,1024))
                for j in range(pad.shape[0]):
                    pad[j,256:768,256:768]=temp2[j,:,:]
                temp2=pad
            # temp2=temp2.astype(np.int16)
            for j in range(1, temp2.shape[0]):
                path=os.path.join(out_dir, f.split(".")[0] + "-" + str(j+39) + ".npy")
                np.save(path, temp2[j])
                # fh.write(f.split(".")[0] + "-" + str(j+39) + "\n")
                train_list.append(f.split(".")[0] + "-" + str(j+39) + "\n")
        random.shuffle(train_list)
        for i in range(len(train_list)):
            fh.write(train_list[i])
        fh.close()

    def label_reading(self):
        label_list=os.listdir(self.label_dir)
        out_dir = os.path.join(self.train_out, "gt")
        for f in label_list[self.train_start : self.train_end-10]:
            print(f)
            label_list_2=os.path.join(self.label_dir,f)
            label_names=os.listdir(label_list_2)
            for f2 in label_names:
                for j,f2 in zip(range(4),label_names):
                    temp2 = nib.load(os.path.join(label_list_2, f2))
                    temp2 = temp2.get_fdata()
                    # temp2 = temp2.astype(np.int16)
                    temp2=np.rollaxis(temp2,2,0)
                    temp2=temp2[40:90,:,:]
                    if temp2.shape[1]!=1024:
                        pad=np.zeros((50,1024,1024))
                        for j in range(pad.shape[0]):
                            pad[j,256:768,256:768]=temp2[j,:,:]
                        temp2=pad 
                    if j == 0:
                        temp = temp2
                    else:
                        temp += temp2
            # temp=temp.astype(np.int16)
            for j in range(1, temp2.shape[0]):
                np.save(os.path.join(out_dir, f.split(".")[0] + "-" + str(j+39) + ".npy"), temp[j])
            


    def val_train_reading(self):
        train_names=os.listdir(self.train_dir)
        out_dir = os.path.join(self.val_out, "im")
        fh = open("val.txt", "w")
        val_list=[]
        for f in train_names[self.train_end-10:self.train_end]: #f is 00.nii.gz
            print(f)
            train_path=os.path.join(self.train_dir,f)
            temp1=nib.load(train_path)
            temp2=temp1.get_fdata()
            # temp2=temp2.astype(np.int16)
            temp2=np.rollaxis(temp2,2,0)  
            temp2=temp2[40:90,:,:]
            if temp2.shape[1]!=1024:
                pad=np.zeros((50,1024,1024))
                for j in range(pad.shape[0]):
                    pad[j,256:768,256:768]=temp2[j,:,:]
                temp2=pad
            # temp2=temp2.astype(np.int16)
            for j in range(1, temp2.shape[0]):
                np.save(os.path.join(out_dir, f.split(".")[0] + "-" + str(j+39) + ".npy"), temp2[j])
                val_list.append(f.split(".")[0] + "-" + str(j+39) + "\n")
        random.shuffle(val_list)
        for i in range(len(val_list)):
            fh.write(val_list[i])
        fh.close()
        return temp2

    def val_label_reading(self):
        label_list=os.listdir(self.label_dir)
        out_dir = os.path.join(self.val_out, "gt")
        for f in label_list[self.train_end-10:self.train_end]:
            print(f)
            label_list_2=os.path.join(self.label_dir,f)
            label_names=os.listdir(label_list_2)
            for f2 in label_names:
                for j,f2 in zip(range(4),label_names):
                    if 'Parotid' in f2:
                        temp2 = nib.load(os.path.join(label_list_2, f2))
                        temp2 = temp2.get_fdata()
                        # temp2 = temp2.astype(np.int16)
                        temp2=np.rollaxis(temp2,2,0)
                        temp2=temp2[40:90,:,:]
                        if temp2.shape[1]!=1024:
                            pad=np.zeros((50,1024,1024))
                            for j in range(pad.shape[0]):
                                pad[j,256:768,256:768]=temp2[j,:,:]
                            temp2=pad 
                        if j == 0:
                            temp = temp2
                        else:
                            temp += temp2
            # temp=temp.astype(np.int16)
            for j in range(1, temp.shape[0]):
                np.save(os.path.join(out_dir, f.split(".")[0] + "-" + str(j+39) + ".npy"), temp[j])


    def test_reading(self):
        test_names=os.listdir(self.test_dir)
        out_dir = os.path.join(self.test_out, "im")
        fh = open("test_list.txt", "w")
        for f in test_names: #f is 00.nii.gz
            print(f)
            test_path=os.path.join(self.test_dir,f)
            temp1=nib.load(test_path)
            temp2=temp1.get_fdata()
            temp2=np.rollaxis(temp2,2,0)  
            # temp2=temp2[40:90,:,:]
            if temp2.shape[1]!=1024:
                pad=np.zeros((temp2.shape[0],1024,1024))
                for j in range(pad.shape[0]):
                    pad[j,256:768,256:768]=temp2[j,:,:]
                temp2=pad
            temp2=temp2.astype(np.int16)
            for j in range(1, temp2.shape[0]):
                np.save(os.path.join(out_dir, f.split(".")[0] + "-" + str(j) + ".npy"), temp2[j])
                fh.write(f.split(".")[0] + "-" + str(j) + "\n")
        fh.close()
        return temp2
        


a = DataFeeder()
a.train_reading()  
# a.label_reading()
# a.val_train_reading()
# a.val_label_reading()  
# a.test_reading()