import torch
import numpy as np
from data import dataset
import nibabel as nib
import os
from model import unet
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import cv2



device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
Model= unet.Unet().to(device)
Model.load_state_dict(torch.load('weight.pth'))

test_dataset=DataLoader(dataset.test_data)

dir_test_img='/media/ncclab/database4/database4/zhichao/koopman_2/real_data_case_ieeg/CT/CT-predict/Dataset/Dataset/test'
dir_test_result='/media/ncclab/database4/database4/zhichao/koopman_2/real_data_case_ieeg/CT/CT-predict/Dataset/Processed_data/test/gt'
dir_test_nib='/media/ncclab/database4/database4/zhichao/koopman_2/real_data_case_ieeg/CT/CT-predict/Dataset/Dataset/test_nib'
dir_test_list='/media/ncclab/database4/database4/zhichao/koopman_2/real_data_case_ieeg/CT/CT-predict/data/test_list.txt'
im_list=[]
with open(dir_test_list) as fh:
            for line in fh:
                im_list.append(line)
                
for i,x in zip(range(test_dataset.__len__()),test_dataset):
    result=Model(x)
    x=torch.Tensor.cpu(x)
    result=torch.Tensor.cpu(result)
    x=x.numpy()
    x=np.squeeze(x)
    result=result.detach().numpy()
    result=np.squeeze(result)
    result=cv2.threshold(result,0.9,1024,cv2.THRESH_BINARY)[1]
    np.save(os.path.join(dir_test_result,im_list[i][:-1]+'.npy'),result)

a=['79','80','81','82','83','84','85','86','87','88','89','90','91','92','93']

test_names=os.listdir(dir_test_result)
print(test_names)

for i in a:
    temp=np.zeros((1024,1024,1))
    print(i)
    for f in test_names:
        if(i in f.split('-')[0]):
            print(f)
            temp2=np.load(os.path.join(dir_test_result,f))
            temp2=np.reshape(temp2,[1024,1024,1])
            temp=np.concatenate((temp,temp2),axis=2)
    temp=temp[:,:,1:]
    temp=nib.Nifti1Image(temp,np.eye(4))    
    nib.save(temp,('00'+i+'_seg.nii.gz'))
    
    

        
        
    
