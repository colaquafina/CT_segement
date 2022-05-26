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

train_dataset=DataLoader(dataset.train_data)

dir_train_list='/media/ncclab/database4/database4/zhichao/koopman_2/real_data_case_ieeg/CT/CT-predict/data/train.txt'
dir_Dataset='/media/ncclab/database4/database4/zhichao/koopman_2/real_data_case_ieeg/CT/CT-predict/Dataset'
dir_train_im='/media/ncclab/database4/database4/zhichao/koopman_2/real_data_case_ieeg/CT/CT-predict/Dataset/Processed_data/train/im'
dir_train_gt='/media/ncclab/database4/database4/zhichao/koopman_2/real_data_case_ieeg/CT/CT-predict/Dataset/Processed_data/train/gt'
im_list=[]
with open(dir_train_list) as fh:
            for line in fh:
                im_list.append(line)
                
def dice_loss(pred, gt):
    intersection = pred * gt
    dice = (2 * intersection.sum() ) / (pred.sum() + gt.sum())
    return dice.mean()
    

for i,(x,gt) in zip(range(100),train_dataset):
    # x=np.load(os.path.join(dir_train_im,i.split('\n')[0]+'.npy'))
    # x=torch.from_numpy(x).unsqueeze(0).to(device)
    pred=Model(x)
    x=torch.Tensor.cpu(x)
    x=x.numpy()
    x=np.squeeze(x)
    pred=torch.Tensor.cpu(pred)
    pred=np.squeeze(pred)
    pred=pred.detach().numpy()
    # gt=np.load(os.path.join(dir_train_gt,i+'.npy'))
    gt=torch.Tensor.cpu(gt)
    gt=gt.numpy()
    gt=np.squeeze(gt)
    loss=dice_loss(pred,gt)
    pred=cv2.threshold(pred,0.9,1024,cv2.THRESH_BINARY)[1]
    plt.subplot(2,2,1)
    plt.imshow(x,cmap='gray')
    plt.subplot(2,2,2)
    plt.imshow(pred,cmap='gray')
    plt.title('loss: '+str(loss))
    plt.subplot(2,2,3)
    plt.imshow(gt,cmap='gray')
    plt.savefig(os.path.join(dir_Dataset,str(i+40)+'.pdf'))
    # print('ok')
    
    
    
    
    