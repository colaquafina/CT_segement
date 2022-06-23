# CT_segement 

#python #unet

这是一个用unet来切割医学图像的代码，感谢李易的鼎力支持和NCC lab的服务器

详细的介绍假期会补上

it use unet to segement the CT pictures 

I would introduce it more specifically after the final exam.

最后的比赛连三等奖都没拿到，简直丢脸到家了，但随缘了，我不会伤心的

## 暑假来了家人们

这个比赛提供给我们的数据是CT图像以及医生标注的结果，根据这些数据训练得到网络后把测试集的CT图像标注出来，并打包成nii.gz的格式

给的数据很怪，不同病人拍的张数不同，有些病人甚至连图片的size也有不同，所以一开始要先处理数据，利用databse把他整成适合训练的数据，代码在data/dataset.py哪里，data文件夹下的三个txt文件是每导入一个训练集就会把训练集的名字加到txt里面，这样子不仅方便打乱数据，也可以节省空间，不需要保存npy文件，而是保存名字

然后就是写unet的代码，在model/unet.py。

训练过程的代码写在train.py里，loss用的是dice loss。我们一开始loss写错了所以训练效果很差，直到ddl前一天才发现，所以没拿到奖。训练时候的参数卸载configure.py里。

predict.py和testing.py均是用于测试模型的训练效果，其中predict.py还有打包测试结果成nii.gz的功能

最后的模型训练效果还不错，我们的训练值进行到了第70个epoches（也就是一半），大部分数据的测试结果都有正确率90以上。
