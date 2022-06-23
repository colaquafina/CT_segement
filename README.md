# CT_segement 

#python #unet

这是一个用unet来切割医学图像的代码，感谢李易的鼎力支持和NCC lab的服务器

详细的介绍假期会补上

it use unet to segement the CT pictures 

I would introduce it more specifically after the final exam.

最后的比赛连三等奖都没拿到，简直丢脸到家了，但随缘了，我不会伤心的

## 暑假来了家人们

这个比赛提供给我们的数据是CT图像以及医生标注的结果，根据这些数据训练得到网络后把测试集的CT图像标注出来，并打包成nii.gz的格式

给的数据很怪，不同病人拍的张数不同，有些病人甚至连图片的size也有不同，所以一开始要先处理数据，利用databse把他整成适合训练的数据，代码是

然后就是写unet的代码，这里用的loss function是dice loss。关于这个dice loss，我们一开始写错了，导致训练结果很差，在ddl前一晚上才发现，所以没拿到奖也没啥办法
