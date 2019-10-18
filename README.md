### 百度练习赛（PaddleCamp专场）- 猫十二分类问题

赛题背景

* 本次练习赛主题为图像分类---十二种猫分类。

* 图像分类任务是指将图像主体内容按照所属类别分类。

* 图像分类任务是其他图像任务的基石。

本项目采用的神经网络是**ResNet18**，通过**pytorch**实现ResNet网络，通过交叉训练、数据增强等方式弥补数据不足带来的问题，通过优化，在本地获得了90%的准确率。

网络模型只有1.3兆大小，通过使用ResNet18与训练好的模型，调整全连接层，准确率会更高，但是模型大小增加到48兆，所需的训练时间也更长。

[数据集](https://pan.baidu.com/s/1MgwkIGmVS7YRjiFQjvOvRA&shfl=sharepset)

下载数据集之后，在根目录下创建`data`文件夹，将两个压缩包解压到`data`目录下。

```
# 训练自己的模型
python train_use_selfnet.py

# 使用迁移学习
python train_use_transfer_resnet18.py
```

项目注释比较完整，应该挺友好的。