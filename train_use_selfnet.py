#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by iFantastic on 2019-10-17
# Created by Author: czliuguoyu@163.com
import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from cat12.CatDataset import Cat
from cat12.ResNet import ResNet18


torch.manual_seed(1234)

epochs = 20
batch_size = 128
learn_rate = 1e-3

data_path = os.path.abspath('data')
file_name = 'train_list.txt'

train_data = Cat(data_path, 224, mode='train', filename=file_name)
val_data = Cat(data_path, 224, mode='val', filename=file_name)
test_data = Cat(data_path, 224, mode='test', filename=file_name)

train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=0, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, num_workers=0)
test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=0)


def evaluate(model, loader):

    '''
    验证模型准确率
    :param model: 模型
    :param loader: 验证数据
    :return acc: 准确率
    '''

    correct, total = 0, len(loader.dataset)

    for img, label in loader:

        with torch.no_grad():
            logits = model(img)
            predict = logits.argmax(dim=1)

        correct += torch.eq(predict, label).sum().float().item()

    acc = correct / total

    return acc


def main():

    model = ResNet18(num_class=12)
    # 已存在模型参数
    if os.path.exists(os.path.join(os.path.abspath(''), 'cat.cptk')):
        model.load_state_dict(torch.load('cat.cptk'))

    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=learn_rate, weight_decay=1e-3)
    # 损失函数
    criteon = nn.CrossEntropyLoss()

    best_acc, best_epoch = 0, 0

    for epoch in range(epochs):

        for step, (img, label) in enumerate(train_loader):

            logits = model(img)
            loss = criteon(logits, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 10 == 0:

                print('After {} steps, the loss is {}.'.format(step, loss.item()))

        acc = evaluate(model, val_loader)

        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch
            torch.save(model.state_dict(), 'cat.cptk')
            print('*** Saving the new model in the local. ***')

        print('After {} epochs, the accuracy is {}, the loss is {}'.format(epoch, best_acc, loss.item()))

    print('the best accuracy is {}.\nthe best epoch is {}.'.format(best_acc, best_epoch))

    model.load_state_dict(torch.load('cat.cptk'))
    test_acc = evaluate(model, test_loader)

    print('test accuracy is: ', test_acc)


if __name__ == '__main__':
    main()
