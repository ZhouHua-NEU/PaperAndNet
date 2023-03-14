import torch
from matplotlib import pyplot as plt
from math import ceil

#画损失函数图
def plot_curve(data):
    fig = plt.figure()
    plt.plot(range(len(data)), data, color='blue')
    plt.legend(['value'], loc='upper right')
    plt.xlabel('step')
    plt.ylabel('value')
    plt.show()


#画出每一个batchsize中的图片
def plot_image(img,label,pred=None):
    fig = plt.figure()
    for i in range(len(img)):
        plt.subplot(ceil(len(img)**0.5), ceil(len(img)**0.5), i + 1)
        plt.tight_layout()
        plt.imshow(img[i].transpose(0,1).transpose(1,2))
        if(pred is not None):plt.title("true:{}predict:{}".format(label[i].item(),pred[i].item()))
        else:plt.title("true:{}".format(label[i].item()))
        plt.xticks([])
        plt.yticks([])
    plt.show()


def one_hot(label, depth):
    out = torch.zeros(label.size(0), depth)
    idx = torch.LongTensor(label).view(-1, 1)
    out.scatter_(dim=1, index=idx, value=1)
    return out