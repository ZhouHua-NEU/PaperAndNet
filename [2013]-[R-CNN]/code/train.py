from torch import nn
from torch.nn import functional as F
from torch import optim
import torch,torchvision
from tqdm import  tqdm

from model.model import *
from utils import plot_image, plot_curve, one_hot
from dataset.datasets import *


import argparse
parser = argparse.ArgumentParser()

#model
parser.add_argument('--model_name', default='Googlenet', type=str)
parser.add_argument('--numclass',default=5,type=int)
parser.add_argument('--model_loadpath', default='model/model.pth', type=str)
parser.add_argument('--model_savepath', default='model/model.pth', type=str)
parser.add_argument('--Inception',default='InceptionV1',type=str,choices=['InceptionV1','InceptionV2','InceptionV2','InceptionV4','InceptionResnet'])

#optimiter
parser.add_argument('--epochs', default=2, type=int, metavar='N',help='number of total epochs to run')
parser.add_argument('--batchsize', default=4, type=int,metavar='N', help='batch size ')
parser.add_argument('--learning_rate', default=0.0001, type=float,metavar='LR', help='initial learning rate ')
parser.add_argument('--weight_decay', default=1e-5, type=float,metavar='W', help='weight decay ')
parser.add_argument('--device', default=("cuda" if torch.cuda.is_available() else "cpu"), type=str)

#datasets
parser.add_argument('--data_path',default='datasets/flower_photos/',type=str)
parser.add_argument('--pin_memory',default=(True if torch.cuda.is_available() else False),type=bool)

args = parser.parse_args()

device = torch.device(args.device)

train_loader,test_loader= local_datasets(args.data_path,args.batchsize,args.pin_memroy)

#查看图片和标签
x, y = next(iter(train_loader))
print(x.shape,y.shape)
plot_image(x, y)

#定义网络并进行训练
net = googlenet(args.Inception,args.numclass)
#net = torch.load(args.model_loadpath,map_location=device)
from torchinfo import summary
with open("model/%s_summary.txt"%args.Inception, "w", encoding='utf-8') as f:
    f.write(str(summary(net, input_size=(args.batchsize, 3, 32, 32))))
    f.close()

optimizer = optim.Adam(net.parameters(), lr=args.learning_rate,weight_decay=args.weight_decay)
train_loss = []
less_loss = 1000
from torch.optim import lr_scheduler
scheduler = lr_scheduler.StepLR(optimizer,step_size=2,gamma = 0.6)
loss_function = nn.CrossEntropyLoss()
pbar=tqdm(range(args.epochs))
for epoch in pbar:
    one_epoch_total_loss = 0.0
    for batch_idx, (x, y) in enumerate(train_loader):
        out = net(x).requires_grad_(True)
        y_onehot = one_hot(y, args.numclass)
        loss = loss_function(out.to(device), y_onehot.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        one_epoch_total_loss += loss
        train_loss.append(loss.item())
        print(batch_idx,'/',len(train_loader),'average loss:',(one_epoch_total_loss/batch_idx).item())
    pbar.set_description("epoch %d total loss : %s" %(epoch,round(one_epoch_total_loss.item(),2)))
    scheduler.step()
    if(one_epoch_total_loss<less_loss):
        less_loss=one_epoch_total_loss
        torch.save(net,args.model_savepath)
        print(less_loss)
    if(less_loss<=0.01):
        break
#画出损失函数图
plot_curve(train_loss)

#计算正确率
total_correct = 0
for x,y in test_loader:
    out = net(x)
    pred = out.argmax(dim=1)
    correct = pred.eq(y).sum().float().item()
    total_correct += correct

total_num = len(test_loader.dataset)
acc = total_correct / total_num
print('test acc:', acc)


#画出使用模型预测的结果
x, y = next(iter(test_loader))
pred = net(x).argmax(dim=1)
plot_image(x,y,pred)