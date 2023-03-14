import os
from glob import glob
import torch
from torchvision import transforms as tr
from torch.autograd import Variable
from torch.utils.data import DataLoader,Dataset
from PIL import Image

#加载本地数据的类
class LocalDataset(Dataset):
    def __init__(self, data_list,data_list_label, data_path,transform=None):
        self.transform = transform
        self.x = data_list
        self.y = data_list_label
        self.data_path = data_path

    def __getitem__(self, index):
        img = Image.open(self.x[index]).convert('RGB')
        target = self.y[index]
        if self.transform is not None:img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.x)



#加载本地的数据集
def local_datasets(data_path,bt,PM,NM=0):
    numclass_path = glob(os.path.join(data_path, '*'))
    numclass_path = [os.path.basename(i) for i in numclass_path]

    train = []
    train_label = []
    val = []
    val_label = []
    for i in numclass_path:
        current_path = glob(os.path.join(data_path, i, '*'))
        l = numclass_path.index(i)
        train_size = int(len(current_path) * 0.7)
        val_size = len(current_path) - train_size
        for j in current_path[:train_size]:
            train.append(j)
            train_label.append(l)
        for k in current_path[train_size:]:
            val.append(k)
            val_label.append(l)


    data_transform = {"train": tr.Compose([tr.RandomResizedCrop(224),tr.RandomHorizontalFlip(),tr.ToTensor(),tr.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
                      "val": tr.Compose([tr.Resize((224, 224)),tr.ToTensor(),tr.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}

    train_dataset = LocalDataset(train,train_label,data_path,transform=data_transform["train"])
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=bt, shuffle=True,num_workers=NM,pin_memory=PM)

    validate_dataset = LocalDataset(val,val_label,data_path,transform=data_transform["val"])
    validate_loader = torch.utils.data.DataLoader(validate_dataset,batch_size=bt, shuffle=True,num_workers=NM,pin_memory=PM)

    return train_loader,validate_loader