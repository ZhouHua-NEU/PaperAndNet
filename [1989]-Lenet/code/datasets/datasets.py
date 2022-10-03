import numpy as np
import torchvision
import gzip
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms

#加载本地数据的类
class LocalDataset(Dataset):
    def __init__(self, folder, data_name, label_name,transform=None):
        (train_set, train_labels) = self.load_data(folder, data_name, label_name)
        self.train_set = train_set
        self.train_labels = train_labels
        self.transform = transform

    def __getitem__(self, index):

        img, target = self.train_set[index], int(self.train_labels[index])
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.train_set)

    def load_data(self,data_folder, data_name, label_name):
        with gzip.open(data_folder+'\\'+label_name, 'rb') as lbpath:
            y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)

        with gzip.open(data_folder+'\\'+data_name, 'rb') as imgpath:
            x_train = np.frombuffer(
                imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28)
        return (x_train, y_train)

#加载本地的数据集
def local_datasets(data_path,train_name,train_label_name,test_name,test_label_name,bt):
    train_dataset = LocalDataset(data_path, train_name,train_label_name,transform=transforms.ToTensor())
    train_loader = DataLoader(dataset=train_dataset,batch_size=bt, shuffle=False,)

    test_dataset = LocalDataset(data_path, test_name,test_label_name,transform=transforms.ToTensor())
    test_loader = DataLoader(dataset=test_dataset,batch_size=bt, shuffle=False,)

    return train_loader,test_loader

#从网络上下载数据集
def download_datasets(bt):
    train_loader = DataLoader(
    torchvision.datasets.MNIST('mnist_data', train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=bt, shuffle=True)

    test_loader = DataLoader(
        torchvision.datasets.MNIST('mnist_data/', train=False, download=True,
                                transform=torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(
                                        (0.1307,), (0.3081,))
                                ])),
        batch_size=bt, shuffle=False)

    return train_loader,test_loader
