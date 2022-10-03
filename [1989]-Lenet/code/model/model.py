import torch.nn as nn


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2),  # 28x28x1-->28x28x6
            nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 14x14x6
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0),  # 10x10x16
            nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 5x5x16
        )

        self.fc = nn.Sequential(
            nn.Linear(400, 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, 10),
        )

    def forward(self, x):
        y = self.conv(x)
        output = self.fc(y.view(x.shape[0], -1))
        return output