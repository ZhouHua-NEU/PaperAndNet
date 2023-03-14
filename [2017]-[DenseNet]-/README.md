# 网络基本信息-Basic network information
| 网络名称-Model Name           |                 Densenet                 |
|-------------------------------|:----------------------------------------:|
| 论文名称-Paper Name           | Densely Connected Convolutional Networks |
| 发表年份-Publish Year         |                   2017                   |
| 创新结构-Innovative structure |                 Densenet                 |
| 数据集-Datasets               |            Cifar;Svhn;ImageNet           |
| 比赛-Competion                |                  -------                 |
| 改进-Imporve                  |                  -------                 |

# 创新点-Innovation Points
- **Densenet** 提出了密集残差块结构，复用特征图，减少了参数量。

- **Densenet** proposes a dense residual block structure, reuses feature maps, and reduces the number of parameters.


# 启示和思考-What we should learn and my think about paper
- **Densenet是如何想出来的** 除了文中提到了作者参考其他文章的思想，Densenet最大的思想启迪来自于 Resnet，从resnet的原理中可以很容易的想到Densenet，在深度学习中，那些对称的，递归的结构除了具有空间美之外，也具有良好的效果，因此，尝试着借鉴数学几何，构建具有对称，递归，精妙的网络结构是值得鼓励的。


- **How Densenet came up with** In addition to the author's reference to other articles in the article, Densenet's biggest ideological inspiration comes from Resnet, from the principle of resnet can easily think of Densenet, in deep learning, those symmetrical, recursive structures in addition to spatial beauty, but also have good effects, so trying to learn from mathematical geometry, to build symmetrical, recursive, exquisite network structures is worth encouraging.


# 参考文献-References
code：[https://github.com/weiaicunzai/pytorch-cifar100/blob/master/models/densenet.py](https://github.com/weiaicunzai/pytorch-cifar100/blob/master/models/densenet.py)