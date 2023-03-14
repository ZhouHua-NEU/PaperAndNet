# 网络基本信息-Basic network information
| 网络名称-Model Name           | Resnet                                       |
|-------------------------------|----------------------------------------------|
| 论文名称-Paper Name           | Deep Residual Learning for Image Recognition |
| 发表年份-Publish Year         | 2015                                         |
| 创新结构-Innovative structure | Resnet                                       |
| 数据集-Datasets               | ISLVRC2015                                   |
| 比赛-Competion                | ImageNet                                     |
| 改进-Imporve                  | -------                                      |

# 创新点-Innovation Points
- **Resnet** 提出了残差网络，能够搭建更深的网络。

- **Resnet** Proposed residual networks, enabling deeper networks.


# 启示和思考-What we should learn and my think about paper
- **Resnet深层原理** Resnet固然好用，但我们应该怎么理解其深层原理？一方面， 由于与前一层有连接，剃度从后往前传播过程中，对前面的层也会有一个剃度，防止剃度弥散，难以训练；一方面，从空间上看，每增加一个残差连接，数据的流向就有两种选择，流向残差连接时相当于将残差中间的层短接，此时网络退化到更短的层，相当于并向训练；从特征提取来看，残差可以根据需要添加或除去不必要的特征。

- **Resnet Deep Principle** Resnet is easy to use, but how should we understand its deep principle? On the one hand, due to the connection with the previous layer, during the propagation of the shaving from the back to the front, there will also be a shaving degree on the front layer, preventing the shaving from spreading and difficult to train; On the one hand, from a spatial point of view, every time a residual connection is added, there are two choices for the flow direction of the data, and the flow direction is equivalent to shorting the layer in the middle of the residual when the residual connection is connected, and the network degenerates to a shorter layer, which is equivalent to training to the residual; From the point of view of feature extraction, the residuals can add or remove unnecessary features as needed.