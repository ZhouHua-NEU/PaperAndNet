# 网络基本信息-Basic network information
| 网络名称-Model Name           | Alexnet                                                         |
|-------------------------------|-----------------------------------------------------------------|
| 论文名称-Paper Name           | ImageNet Classification with Deep Convolutional Neural Networks |
| 发表年份-Publish Year         | 2012                                                            |
| 创新结构-Innovative structure | RELU;Dropout;LRN;Maxpool                                                     |
| 数据集-Datasets               | ISLVRC2012                                                      |
| 比赛-Competion                | ImageNet                                                        |
| 改进-Imporve                  | 更深网络结构-Deeper;两块GPU训练-Two GPUs for training           |


# 创新点-Innovation Points
- **RELU激活函数** 2011年，RELU激活函数被提出，2012年Alexnet将其发扬光大，成功解决Sigmoid函数在网络较深时剃度弥散的问题。ReLU函数至今任然在使用。

- **Dropout** 2012年Hinton提出了Dropout方法，为了解决网络过拟合的问题，ALexnet使用Dropout在正向传播时随机损失特定的神经元。虽然现在Dropout已经很少使用，但是这种思想值得借鉴。

- **Maxpool** 将平均池化层改为最大池化层，我将其理解为最大池化层能够使模型关注图像中与预测结果最相关的特征，并且避免了信息丢失和剃度模糊。

- **两块GPU训练** 从论文图片也可以看出，作者使用了两块GPU进行训练，为了加快训练，Alexnet使用两块GTX 580(3GB $\times$ 2)进行训练。

- **LRN层** 使用LRN层，增大对重要信息的敏感度同时减小对非重要信息的敏感度，提高模型的泛化能力，LRN层现在已经变得不再流行。

- **数据增强** 将$256 \times 256$的图像裁剪成$224\times 224$并对图像进行水平旋转，减少过拟合，增加模型泛华能力。

- **RELU Activation Function** In 2011, the RELU activation function was proposed, and in 2012 Alexnet carried it forward, successfully solving the problem of shaving diffusion of the Sigmuid function when the network is deeper. The ReLU function is still in use today.

- **Dropout** In 2012, Hinton proposed the Dropout method, in order to solve the problem of network overfitting, ALexnet used Dropout to randomly lose specific neurons during forward propagation. Although Dropout is rarely used now, this kind of thinking is worth learning.

- **Maxpool** changes the average pooling layer to the largest pooling layer, which I understand as the maximum pooling layer that allows the model to focus on the features in the image that are most relevant to the prediction and avoids information loss and blurred shaving.

- **Two-Piece GPU Training** As can also be seen from the paper picture, the author used two GPUs for training, and to speed up training, Alexnet used two GTX 580 (3GB $times$2) for training.

- **LRN layer** With the LRN layer, increasing the sensitivity to important information while reducing the sensitivity to non-important information, improving the generalization ability of the model, the LRN layer has now become no longer popular.

- **Data Enhancement** Crop an image of $256 times 256$ to $224times 224$ and rotate the image horizontally to reduce overfitting and increase the model's pan-Chinese capability.


# 启示和思考-What we should learn and my think about paper
- **将多个idea进行组合改进** Alexnet的出现，是深度学习的一个转折点和爆发点，从2012年以后，各种网络开始出现，人工智能再一次迎来了春天，从上面的创新点中，RELU和Dropout是组合了别人的创新点，Maxpool是对Avegpool的改进，LRN和数据增强是为了解决模型的泛华能力，其中LRN层从神经科学“侧抑制”(被激活的神经元抑制周围的神经元)中获得启示，两块GPU是为了解决实际的问题。6个创新点中，除了LRN较难想出来之外，其他5个是比较好想的，因此很多时候我们仅仅需要巨人的肩膀上进行组合，改进，优化，就是创新。

- **Combining multiple ideas to improve** The emergence of Alexnet, is a turning point and a flashpoint of deep learning, since 2012, various networks began to appear, artificial intelligence once again ushered in the spring, from the above innovation points, RERU and Dropout are combined with other people's innovation points, Maxpool is an improvement on Avegpool, LRN and data augmentation is to solve the pan-Chinese capabilities of models, in which the LRN layer is "side suppressed" from neuroscience. (Activated neurons inhibit surrounding neurons) to gain enlightenment, two GPUs are meant to solve the actual problem. Among the 6 innovation points, in addition to LRN is more difficult to come up with, the other 5 are easier to think, so many times we just need to combine, improve, and optimize on the shoulders of giants.


