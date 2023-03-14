# 网络基本信息-Basic network information
| 网络名称-Model Name           | Inception-V1                                                         | Inception-V2                                                                                | Inception-V3                                              | Inception-V4                                                                      |
|-------------------------------|----------------------------------------------------------------------|---------------------------------------------------------------------------------------------|-----------------------------------------------------------|-----------------------------------------------------------------------------------|
| 论文名称-Paper Name           | Going deeper with convolutions                                       | Batch Normalization Accelerating Deep Network Training by Reducing Internal Covariate Shift | Rethinking the Inception Architecture for Computer Vision | Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning |
| 发表年份-Publish Year         | 2014                                                                 | 2015                                                                                        | 2015                                                      | 2016                                                                              |
| 创新结构-Innovative structure | Inception                                                            | BN                                                                                          | -------                                                   | Inception-Resnet                                                                  |
| 数据集-Datasets               | ISLVRC2014                                                           | ISLVRC2014                                                                                  | ISLVRC2014                                                | ISLVRC2014                                                                        |
| 比赛-Competion                | ImageNet                                                             | ImageNet                                                                                    | ImageNet                                                  | ImageNet                                                                          |
| 改进-Imporve                  | 多监督,concat,(1,1)卷积-Multi-supervision, concat, (1,1) convolution | -------                                                                                     | 更小卷积核                                                | 一维卷积-One-dimensional convolution                                              |


# 创新点-Innovation Points
- **Inception-V1** Inception-V1采用多监督的方式，使网络能够收敛，更快收敛;Concat结构开创了特征融合开创了先河，为2015年Resnet的出现埋下了伏笔；（1，1）卷积的使用，降低了参数量。 

- **Inception-V2** Inception-V2提出了BN算法，从此成为神经网络的必备要素。
  
- **Inception-V3** Inception-V3使用多个更小的卷积核代替大的卷积核，进一步减少参数量，增大感受野。

- **Inception-V4** Inception-V4使用（1，n），（n，1）的卷积，进一步减少参数量；Inception-Resnet将Inception与Resnet进行了结合。

- **Inception-V1** Inception-V1 uses a multi-supervised approach, enabling the network to converge and converge faster; The Concat structure pioneered feature fusion and set the stage for the emergence of Resnet in 2015; (1, 1) The use of convolution, reducing the amount of parameters. 

- **Inception-V2** Inception-V2 proposed the BN algorithm, which has since become a necessary element of neural networks.
  
- **Inception-V3** Inception-V3 uses multiple smaller convolutional nuclei instead of large convolutional nuclei, further reducing the amount of parameters and increasing the receptive field.

- **Inception-V4** Inception-V4 uses convolution of (1,n), (n,1), further reducing the amount of parameters; Inception-Resnet combines Inception with Resnet.

# 启示和思考-What we should learn and my think about paper
- **更小卷积核意味着看的更仔细** 无论是Zfnet还是VGG，都在用更小的卷积核，因此，更小的卷积核，意味着看得更仔细，感受野更大。
  
- **跳出当下，大胆创新** Inception设计的初衷是为了减小网络参数量，以便能够在工业上应用，因此大胆提出了与传统神经网络不一样的结构，获得了成功。因此创新可以考虑跳出当前的传统，大胆的进行探索。

- **Smaller convolutional kernels mean looking more closely** Both Zfnet and VGG are using smaller convolutional nuclei, so smaller convolutional kernels mean looking more closely and feeling the wilds larger.
  
- **Jump out of the moment, bold innovation** Inception was designed to reduce the amount of network parameters so that it can be applied in industry, so it boldly proposed a structure different from traditional neural networks, and was successful. Therefore, innovation can consider jumping out of the current tradition and boldly exploring.



