# 文档结构-Document structure
感谢来到我的github仓库，这是我复现网络结构的一个仓库，包含了源代码，论文，论文图片以及结构的创新点，仓库的目录结构如下：

Thanks for coming to my github repository, which is a repository for my reproduction of the network structure, containing source code, papers, paper pictures, and innovative points of the structure, the directory structure of the repository is as follows:

$====================================================$  
LeNet5$~~~~~~~~$$~~~~~~~~$$~~~~~~~~$$~~~~~~~~$$~~~~~~~~$模型名字-The name of model    
$~~~~~~~~$│$~~~~~~~~$$~~~~~~~~$$~~~~~~~~$$~~~~~~~~$$~~~~~~~~$$~~~~~~~~$│  
$~~~~~~~~$├─code$~~~~~~~~$$~~~~~~~~$$~~~~~~~~$$~~~~~~~~$$~~~~~~~$├─模型代码-The code of model                     
$~~~~~~~~$│$~~~~~~$└─datasets$~~~~~~~~$$~~~~~~~~$$~~~~~~~~$$~$│$~~~~~~$└─模型数据集-The datasets of model                       
$~~~~~~~~$│$~~~~~~$$~~~~~~$└─mnist_data\ $~~~~~~~~~~~~~$│$~~~~~~$$~~~~~~$└─数据集-The datasets                     
$~~~~~~~~$│$~~~~~~$$~~~~~~$└─datasets.py$~~~~~~~~~~~~~~~$│$~~~~~~$$~~~~~~$└─加载数据代码-The code of loading datasets                                         
$~~~~~~~~$│$~~~~~~$└─model$~~~~~~~~~~~$$~~~~~~~~$$~~~~~~~~$$~$│$~~~~~~$└─模型-The model                              
$~~~~~~~~$│$~~~~~~$$~~~~~~$└─Lenet_summary.txt$~~~~$│$~~~~~$$~~~~~~~$└─模型结构参数-The model detail and parameter of model                          
$~~~~~~~~$│$~~~~~~$$~~~~~~$└─lenet.pth $~~~~~~~~~~~~~~~~~$│$~~~~~$$~~~~~~~$└─预训练模型-Pre-train the model      
$~~~~~~~~$│$~~~~~~$$~~~~~~$└─model.py  $~~~~~~~~~~~~~~~~~$│$~~~~~$$~~~~~~~$└─代码模型-The python code of model      
$~~~~~~~~$│$~~~~~~$└─train.py $~~~~~~~~$$~~~~~~~~$$~~~~~~~~$$~$│$~~~~~~$└─训练文件-The code of train                                   
$~~~~~~~~$│$~~~~~~$└─utils.py $~~~~~~~~~$$~~~~~~~~$$~~~~~~~~$$~$│$~~~~~~$└─实用函数代码-The code of some useful function                                   
$~~~~~~~~$│$~~~~~~~~$$~~~~~~~~$$~~~~~~~~$$~~~~~~~~$$~~~~~~~~$$~~~~~~~~$│           
$~~~~~~~~$├─images$~~~~$$~~~~~~~~$$~~~~~~~~$$~~~~~~~~$$~~~~~~~$├─论文图片-The important images of paper                       
$~~~~~~~~$│$~~~~~~$└─Lenet-1.png$~~~$$~~~~~~~~$$~~~~~~~~$$~$│$~~~~~~$└─论文中的图片-The images of paper                                           
$~~~~~~~~$│$~~~~~~$└─Lenet-5.png$~~~$$~~~~~~~~$$~~~~~~~~$$~$│$~~~~~~$└─论文中的图片-The images of paper                            
$~~~~~~~~$│$~~~~~~~~$$~~~~~~~~$$~~~~~~~~$$~~~~~~~~$$~~~~~~~~$$~~~~~~~~$│               
$~~~~~~~~$├─paper$~~~~~~$$~~~~~~~~$$~~~~~~~~$$~~~~~~~~$$~~~~~~~$├─论文-The origion paper of model                       
$~~~~~~~~$│$~~~~~~$└─LeNet-1.pdf$~~~$$~~~~~~~~$$~~~~~~~~$$~$│$~~~~~~$└─论文中的-The images of paper                       
$~~~~~~~~$│$~~~~~~$└─LeNet-5.pdf$~~~$$~~~~~~~~$$~~~~~~~~$$~$│$~~~~~~$└─论文中的图片-The images of paper                              
$~~~~~~~~$│$~~~~~~~~$$~~~~~~~~$$~~~~~~~~$$~~~~~~~~$$~~~~~~~~$$~~~~~~~~$│                                                 
$~~~~~~~~$├─README.md$~~~~$$~~~~$$~~~~~$$~~~~~~~$$~~~~~~~$├─论文介绍-The introduction of paper   
$~~~~~~~~$│$~~~~~~~~$$~~~~~~~~$$~~~~~~~~$$~~~~~~~~$$~~~~~~~~$$~~~~~~~~$│   
$=====================================================$    


# 数据集-Datasets
由于github上不鼓励上传大文件，因此你可以到官方下载数据集，也可以到我的百度网盘下载数据集，百度网盘的数据在github的基础上添加了数据集文件。[百度网盘](https:ww)

Since uploading large files is not encouraged on github, you can download the dataset to the official or download the dataset to my Baidu network disk, and the data of the Baidu network disk adds a dataset file on the basis of github.