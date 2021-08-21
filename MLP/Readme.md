# BaseLine2(修改版)

## 说明 

baseline2本身是多层感知机(MLP)模型,这份代码是从视频课中的jupyternotebook中转换过来的,原来的ipy文件主代码和工具函数都写在一个文件里,不方便查看逻辑,所以这里我把这些代码分开分了,他们分别在以下文件中.
- baseline2.py(主文件)
- model.py(模型和数据读取)
- utls.py(工具函数)

## 运行流程

所需环境:
- pytorch
- tensorboardX
- sklearn


1. 首先运行```data_prepare.py --input_path your\dataset\path```文件,会生成一个```digix-data.hdf```文件,这是数据的预处理,目的是把数据以16bit的格式存储为一个二进制文件,方便主函数直接读取.```INPUT_PATH = '/media/zxl/数据/DIGIX比赛/Dataset'```是数据集的绝对路径.Dataset文件夹下要有testdata和traindata这两个文件夹.
2. 然后运行```baseline2.py```,可用指令如下:

```py
'--input_path',type = str,default='/media/zxl/数据/DIGIX比赛/Dataset',help='The input_path of the dataset'
'--bts_train',type = int,default=100,help = 'The batchsize of train'
'--bts_val',type = int,default = 100,help = 'The batchsize of val'
'--num_workers',type = int,default=10,help='The number of the core for data reading'
'--log_root',type = str,default='./log',help='The folder for saving training logs'
'--chk_root',type = str,default='./checkpoint',help='The root folder for saving checkpoints'
```

3. 训练日志会自动保存到```--log_root```指令对应的文件夹,命名格式是```年-月-日_时_分_秒```,如:```log/2021-08-20_23_12_44```.进入```log_root```文件夹下运行```tensorboard --logdir log/2021-08-20_23_12_44```即可在本地浏览器中查看loss收敛情况.