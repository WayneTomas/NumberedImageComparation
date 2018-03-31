# 图像对比程序说明文档

## 1、运行环境需求

- Windows 10（可选）
- Python 3.5以上（建议直接安装Anaconda，与深度学习相关的所有环境一键安装完毕，就不用自己去配各种依赖了）
- Tensorflow 1.6.0（推荐，自带Keras）
- cuDNN (7.0 on GTX 1070，你的显卡所对应的cuDNN版本不一定和我的版本一样)
- Nvidia CUDA (9.1 on GTX 1070，你的显卡所对应的Nvidia CUDA不一定和我的版本一样)

## 2、运行说明

- 训练所需的数据集在`Pic/`目录下。

- 命令行切换到项目目录下，执行`python train.py -t 训练轮数 -b 每轮的样本batch数 `即可进行训练，每20轮训练会报告一次在验证集上的准确率同时保存当前的模型数据到`Model/`目录下（**模型文件体积特别大，所以注意一下硬盘空间，每次重新训练的时候会删除现有的模型文件**），以便进行预测的时候直接加载使用，同时每20轮还会在工程目录下`model.log`文件中记录一次当前在验证集上的Loss以及准确率的情况，便于训练完毕后模型的选择（**同样的，重新训练会删除此log文件**）；此外，每30轮训练还会报告一次在测试集上的准确率，**默认训练200轮，每轮的batch为128**。**（我这边基本120轮就感觉过Over fitting了）**

  例：

  ```shell
  python train.py -t 200 -b 128
  ```

- 命令行切换到项目目录下，执行`python predict.py -p1 图片1的文件路径 -p2 图片2的文件路径 -m 保存的模型文件名`即可加载保存好的模型并进行预测，输出结果。

  例：

  ```shell
  python predict.py -p1 Pic/2017110310043432010008139732/2017110310043432010008139732-1.png -p2 Pic/2017110310043432010008139732/2017110310043432010008139732-2.png -m Model/1.0.ckpt
  ```

## 3、附录

1. Tensorflow for Windows 安装方法：https://blog.csdn.net/u010099080/article/details/53418159
2. cuDNN下载地址：https://developer.nvidia.com/cudnn（最新的是cuDNN 7.1，在我的机器上只有cuDNN 7.0才能使用）
3. Nvidia CUDA下载地址：https://developer.nvidia.com/cuda-downloads
4. Anaconda下载地址：https://www.anaconda.com/download/
5. Keras中文文档：http://keras-cn.readthedocs.io/en/latest/

## 4、训练结果

**以下是我训练200轮，每轮batch 128的结果：**

>Model 1.0, loss: 0.6177894924626206, acc: 0.6212121212121212
>Model 2.0, loss: 1.4450573926283554, acc: 0.5757575757575758
>Model 3.0, loss: 0.6427941044058764, acc: 0.8181818181818182
>Model 4.0, loss: 0.24792136150327596, acc: 0.9242424242424242
>Model 5.0, loss: 0.12978421625765887, acc: 0.9545454545454546
>Model 6.0, loss: 0.23218894538921164, acc: 0.9393939393939394
>Model 7.0, loss: 0.4515750364710887, acc: 0.8181818181818182
>Model 8.0, loss: 0.515285979949333, acc: 0.8484848484848485
>Model 9.0, loss: 0.2546916030855341, acc: 0.9393939393939394
>Model 10.0, loss: 0.7277257641103598, acc: 0.8939393939393939