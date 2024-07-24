### Med-VQA

#### 一、总体思路

调研了一下现有文献，改进点主要在一下几个方面：

##### 1、Image Encoder

图像编码器要学习到更丰富的解剖结构特征，关注于key regions。

现有的方法：

1）detection model来训练image encoder（weights fixed）

2）image-text alignment，基本都是clip的方式计算similarity，对比训练。

我打算采用mae这种掩码预训练（pretraining and finetuning）。]

##### 2、Feature Difference Learning

现有方法：

1）raw images相减 

2）global features相减

我觉得可以考虑层次化特征相减，逐层diff。另外ICCV 2021有一篇paper将CAM作为先验。

##### 3、Text Prompt

现有方法：

1）引入keyword，构建dict和candidate是一个不错的思路，但是开销太大了，需要人为对关键词进行filtering。

2）

我打算设计一个multi-stage（or multi-crop）的方法，所谓multi-stage，就是先不直接回答difference。先回答main和ref有什么，然后依据这个进一步回答difference。

multi-crop就是detect and crop每个解剖结构区域，和originial image一起作为输入。

#### 二、代码实现

##### 1、MAE Pretraining

参考两个repo：

https://github.com/facebookresearch/mae

https://github.com/lambert-x/medical_mae



