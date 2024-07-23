import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image

class MyImageDataset(Dataset):
    def __init__(self, length=10):
        self.length = length

    def __len__(self):
        return self.length
    def __getitem__(self, idx):
        # 加载图像并转换为 numpy 数组
        
        image_np = np.random.rand(4, 256, 256) # 假设这里是一个 3 通道的 256x256 图像
        # 将图像缩放到 0 到 255 的范围
        image_np = (image_np * 255).astype(np.uint8)
        return image_np



# 创建 Dataset 和 DataLoader
dataset = MyImageDataset(10)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# 迭代 DataLoader
for batch in dataloader:
    # batch 是一个 numpy 数组
    print(type(batch))  # 输出: <class 'numpy.ndarray'>
    # 检查数据类型
    print(batch.dtype)
