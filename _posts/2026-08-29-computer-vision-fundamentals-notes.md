---
layout: post
title: "计算机视觉基本代码及问题总结"
date: 2026-08-29 22:15:00 +0800
description: "关于计算机视觉、PyTorch、NumPy、Dataset、DataLoader、训练与验证流程的基础代码和常见问题总结。"
tags: [computer-vision, pytorch, numpy, deep-learning]
---

> 本文是个人学习过程中持续整理的计算机视觉基础代码与问题笔记，仅供学习和参考。

# 一次运行多少算力如何计算？4GB显存什么意思？

在原生*python*中，float默认是FP64即双精度浮点数，此时计算机存储的有效位数是64，即$2^{64}$ 
```python
def input_memory_mib(batch: int, channels: int, height: int, width: int, bytes_per_value: int=4) -> float:
	# 一个批次张量本身占用的空间大小
	values = batch * channels * height * width
	# 返回MiB
	# 32bit = 4Byte
	# 1k = 1024
	# 1M = 1024 * 1024
    return values * bytes_per_value / 1024**2
for image_size in [512, 640, 1024, 1280]:

    memory = input_memory_mib(batch=1, channels=3, height=image_size, width=image_size)
	print(f"B=1, RGB, {image_size}×{image_size}: input only {memory:.2f} MiB")
```
# Python容器表达一条检测记录

```python
# 字典形式保存不同的属性
record = {
    "image_id": "synthetic_0001",
    "image_size": (384, 640), # 元组常常保存不会修改的1shape
    "boxes": [[100, 150, 106, 160, 1]],
    "source": "synthetic",
}

# 集合一般用于查重
required_keys = {"image_id", "image_size", "boxes", "source"}
```

# 路径与数据目录

**from pathlib import Path**

*Path* 负责拼接路径，检查存在性和遍历文件，避免手工处理斜杠
```python
# 一般地，可以指定course的源文件根目录，如果不指定就是默认执行路径下的../data
DATA_ROOT = Path(os.getenv("COURSE_DATA_ROOT", "../data"))

candidate_paths = [
	DATA_ROOT / "VisDrone" / "VisDrone2019-DET-val",
	DATA_ROOT / "VisDrone2019-DET-val",
]

existing_paths = [path for path in candidate_paths if path.exists()]

```

# 学习生成合成航拍样例

注意： 设计数组按照\[y, x]索引，框按照\[x1, y2, x2, y2, class_id]保存

```python
def make_aerial_sample(width: int = 640, height: int = 384, n: int = 18, seed: int = 42):
	"""
	参数说明： width，height表示输出图像的宽和高，单位为像素
	n: 表示生成目标的数量
	seed： 独立随机种子
	
	Return: 
		image: uint8数组，每个数值表示[0, 255]的颜色范围，shape=[heigh, width, 3]
		boxes: float32属猪： shaoe=[n, 5], 列为x1,y1,x2,y2,class_id
	"""
	# 设置随机数种子生成器，用于生成随机数，随机种子保证可复现
	rng = np.random.default_rng(seed)
	
	# Numpy图像采用HWC顺序，即高，宽，通道
	# 初始化image
	image = np.zeros((height, width, 3), dtype=np.uint8)
	# 对所有元素广播赋值为RGB
	image[:] = (92, 112, 88)
	# 中间三分之一的行模拟道路
	image[height // 3, 2 * height // 3] = (75, 75, 75)
	
	# 模拟检测框
	boxex = []
	# 生成n个目标
	for index in range(n):
		target_width, target_height = rng.integers(5, 18, size=2)
		# 左上角最大值减去目标尺寸，保证右下角不越界
		x1 = int(rng.integers(0, width - target_width))
		y1 = int(rng.integers(height // 3, 2 * height // 3 - target_height))
		class_id = index % 3 # 保证三类
		x2 = x1 + int(target_width)
		y2 = y1 + int(target_height)
		boxe.append([x1, y1, x2, y2, class_id])
		# 图像切片，先写入y，再写入x
		image[y1:y2, x1:x2] = (180 + class_id * 20, 180, 170)
	return image. np.asarray(boxes, dtype=np.float32)

image, boxes = make_aerial_sample()
print("image shape/dtype:", image.shape, image.dtype)
print("boxes shape/dtype:", boxes.shape, boxes.dtype)
print("first three boxes:\n", boxes[:3])
		
```
❗所有切片和广播操作底层都是numpy，必须将列表转换成np.array
# HWC数组转换为BCHW Tensor

**IMPORT**:  torch
```python
image_tensor = torch.from_numpy(image)
# 归一化numpy是HWC(0, 1, 2) 转换为（2， 0， 1）CHW
image_tensor = image_tensor.permute(2, 0, 1).float() / 255.0
image_tensor = image_tensor.unsequeeze(0) # ====> (B, C, H, W) batch=1

# 检查是否归一化
print("range:", float(image_shape.min()), float(image_shape.max()))
```

# Dataset与DataLoader

首先，一般地，我们为了增强数据集会专门使用transforms来对数据集的图片进行各种增强。
以下为transform.py的模板
```python
from __future__ import annotations

from copy import deepcopy
from typing import Any, Optional

import torch
from torchvision.transforms import v2

def _get_detection_fields(inputs: Any) -> tuple[torch.Tensor, torch.Tensor]:
	"""去除所有需要与无效检测框同步过滤的逐目标字段
	
	label 和 iscrowd 的第一维都与框数量N对应。如果某个框被删除，
	这两个字段必须使用同一布尔掩码过滤，否则模型或评估器会收到错位数据。
	""" 
	# v2.Compose 以 transform(image, target) 形式调用时，inputs 为二元组。
	target = inputs[1]
	return target["labels"], target["iscrowd"]

#===========================分类超参数
# 一般用于分类任务：如判别，语义分割
CLASSIFICATION_TRANSFORM_CONFIG: dict[str, dict[str, Any]] = {
	# 将所有图像直接缩放到固定尺寸，保证分类batch能够torch.stack
	"resize": {
		"enabled": True,
		# (height, width)
		"size": (224, 224),
	},
	# 随机裁剪一块区域后缩放到size，可增强尺度与构图鲁棒性,与resize选用一个
	"random_resize_crop":{
		"enabled": False,
		"size": (224, 224),
		# 随机裁剪区域占原图的面积和比例范围
		"scale": (0.8, 1.0),
		# 裁剪区域的宽高比范围，（3/4，4/3）
		"ratio": (0.75, 1.33333),
	},
	# 水平翻转，适合不存在固定左右语义的自然图像
	"horizontal_flip":{
		"enabled": True,
		# 执行反转的概率
		"probability": 0.5,
	},
	# 旋转：在[-degrees, degrees]内部随机旋转，会在边缘产生填充区域。
	"rotation":{
		"enabled": True,
		# 最大旋转角度，小角度常用于相机姿态轻微变化的场景
		"degrees": 10,
	},
	# 随机扰动颜色，用于模拟光照、相机曝光和成像差异
	"color_jitter":{
		"enabled": True,
		# 亮度变化强度，0.2表示亮度因子大约从[0.8, 1.2]采样
		"brightness": 0.2,
		# 对比度变化强度；数值越大，明暗反差越明显
		"contrast": 0.2,
		# 饱和度变化强度：灰度或非RGB数据不应使用
		"saturation": 0.2,
		# 色相偏移强度，合法范围通常在[0, 0.5]，不宜设置过大
		"hue": 0.05,
	},
	# 按概率转为三通道灰度图，降低模型对颜色信息的依赖
	"grayscale":{
		"enabled": False,
		"probability": 0.1,
	},
	# 高斯模糊，用于模拟失焦、运动或低质量成像
	"gaussian_blur": {
		"enabled": False,
		# 卷积核大小，必须为正奇数，如3， 5， 7
		"kernel_size": 3,
		# 高斯核标准差采样范围：越大越模糊
		"sigma": (0.1, 2.0),
	},
	# normalize, 按通道执行(image - mean)/std。使用预训练权重时应匹配其统计量	
	"normalize":{
		"enabled": True,
		# 使用ImageNet RGB均值和标准差
		"mean": (0.485, 0.456, 0.406),
		"std": (0.229, 0.224, 0.225),	
	},
	# 随机擦除矩阵区域，模拟遮挡；在Tensor化和归一化之后执行
	"random_erasing": {
		"enabled": False,
		# 执行擦除的概率
		"probability": 0.25,
		# 擦除区域占整幅图像面积的比例范围
		"scale": (0.02, 0.20),
		# 擦除区域宽高比范围
		"ratio": (0.3, 3.3),
	},
}

# 一般用于目标检测任务
DETECTION_TRANSFORM_CONFIG: dict[str, dict[str, Any]] = {
	 # 同步缩放图像与检测框。固定二元尺寸会改变原始宽高比。
     # torchvision内部检测模型本身有预处理模块；若交由模型处理可关闭此项。
     # 也可以使用letterbox：不会放缩，而是填充空白区域
     "resize":{
	     "enabled": False,
	     "size": (640, 640),
     },
     "horizontal_flip": {
	     "enabled": True,
	     "probability": 0.5,
     },
     # 上下翻转，低空、道路等方向敏感场景通常保持关闭
     "vertical_flip": {
	     "enabled": False,
	     "probability": 0.5,
     },
     "rotation": {
	     "enabled": False,
	     "degress": 10,
     },
     # 只改变图像的颜色/亮度，不改变检测框位置
     "photemetric_distort": {
	     "enabled": True,
	     "probability": 0.5,
	     # 亮度乘法因子采样范围
	     "brightness": (0.875, 1.125),
	     # 对比度乘法因子的采样范围
	     "contrast": (0.5, 1.5),
	     # 饱和度乘法因子的采样范围
	     "saturation": (0.5, 1.5),
	     # 色相偏移范围，必须在[-0.5, 0.5]内
	     "hue": (-0.05, 0.05),
     },
     # 随机选择与目标具有一定IoU的区域进行裁剪，同时裁剪检测框
     # 小目标容易被裁掉，因此默认关闭，启用后必须清理无效框
     "iou_crop": {
	     "enabled": False,
	     # 裁剪区域相对原图的最小尺度，越小表示允许更激进的裁剪
	     "min_scale": 0.5,
	     "max_scale": 1.0,
	     # 裁剪区域允许的最小宽高比
	     "min_aspect_ratio": 0.75,
	     "max_aspect_ratio": 1.3333,     
     },
     # 对 Tensor 按通道标准化。torchvision内部检测模型内部已有归一化，默认关闭；
     "normalize": {
        "enabled": False,
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
    },
    # 删除裁剪/旋转后越界、退化或过小的框，并同步过滤labels和iscrowd。
    "sanitize_boxes": {
	    "enabled": True,
	    # 框的最小边长（像素），宽或高小于该值则删除
	    "min_size": 1.0,
	    # 框的最小面积，面积小于该值删除
	    "min_area": 1.0,
    },
}

# 构建函数，一般不用修改
def build_classification_transform(
	train: bool,
	config: Optional[dict[str, dict[str, Any]]] = None,
) -> v2.Compose:
	"""
	构建分类变换：用于图片分类任务
	变换顺序：几何变换->颜色变换->Tensor/Float32->标准化->随机擦除
	"""
	cfg = deepcopy(config or CLASSIFICATION_TRANSFORM_CONFIG)
	transforms = []
	
	crop = cfg["random_resized_crop"]
	resize = cfg["resize"]
	if train and crop["enabled"]:
		transforms.append(v2.RandomResizedCrop(crop["size"],
		scale = crop["scale"], ratio=crop["ratio"]))
	elif resize["emanled"]:
		transforms.append(v2.Resize(resize["size"], antialias=True))
	
	if train:
		flip = cfg["horizontal_flip"]
		if flip["enabled"]:
			transforms.append(v2.RandomHorizontalFlip(flip["probability"]))
		
		flip = cfg["vertical_flip:"]
		if flip["enabled"]:
			transforms.append(v2.RandomVerticalFlip(flip["probability"]))
		
		rotation = cfg["rotation"]
		if rotation["enabled"]:
			transforms.append(v2.RandomRotation(rotation["degress"]))
			
		jitter = cfg["color_jitter"]
		if jitter["enabled"]:
			transforms.append(
				v2.ColorJitter(
					jitter["brightness"],
					jitter["constrast"],
					jitter["saturation"],
					jitter["hue"],
				)
			)
		grayscale = cfg["grayscale"]
		if grayscale["enabled"]:
			transforms.append(v2.RandomGrayscale(grayscale["probability"]))
		
		blur = cfg["gaussian_blur"]
		if blur["enabled"]:
			transforms.append(v2.GaussianBlur(blur["kernel_size"], blur["sigma"]))
	 
	transforms.extend([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])

	normalize = cfg["normalize"]
    if normalize["enabled"]:
        transforms.append(v2.Normalize(normalize["mean"], normalize["std"]))
        
    erase = cfg["random_erasing"]
    if train and erase["enabled"]:
        transforms.append(
            v2.RandomErasing(
                erase["probability"], scale=erase["scale"], ratio=erase["ratio"]
            )
        )

    return v2.Compose(transforms)

def build_detection_transform(
    train: bool,
    config: Optional[dict[str, dict[str, Any]]] = None,
) -> v2.Compose:

    """构建目标检测变换。
    Args:
        train: True 启用随机检测增强；False 仅保留确定性预处理。
        config: 可选自定义配置；建议复制默认配置后只修改所需参数。

    变换顺序为：光度扰动 -> IoU 裁剪 -> 翻转/旋转 -> 缩放 -> 清理无效框
    -> Tensor/float32 -> 可选标准化。BoundingBoxes 使用 tv_tensors 表示，因而
    torchvision v2 能对图像和框应用完全一致的随机几何参数。
    """

    cfg = deepcopy(config or DETECTION_TRANSFORM_CONFIG)
    transforms = []

  
    if train:
        distort = cfg["photometric_distort"]
        if distort["enabled"]:
            transforms.append(
                v2.RandomPhotometricDistort(
                    brightness=distort["brightness"],
                    contrast=distort["contrast"],
                    saturation=distort["saturation"],
                    hue=distort["hue"],
                    p=distort["probability"],
                )
            )

        crop = cfg["iou_crop"]
        if crop["enabled"]:
            transforms.append(
                v2.RandomIoUCrop(
                    min_scale=crop["min_scale"],
                    max_scale=crop["max_scale"],
                    min_aspect_ratio=crop["min_aspect_ratio"],
                    max_aspect_ratio=crop["max_aspect_ratio"],
                )
            )

        flip = cfg["horizontal_flip"]
        if flip["enabled"]:
            transforms.append(v2.RandomHorizontalFlip(flip["probability"]))

        flip = cfg["vertical_flip"]
        if flip["enabled"]:
            transforms.append(v2.RandomVerticalFlip(flip["probability"]))

        rotation = cfg["rotation"]
        if rotation["enabled"]:
            transforms.append(v2.RandomRotation(rotation["degrees"]))

    resize = cfg["resize"]
    if resize["enabled"]:
        transforms.append(v2.Resize(resize["size"], antialias=True))

    sanitize = cfg["sanitize_boxes"]
    if sanitize["enabled"]:
        transforms.append(
            v2.SanitizeBoundingBoxes(
                min_size=sanitize["min_size"],
                min_area=sanitize["min_area"],
                labels_getter=_get_detection_fields,
            )
        )

    transforms.extend([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
    normalize = cfg["normalize"]
    if normalize["enabled"]:
        transforms.append(v2.Normalize(normalize["mean"], normalize["std"]))
    return v2.Compose(transforms)

```

## Foundational concept

==Dataset== 和 ==DataLoader== 是处理数据的一对搭档。训练深度学习模型时，我们需要高效地读取海量数据，将它们分组（batch 批次）、打乱顺序（shuffle）并喂给模型。

- **`Dataset` 就像是一个巨大的仓库（或者菜单）**：它知道仓库里总共有多少件货物，也知道如何根据编号准确地取出一件特定的货物。
- **`DataLoader` 就像是一支运输车队**：它负责去仓库（Dataset）里，一次性打包多件货物（Batch），打乱顺序（Shuffle），然后用多辆车并行（多线程/多进程）地把货物运送给模型。
## 程序编写
`Dataset`的核心职责是**定义数据以及如何获取数据**。

==注== PyTorch提供了一个非常强大的视觉工具包`torchvision`。其中用于数据预处理和增强接收的对象就是**PIL**图像格式。用**PIL**读图，后续接上PyTorch的预处理流水线。
- 现实生活中的图片格式五花八门：有的是标准的RGB彩色图(3通道)，也有的是带透明度的PNG(RGBA，4通道)。在`Dataset`中使用`Image.open(img_path).convert('RGB')`, 可以强制将所有图片统一成标准的三通道RGB格式
- 除了读取，它还内置了简易的图像缩放，裁剪和旋转等功能。
- 由于最初的PIL库在2011左右就停止了更新，并且完全不支持Python3，开源社区在它的基础上派生(Fork)出了一个新的项目，取名为*Pillow*，你可以在终端安装：`pip install Pillow`。在调用时可以使用老派写法：`from PIL import Image` 。

==注== Python标准库`typing`: 它的核心作用是为代码引入**类型注解（Type Hinting）**。
- `Optional`可选类型，表示这个参数可以是某种特定的类型，也可以是None。比如：`Optional[x]`实际上等价于`x`或`None`。
- `Callable`可调用对象，表示这是一个可以被调用的对象（通常指函数或实现了`__call__`方法的类）。比如：`Callable[[参数类型1, 参数类型2]，返回值类型]`。如果不严格限制参数和返回值，直接写`Callable`即可。在深度学习通，数据预处理流水线（如`torchvision.transforms`）本质上就是一堆函数。用`Callable`标记它，IDE就知道这个东西可以加括号运行。
- `Any`任意类型，高速检查器这个是任意类型，别管它~，当处理极其复杂的多模态数据输入，或者读取一个包含各种杂乱数据类型的JSON字段时，就可以这样标记。
```python
import json
import os
import csv  # if use
import random
from pathlib import Path
from PIL import Image
from typing import Any, Callable, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import functional as TF

class MyDataset(Dataset):
	# 继承Dataset类
	def __init__(self,
				 data_root: str | Path, 
				anno_file: str | Path, 
				transform: Optional[Callable] = None) -> None:
		self.data_root = data_root  # 数据文件的目录
		self.transform = transform
		
		# 在初始阶段，只读取轻量级标注信息，不把所有图像加载到内存。
		self.sample = self._load_annotations(Path(anno_file))
		
		if not self.samples:
			raise RuntimeError(f"Dataset is empty: {anno_file}")
	
	"""
	举例：读取一个CSV文件，如果是其他类型的文件如txt,只需要更改_load_annotations()函数即可，该函数的位置可以自定义。
	"""
	def _load_annotations(self, path: Path) -> list[dict[str, Any]]:
		"""
		举例：把CSV转换为Dataset内部统一的样本列表。
		"""
		if not path.is_file():
			raise FileNotFoundError(f"File is not EXISTING: {path}")
		
		samples = []
		
		with path.open("r", encoding="utf-8-sig", newline="") as file:
			reader = csv.DictReader(file)
			if not {"image", "label"}.issubset(reader.fieldnames or []):
				raise ValueError("CSV 必须包含 image 和 label 两列")
			
			for line_number, row in enumerate(reader, start=2):
				image_name = (row["image"] or "").strip()
				label_text = (row["lebel"] or "").strip()
				if not image_name or not label_text:
					raise ValueError(f"CSV 第 {line_number} 行 image 或 label 为空")
				image_path = self.data_root / image_name
				if not image_path.is_file():
					raise FileNotFoundError(f"Image is not existing: {image_path}")
				try:
					label = int(label_text)
				except ValueError as exc:
					raise ValueError(f"CSV 第 {line_number} 行 label 必须为整数") from exc
				samples.append({"image_path": iamge_path, "label": label})
		return samples
		
	def __len__(self) -> int:
		return len(self.samples)
	
	def __getitem__(self, index: int) -> dict[str, Any]:
		"""按照索引读取一个样本；图像在此按需加载，避免大量内存占用"""
		sample = self.samples[index]
		with Image.open(sample["image_path"]) as image:
			 # 统一为 RGB，避免灰度图、RGBA 图导致通道数不一致。
			image = image.convert("RGB")
		
		if self.transform is not None:
			image = self.transform(image)
		else:
			# 即使未提供transform，也保证Dataset返回模型可用的float Tensor。
			image = TF.convert_image_dtype(TF.pil_to_tensor(image), torch.float32)		
		
		return {
			"image": image, # Tensor [C H W]
			"label": torch.tensor(sample["label"],dtype=torch.long),
			"image_path": str(sample["image_path"]),
		}
		
def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
	"""将多个样本合成batch"""
	# torch.stack 要求所有图像尺寸相同
	
	return {
		"image": torch.stack([sample["image"] for sample in batch]),
        "label": torch.stack([sample["label"] for sample in batch]),
        "image_path": [sample["image_path"] for sample in batch],
	}
	
def seed_work(worker_id: int) -> None:
	"""让不同 worker可复现且具有互不相同的NumPy/Python随机状态。"""
	del worker_id
	worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)

def build_dataloader(
	dataset: Dataset,
	batch_size: int = 32,
	num_workers: int = 4,
	is_train: bool = True,
	distributed: bool = False,
	seed: int = 42,
) -> tuple[DataLoader, Optional[DistributedSampler]]:
	"""构建单机或 DDP DataLoader。
    Args:
        dataset: 已创建的 Dataset 对象。
        batch_size: 每个进程、每次迭代返回的样本数；DDP 下不是全局 batch size。
        num_workers: 并行读取数据的子进程数。
        is_train: 训练集会打乱并丢弃最后一个不完整 batch；验证集则不会。
        distributed: 是否使用 DistributedSampler。启用前需初始化 DDP 进程组。
        seed: 控制样本打乱顺序和 worker 随机种子的基础种子。
    Returns:
        (loader, sampler)。DDP 训练时每轮调用 ``sampler.set_epoch(epoch)``。
    """
    # DDP 由 sampler 负责分片和打乱，因此设置 sampler 后不能再设置 shuffle=True。
    sampler = (
	    DistributedSampler(dataset, shuffle=is_train, seed=seed)
	    if distributed
	    else None
    )
    
    # 设置随机种子生成器
    generator = torch.Generator().manual_seed(seed)
	
	loader = DataLoader(
		dataset,
		batch_size = batch_size,
		shuffle = is_train and sampler is None.
		sampler = sampler,
		num_workers = num_workers,
		# 仅CUDA训练时锁页内存才有收益，配合 non_blocking=True 传输。
		pin_memory = torch.cuda.is_available(),
		# 训练时保持batch尺寸一致，若数据量小于batch_size，应按需改为False。
		drop_last =is_train,
		collate_fn = collate_fn,
		worker_init_fn = seed_worler,
		# worker常驻可避免每个epoch重复创建进程，提高多轮训练效率
		persistent_workers = num_workers > 0,
	)
	
	return loaderm, sampler

```

**解析**: ` with path.open("r", encoding="utf-8-sig", newline="") as file:`
1. `with ... as file:` 被称为`上下文管理器`。当你打开一个文件后，如果程序中途报错崩溃，文件往往处于未关闭状态，可能导致文件损坏或者内存泄漏。使用`with`之后，无论程序是正常运行结束，还是中途崩溃，都会帮你关闭这个文件，相当于执行了`file.close()`操作
2. 以前可以通过`open(path, ...)`进行文件操作，`pathlib.Path`提供了面向对象的方式操作文件路径
3. `“r”`是只读模式
4. `encoding="utf-8-sig"`: 带签名的UTF-8，在Windows系统中，保存UTF-8文件时，往往会在开头塞入三个不可见的字符，叫做BOM(Byte Order Mark, 字节顺序标记)，如果时Linux系统可以写`utf-8`即可。当python读取这三个字符时，会导致JSON解析崩溃，或者第一个key匹配不上，而`utf-8-sig`可以直接识别并自动剔除这个不可见的BOM头。
5. `newline=""`：表示禁用自动行转换，将换行符的处理权交给后续的解析器。不同操作系统的换行符是不一样的（Windows 是 `\r\n`，Linux/Mac 是 `\n`）。如果不写这个参数，Python 会在读取时自作主张地把所有换行符都统一转成 `\n`。在读取常规 TXT 时这没问题，但如果你在读取 **CSV 文件**（使用 Python 内置的 `csv.reader`），这种自动转换会导致读取出来的数据**莫名其妙多出很多空行**。

# 最小训练逻辑

由上可知：`Dataset`回答“第i个样本是什么”，`DataLoader`负责组成batch、打乱和并行读取。目标数量不同的检测任务通常需要自定义`collate_fn`，不能直接把不同长度的框数组堆成规则张量。

```python
optimizer.zero_grad() # 梯度清0
prediction = model(batch) # 预测值
loss = loss_fn(prediction, target) # 计算损失
loss.backward() # 反向传播
optimizer.step() # 更新参数
```

### 一个epoch训练代码示例
```python
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    sample_count = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        predictions = model(images)
        loss = criterion(predictions, targets)

        if not torch.isfinite(loss):
            raise FloatingPointError(f"出现非有限loss: {loss.item()}")

        loss.backward()
        optimizer.step()

        batch_size = images.shape[0]
        total_loss += loss.item() * batch_size
        sample_count += batch_size

    return total_loss / max(sample_count, 1)


@torch.inference_mode()
def validate(model, loader, criterion, device):
	model.eval()
	total_loss = 0.0
	sample_count = 0
	
	for images, targets in loader:
		images = images.to(device, non_blocking=True)
		targets = targets.to(device, non_blocking=True)
		predictions = model(images)
		loss = criterion(predictions, targets)
	
	  
	
		batch_size = images.shape[0]
		total_loss += loss.item() * batch_size
		sample_count += batch_size
	
	  
	
	return total_loss / max(sample_count, 1)
```

- 将 `@torch.inference_mode()` 作为函数装饰器（Decorator）放在函数开头，完全等价于在函数内部用 `with torch.inference_mode():` 把所有代码包起来。`@torch.inference_mode()`的主要作用是关闭底层求导引擎，推理时，PyTorch 不再构建庞大的计算图，也不再占用显存去保留中间层的激活值（Activations）。
- 第一句紧跟 `model.eval()`。装饰器管底层内存和速度，`eval()` 管上层 Dropout/BatchNorm 的数学行为。==冻结 BatchNorm 等归一化层==：强制模型停止计算当前 Batch 数据的均值和方差，转而使用训练期间累积好的全局均值和方差进行归一化。这防止了模型在遇到极端分布的小 Batch 测试集时发生剧烈波动。 ==旁路 Dropout 等正则化层==：强制 Dropout 失效。在验证和推理阶段，所有神经元必须 100% 参与工作，不再进行随机丢弃和数值缩放。
- **`non_blocking=True`**：在将数据推向 GPU 时，开启非阻塞传输。这意味着 CPU 发出拷贝指令后可以直接去做别的事，不用傻等数据传完。如果 `DataLoader` 配合设置了 `pin_memory=True`（锁页内存），这可以实现 CPU 数据传输与 GPU 计算的异步重叠，压榨性能。
- **`loss.item() * batch_size`**：这是权重还原写法。一般直接用平均值累加 `total_loss += loss.item()`，这会导致当数据集总数不能被 `batch_size` 整除时（通常是最后一个 Batch 数据较少），最终算出的平均 Loss 产生数学偏差通过乘回 `batch_size` 算总和，最后再统一除以总样本数，保证了绝对的严谨。
- **`max(sample_count, 1)`**：这是防御性编程（Defensive Programming）。哪怕传入了一个空的 `loader`，也能防止触发分母为 0 的系统崩溃。
# 常用Numpy操作
## reshape和transpose
`image.shape==(384,640,3)`的三个轴依次是高度、宽度和通道。沿通道求均值得到灰度近似：

```python
gray = image.astype(np.float32).mean(axis=2)  # [384,640] 沿着第三个axis（轴）计算
```
`reshape`改变形状，不改变axis的语义，不能使用reshape(3, 384, 640)完成HWC到CHW的转换，它会把空间像素和通道混在一起，正确写法是：
```python
chw = np.transpose(image, (2, 0, 1))
```
## 广播规则

NumPy从最右侧轴开始比较shape；维度相等或其中一个为1时才可广播。因此`[N,4]+[4]`和`[H,W,3]-[3]`合法。
```python
image_f = image.astype(np.float32) / 255.0
mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
normalized = (image_f - mean) / std
```
## uint8、RGB/BGR和共享内存
`uint8`只能表示0-255，进行减去均值、加噪声和滤波前应转为`float32`
```python
work = image.astype(np.float32) + noise
output = np.clip(work, 0, 255).astype(np.uint8)
```
Pillow通常返回RGB；OpenCV的`cv2.imread`通常返回BGR。直接用Matplotlib显示BGR图会交换红蓝通道：

```python
bgr = cv2.imread(str(image_path))
if bgr is None:
    raise FileNotFoundError(image_path)
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
```

## View(视图) 与 Copy (副本)

- 切片和转置（view机制）： 当你对一个Tensor进行切片操作或者转置时，Pytorch不会在物理内存中复制这些数据，只是创建了一个新的表头。因为它们指向同一块物理内存，修改view会直接改变原数据。
- `.copy()` 时显式的深拷贝，会开辟一个全新的空间，把数据原样复制过去。
### PyTorch 与 NumPy 的零拷贝 (`torch.from_numpy`)

`torch.from_numpy(array)` 会让 PyTorch Tensor 直接接管 NumPy 数组底层的 C 语言内存指针。这是一个**零成本、零拷贝**的操作。仅限于 **CPU** 上的数据。如果你用 OpenCV 读了一张图转成 NumPy array，然后用 `torch.from_numpy()` 转成 Tensor。接着你对这个 Tensor 做了归一化修改，原本的 NumPy array 也会被同步修改。

### 数据类型转换 (`.float()`) 强制重分配
如果调用了tensor.float(), 会重新开辟一个全新的空间，此时逐个计算并写入浮点数，这种方式比较耗时且占用内存。

### 步长 (Stride)、`permute` 与 连续性 (`.contiguous()`)

这是计算机视觉中最常遇到的痛点。在物理内存中，数据永远是一维线性排列的。多维张量（如 `B, C, H, W`）只是通过**步长 (Stride)** 来定义如何在一维内存中跳转以读取不同维度的数据。
- **`permute`（改变步长）**：在图像预处理中，我们经常需要把 `(H, W, C)` 格式转换为 PyTorch 需要的 `(C, H, W)`，此时会调用 `.permute(2, 0, 1)`。这步操作**并没有移动物理内存中的任何一个字节**，它仅仅是调换了读取各个维度的“步长” \[*这句话不理解的复制去问AI*]。
- **非连续状态 (Non-contiguous)**：经过 `permute` 或转置后，数据在逻辑上是相邻的，但在物理内存的线性地址上却不再相邻了，这种状态被称为“非连续的”。
- **`.contiguous()` 的作用**：许多底层的 CUDA 算子（或者 PyTorch 的 `.view()` 操作）要求数据在物理内存中必须是连续排列的，否则会报错。调用 `.contiguous()` 时：
    - 如果数据已经是连续的，它什么也不做（零成本）。
    - 如果数据不连续，它会**开辟一块新内存**，把数据按照新的逻辑顺序重新物理排列一遍，变成连续状态。

# QA
1. 为什么`python -m pip`比直接写`pip`更安全？
=> 明确使用当前解释器附属的`pip`，防止误调用。
2. 为什么`rng.integers(5,18)`不会返回18？
=> Numpy整数随机空间是左闭右开；
3. 图像`image[10,20]`对应几何坐标(x,y)是什么？
=> 对应几何坐标(20, 10)
4. 为什么不能用`reshape(3,H,W)`代替HWC到CHW的`transpose`？
=> reshape只能重新解释线性元素排列，transpose改变的步长，改变了axis的对应关系。
5. `model.eval()`与`torch.inference_mode()`分别改变什么？
=> `eval()`改变Dropout/BatchNorm行为；`inference_mode()`关闭自动求导并降低开销。
6. 为什么检测Dataset常需要自定义`collate_fn`？
=> 每张图目标数量不同，框张量长度不同，默认stack无法组成规则Tensor。
7. 4GB显存下输入过大时，为什么不应第一时间盲目降到很低分辨率？
=> 分辨率直接决定tiny目标像素数；应先降batch、用轻量模型、AMP或梯度累积并记录代价。
8. 为什么验证loss不能代替检测AP？
=>  loss受定义和权重影响；AP还涉及置信度排序、一对一匹配和IoU协议。
9. 训练检查点至少应保存哪些内容？
=> 至少保存epoch、模型、优化器和配置；严格续训还应保存调度器、scaler与随机状态。
