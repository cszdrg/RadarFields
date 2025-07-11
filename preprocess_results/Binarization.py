import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import os

# === 配置路径 ===
input_path = "000004.png"       # 输入 FFT PNG 文件
output_path = "fft_binary.png"     # 输出二值化图像路径

# === Step 1: 读取 FFT PNG 图像 ===
image = Image.open(input_path).convert("L")  # 转灰度模式（L: 0-255）
to_tensor = transforms.ToTensor()  # 归一化到 [0,1]
fft_img = to_tensor(image).squeeze(0)  # shape: [H, W]

# === Step 2: 处理 NaN、归一化 ===
fft_img = torch.nan_to_num(fft_img, nan=0.0)
fft_img = fft_img - fft_img.min()
fft_img = fft_img / (fft_img.max() + 1e-6)  # 避免除 0

# === Step 3: 转为 numpy 并做 Otsu 二值化 ===
fft_np = (fft_img.cpu().numpy() * 255).astype(np.uint8)
# ret, binary_mask = cv2.threshold(fft_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
ret , binary_mask = cv2.threshold(fft_np, 80, 255, cv2.THRESH_BINARY)

print(ret)
print(binary_mask)

# === Step 4: 保存二值图像 ===
binary_img = Image.fromarray(binary_mask)
binary_img.save(output_path)
print(f"二值图像保存到: {output_path}")
