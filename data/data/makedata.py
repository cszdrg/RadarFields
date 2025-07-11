from scipy.spatial.transform import Rotation as R
from torchvision import transforms
import torch
from pyproj import Proj, Transformer
from pathlib import Path
import shutil
import numpy as np
import json
import os
import cv2
from PIL import Image

def makepose(imu_orientation_x, 
             imu_orientation_y, 
             imu_orientation_z, 
             imu_orientation_w,
             lon,
             lat,
             alt):
    #生成位姿矩阵
    # 读取四元数
    qx, qy, qz, qw = imu_orientation_x, imu_orientation_y, imu_orientation_z, imu_orientation_w
    # 四元数转旋转矩阵
    r = R.from_quat([qx, qy, qz, qw])
    rotation_matrix = r.as_matrix()  # 3x3

    # 设定一个原点
    transformer = Transformer.from_crs("epsg:4326", "epsg:4978", always_xy=True)
    x, y, z = transformer.transform(lon, lat, alt)
    translation = np.array([x, y, z])  # 平移向量 t
    
    T = np.eye(4)
    T[:3, :3] = rotation_matrix  # 上左角放旋转矩阵 R
    T[:3, 3] = translation 
    
    return T, x, y, z

# ------------------------------
# 删除文件
folder = Path("preprocess_results\\occupancy_component\\preprocess_results")
for item in folder.iterdir():
    if item.is_file():
        item.unlink()              # 删除文件
    elif item.is_dir():
        shutil.rmtree(item)        # 删除子目录及其内容

frames = []
with open("data\\data\\Navtech_Polar.txt", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue  # 跳过空行

        parts = line.split()
        frame_id = int(parts[1])     # "000001" → 1
        timestamp = float(parts[3])  # 时间戳字符串 → float

        frames.append((frame_id, timestamp))
# 700个训练
train_indices = []
# 17个测试
test_indices = []
radar2worlds = []
timestamps_radar = []
# 根据帧名获取位姿矩阵 生成数据
BasePath = "data\\data\\GPS_IMU_Twist"
FFTPath = "data\\seq10\\radar"

xx = []
yy = []
zz = []
index = 0
for frame_id, timestamp in frames:
    PosePath = f"{BasePath}/{frame_id:06d}.txt"
    with open(PosePath, 'r', encoding='utf-8') as f:
        i = 0
        for line in f:
            if not line:
                continue  # 跳过空行
            parts = line.split(',')
            if i == 0:
                lon, lat, alt = float(parts[0]), float(parts[1]), float(parts[2])
            if i == 4:
                imu_orientation_x, imu_orientation_y, imu_orientation_z, imu_orientation_w = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
            i = i + 1
    T, x, y, z = makepose(imu_orientation_x, imu_orientation_y, imu_orientation_z, imu_orientation_w, lon, lat, alt)
    xx.append(x)
    yy.append(y)
    zz.append(z)
    index += 1
    if index > 700:
        test_indices.append(frame_id)
    else:
        train_indices.append(frame_id)
    radar2worlds.append(T.tolist())
    timestamps_radar.append(timestamp)

    filename = f"{frame_id:06d}.png"
    occname = str(timestamp)
    old_path = os.path.join(FFTPath, filename)
    occupancy_path = "preprocess_results\\occupancy_component\\preprocess_results"

    #使用二值化生成occupancy   
    # === Step 1: 读取 FFT PNG 图像 ===
    image = Image.open(old_path).convert("L")  # 转灰度模式（L: 0-255）
    to_tensor = transforms.ToTensor()  # 归一化到 [0,1]
    fft_img = to_tensor(image).squeeze(0)  # shape: [H, W]
    fft_img = fft_img[11:, :] # 去除前 11 行元数据
    fft_img = fft_img.T  # 转置为 [W, H] 以匹配原始数据格式

    # === Step 2: 处理 NaN、归一化 ===
    fft_img = torch.nan_to_num(fft_img, nan=0.0)
    fft_img = fft_img - fft_img.min()
    fft_img = fft_img / (fft_img.max() + 1e-6)  # 避免除 0

    # === Step 3: 转为 numpy 并做 Otsu 二值化 ===
    fft_np = (fft_img.cpu().numpy() * 255).astype(np.uint8)
    # ret, binary_mask = cv2.threshold(fft_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ret , binary_mask = cv2.threshold(fft_np, 80, 255, cv2.THRESH_BINARY)
    binary_mask = binary_mask / 255

    # === Step 4: 保存二值数据 ===
    occupancy_path = os.path.join(occupancy_path, occname)
    binary_mask = binary_mask.astype(np.float32)
    np.save(occupancy_path, binary_mask)

    
    # 修改雷达文件名称为时间戳
    new_filename = str(timestamp) + '.png'
    new_path = os.path.join(FFTPath, new_filename)
    os.rename(old_path, new_path)




offsets = [-np.mean(xx), -np.mean(yy), -np.mean(zz)]
scalers = [np.std(xx), np.std(yy), np.std(zz)]
print("offset = ", offsets)
print("scalers = ", scalers)

data = {
    "test_indices": test_indices,
    "train_indices": train_indices,
    "radar2worlds": radar2worlds,
    "timestamps_radar": timestamps_radar,
    "offsets": offsets,
    "scalers": scalers
}

with open('preprocess_results\\preprocess_results.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2)

print("done")

