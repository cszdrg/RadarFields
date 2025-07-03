from scipy.spatial.transform import Rotation as R
from pyproj import Proj, Transformer
import numpy as np
import json
import os

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
    
    return T

frames = []
with open("RadarFields/pre/Navtech_Polar.txt", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue  # 跳过空行

        parts = line.split()
        frame_id = int(parts[1])     # "000001" → 1
        timestamp = float(parts[3])  # 时间戳字符串 → float

        frames.append((frame_id, timestamp))

train_indices = []
radar2worlds = []
timestamps_radar = []
# 根据帧名获取位姿矩阵 生成数据
BasePath = "RadarFields/pre/GPS_IMU_Twist"
FFTPath = "RadarFields/pre/Navtech_Polar"
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
    T = makepose(imu_orientation_x, imu_orientation_y, imu_orientation_z, imu_orientation_w, lon, lat, alt)
    train_indices.append(frame_id)
    radar2worlds.append(T.tolist())
    timestamps_radar.append(timestamp)
    
    # 修改雷达文件名称为时间戳
    filename = f"{frame_id:06d}.png"
    old_path = os.path.join(FFTPath, filename)
    new_filename = str(timestamp) + '.png'
    new_path = os.path.join(FFTPath, new_filename)
    os.rename(old_path, new_path)

offsets = [0, 0, 0]
scalers = [1, 1, 1]

data = {
    "train_indices": train_indices,
    "radar2worlds": radar2worlds,
    "timestamps_radar": timestamps_radar,
    "offset": offset,
    "scalers": scalers
}

with open('data.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2)

print("done")

