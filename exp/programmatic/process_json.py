import json
import os

from tqdm import tqdm

# 设置A路径，即文件夹的路径
A_path = "/inspire/hdd/global_user/hezhengfu-240208120186/jiaxing_activations/pretraindata-o-2d-801M-ctx8192/blocks.15.hook_resid_post"  # 请将此处替换为实际路径

# 获取所有符合条件的文件
files_to_process = []
for root, dirs, files in os.walk(A_path):
    for file in files:
        if file.endswith(".meta.json") and len(file.split("-")) == 4:
            files_to_process.append(os.path.join(root, file))

# 为文件列表添加进度条
for file_path in tqdm(files_to_process, desc="Processing files", unit="file"):
    try:
        # 读取文件内容
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 将原始数据包装成一个包含该数据的列表
        wrapped_data = [data]

        # 将新的数据写回文件
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(wrapped_data, f, ensure_ascii=False, indent=4)

        print(f"已处理文件: {file_path}")

    except Exception as e:
        print(f"处理文件 {file_path} 时发生错误: {e}")
