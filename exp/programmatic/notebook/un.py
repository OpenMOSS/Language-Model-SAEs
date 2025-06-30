import os
from concurrent.futures import ProcessPoolExecutor

import safetensors.torch
from tqdm import tqdm

# 定义 A 路径
path_A = "/inspire/hdd/global_user/hezhengfu-240208120186/jiaxing_activations/pretraindata-i-2d-801M-ctx8192/blocks.15.hook_resid_post"  # 替换为实际的路径

# 获取所有 safetensors 文件的路径
files = []
for x in range(16):
    for y in range(6112, 12223):
        filename = f"shard-{x}-chunk-{y:08d}.safetensors"
        file_path = os.path.join(path_A, filename)
        if os.path.exists(file_path):
            files.append(file_path)


# 处理单个文件的函数
def process_file(file):
    # 加载 safetensors 文件
    data = safetensors.torch.load_file(file)
    t_tensor = data["tokens"]
    a_tensor = data["activation"]

    if len(t_tensor.shape) == 1:
        t_tensor = t_tensor.unsqueeze(0)

    if len(a_tensor.shape) == 2:
        a_tensor = a_tensor.unsqueeze(0)

    # 将修改后的 tensor 保存到原位置
    # 更新字典中的 'tokens' 键
    data["tokens"] = t_tensor
    data["activation"] = a_tensor

    # 使用 safetensors 保存修改后的文件
    safetensors.torch.save_file(data, file)


# 使用 ProcessPoolExecutor 并行处理文件
with ProcessPoolExecutor(max_workers=64) as executor:
    list(tqdm(executor.map(process_file, files), total=len(files), desc="Processing safetensors files"))
