#!/bin/bash

# 遍历./wandb下的所有子文件夹
for dir in ./wandb/*/ ; do
    # 检查目录是否存在（避免无匹配时直接执行）
    if [ -d "$dir" ]; then
        # 提取子文件夹名称（去掉路径和末尾斜杠）
        dir_name=$(basename "$dir")
        
        # 构建完整的wandb路径（格式：wandb/子文件夹名）
        wandb_path="wandb/$dir_name"
        
        echo "正在同步: $wandb_path"
        wandb sync "$wandb_path"
        
        # 检查上一个命令是否成功
        if [ $? -eq 0 ]; then
            echo "✅ 同步成功: $wandb_path"
        else
            echo "❌ 同步失败: $wandb_path"
        fi
        echo "--------------------------------------"
    fi
done

echo "所有任务处理完成"