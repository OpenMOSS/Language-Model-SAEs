#!/usr/bin/env python3
"""
简单的数据库检查脚本
"""

import requests
import json

def check_backend_status():
    """检查后端状态"""
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("✅ 后端服务器正在运行")
            return True
        else:
            print(f"❌ 后端服务器响应异常: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ 后端服务器未运行")
        return False
    except Exception as e:
        print(f"❌ 检查后端状态时出错: {e}")
        return False

def check_dictionary_availability():
    """检查字典可用性"""
    test_dictionaries = [
        "BT4_lorsa_L0A",
        "BT4_tc_L0M",
        "lc0-lorsa-L0",
        "lc0_L0M_16x_k30_lr2e-03_auxk_sparseadam"
    ]
    
    print("\n🔍 检查字典可用性:")
    for dict_name in test_dictionaries:
        try:
            url = f"http://localhost:8000/dictionaries/{dict_name}/features/0"
            response = requests.get(url, headers={"Accept": "application/x-msgpack"}, timeout=10)
            
            if response.status_code == 200:
                print(f"  ✅ {dict_name}: 可用")
            else:
                print(f"  ❌ {dict_name}: HTTP {response.status_code}")
                if response.status_code == 404:
                    print(f"      -> 字典不存在于数据库中")
                elif response.status_code == 500:
                    print(f"      -> 服务器内部错误")
                    
        except Exception as e:
            print(f"  ❌ {dict_name}: 错误 - {e}")

def main():
    print("🧪 简单数据库检查")
    print("=" * 50)
    
    # 检查后端状态
    if not check_backend_status():
        print("\n💡 解决方案:")
        print("1. 启动后端服务器: cd server && python app.py")
        print("2. 或者使用: uvicorn app:app --host 0.0.0.0 --port 8000")
        return
    
    # 检查字典可用性
    check_dictionary_availability()
    
    print("\n💡 如果BT4字典不可用，需要:")
    print("1. 将BT4数据导入到MongoDB数据库中")
    print("2. 确保数据库中有对应的字典名称")
    print("3. 检查数据导入脚本是否正确运行")

if __name__ == "__main__":
    main()
