#!/usr/bin/env python3
"""
Clerp合并工具 - 合并两个circuit JSON文件中的clerp字段

功能：
- 比较两个JSON文件中相同node_id的节点
- 当其中一个节点的clerp字段为空（""或null），另一个不为空时，使用非空的clerp填充
- 直接更新两个原始文件，使它们都包含合并后的clerp数据

使用方法：
python merge_clerp.py file1.json file2.json
"""

import json
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional


def load_json_file(file_path: str) -> Dict[str, Any]:
    """加载JSON文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✅ 成功加载文件: {file_path}")
        return data
    except FileNotFoundError:
        print(f"❌ 文件未找到: {file_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"❌ JSON解析错误 {file_path}: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 加载文件失败 {file_path}: {e}")
        sys.exit(1)


def get_nodes_from_data(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """从数据中提取节点列表"""
    if 'nodes' in data and isinstance(data['nodes'], list):
        return data['nodes']
    elif isinstance(data, list):
        return data
    else:
        # 尝试其他可能的数组属性
        possible_keys = ['data', 'features', 'items', 'activations']
        for key in possible_keys:
            if key in data and isinstance(data[key], list):
                return data[key]
        
        # 尝试Object.values()
        for value in data.values():
            if isinstance(value, list):
                return value
    
    print("❌ 无法在数据中找到节点数组")
    return []


def is_empty_clerp(clerp: Any) -> bool:
    """检查clerp是否为空"""
    return clerp is None or clerp == "" or (isinstance(clerp, str) and clerp.strip() == "")


def merge_clerp_fields(nodes1: List[Dict], nodes2: List[Dict]) -> tuple[List[Dict], List[Dict], Dict[str, Any]]:
    """合并两个节点列表的clerp字段，双向更新"""
    # 创建node_id到节点的映射
    nodes1_map = {node.get('node_id'): node for node in nodes1 if 'node_id' in node}
    nodes2_map = {node.get('node_id'): node for node in nodes2 if 'node_id' in node}
    
    print(f"📊 文件1包含 {len(nodes1_map)} 个节点")
    print(f"📊 文件2包含 {len(nodes2_map)} 个节点")
    
    # 找到共同的node_id
    common_node_ids = set(nodes1_map.keys()) & set(nodes2_map.keys())
    print(f"🔍 找到 {len(common_node_ids)} 个相同的node_id")
    
    merge_stats = {
        'total_common': len(common_node_ids),
        'merged_count': 0,
        'file1_to_file2': 0,
        'file2_to_file1': 0,
        'both_empty': 0,
        'both_non_empty': 0,
        'merged_details': []
    }
    
    # 创建深拷贝以避免修改原始数据
    merged_nodes1 = [node.copy() for node in nodes1]
    merged_nodes2 = [node.copy() for node in nodes2]
    
    # 重新创建映射
    merged_nodes1_map = {node.get('node_id'): node for node in merged_nodes1 if 'node_id' in node}
    merged_nodes2_map = {node.get('node_id'): node for node in merged_nodes2 if 'node_id' in node}
    
    for node_id in common_node_ids:
        node1 = merged_nodes1_map[node_id]
        node2 = merged_nodes2_map[node_id]
        
        clerp1 = node1.get('clerp')
        clerp2 = node2.get('clerp')
        
        empty1 = is_empty_clerp(clerp1)
        empty2 = is_empty_clerp(clerp2)
        
        if empty1 and empty2:
            merge_stats['both_empty'] += 1
        elif not empty1 and not empty2:
            merge_stats['both_non_empty'] += 1
            if clerp1 != clerp2:
                print(f"⚠️  节点 {node_id} 两个文件都有不同的clerp:")
                print(f"   文件1: {clerp1[:50]}...")
                print(f"   文件2: {clerp2[:50]}...")
                print(f"   保持各自的值")
        elif empty1 and not empty2:
            # 文件1为空，文件2不为空，用文件2的值填充文件1
            node1['clerp'] = clerp2
            merge_stats['file2_to_file1'] += 1
            merge_stats['merged_count'] += 1
            merge_stats['merged_details'].append({
                'node_id': node_id,
                'direction': 'file2 -> file1',
                'clerp': clerp2[:50] + ('...' if len(clerp2) > 50 else '')
            })
            print(f"✅ 节点 {node_id}: 从文件2复制clerp到文件1")
        elif not empty1 and empty2:
            # 文件1不为空，文件2为空，用文件1的值填充文件2
            node2['clerp'] = clerp1
            merge_stats['file1_to_file2'] += 1
            merge_stats['merged_count'] += 1
            merge_stats['merged_details'].append({
                'node_id': node_id,
                'direction': 'file1 -> file2',
                'clerp': clerp1[:50] + ('...' if len(clerp1) > 50 else '')
            })
            print(f"✅ 节点 {node_id}: 从文件1复制clerp到文件2")
    
    return merged_nodes1, merged_nodes2, merge_stats


def save_updated_file(data: Dict[str, Any], nodes: List[Dict], file_path: str):
    """保存更新后的文件"""
    # 更新数据中的节点
    if 'nodes' in data:
        data['nodes'] = nodes
    elif isinstance(data, list):
        data = nodes
    else:
        # 找到原来的节点数组位置并替换
        possible_keys = ['data', 'features', 'items', 'activations']
        updated = False
        for key in possible_keys:
            if key in data and isinstance(data[key], list):
                data[key] = nodes
                updated = True
                break
        
        if not updated:
            # 如果找不到合适的位置，创建nodes字段
            data['nodes'] = nodes
    
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"✅ 已更新文件: {file_path}")
    except Exception as e:
        print(f"❌ 保存文件失败 {file_path}: {e}")
        sys.exit(1)


def print_merge_summary(stats: Dict[str, Any]):
    """打印合并统计信息"""
    print("\n" + "="*50)
    print("📋 合并统计信息")
    print("="*50)
    print(f"总共相同节点数: {stats['total_common']}")
    print(f"成功合并的节点: {stats['merged_count']}")
    print(f"从文件2复制到文件1: {stats['file2_to_file1']}")
    print(f"从文件1复制到文件2: {stats['file1_to_file2']}")
    print(f"两个都为空: {stats['both_empty']}")
    print(f"两个都不为空: {stats['both_non_empty']}")
    
    if stats['merged_details']:
        print(f"\n🔄 具体合并详情 (前10个):")
        for detail in stats['merged_details'][:10]:
            print(f"  • {detail['node_id']}: {detail['clerp']}")
        
        if len(stats['merged_details']) > 10:
            print(f"  ... 还有 {len(stats['merged_details']) - 10} 个")


def main():
    parser = argparse.ArgumentParser(
        description="合并两个circuit JSON文件中的clerp字段，直接更新原始文件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python merge_clerp.py file1.json file2.json
  python merge_clerp.py file1.json file2.json --backup
  
注意: 此脚本会直接修改原始文件，建议先备份！
        """
    )
    
    parser.add_argument('file1', help='第一个JSON文件路径')
    parser.add_argument('file2', help='第二个JSON文件路径')
    parser.add_argument('--backup', action='store_true',
                       help='在修改前创建备份文件')
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not Path(args.file1).exists():
        print(f"❌ 文件1不存在: {args.file1}")
        sys.exit(1)
    
    if not Path(args.file2).exists():
        print(f"❌ 文件2不存在: {args.file2}")
        sys.exit(1)
    
    print("🚀 开始合并clerp字段...")
    print(f"📁 文件1: {args.file1}")
    print(f"📁 文件2: {args.file2}")
    print("⚠️  注意: 将直接修改原始文件!")
    print()
    
    # 创建备份（如果用户要求）
    if args.backup:
        backup1 = f"{args.file1}.backup"
        backup2 = f"{args.file2}.backup"
        print(f"📋 创建备份文件...")
        print(f"   {args.file1} -> {backup1}")
        print(f"   {args.file2} -> {backup2}")
        
        import shutil
        shutil.copy2(args.file1, backup1)
        shutil.copy2(args.file2, backup2)
        print("✅ 备份完成")
        print()
    
    # 加载文件
    data1 = load_json_file(args.file1)
    data2 = load_json_file(args.file2)
    
    # 提取节点
    nodes1 = get_nodes_from_data(data1)
    nodes2 = get_nodes_from_data(data2)
    
    if not nodes1:
        print("❌ 文件1中没有找到节点数据")
        sys.exit(1)
    
    if not nodes2:
        print("❌ 文件2中没有找到节点数据")
        sys.exit(1)
    
    # 合并clerp字段
    merged_nodes1, merged_nodes2, stats = merge_clerp_fields(nodes1, nodes2)
    
    # 保存更新后的文件
    print("\n💾 保存更新后的文件...")
    save_updated_file(data1, merged_nodes1, args.file1)
    save_updated_file(data2, merged_nodes2, args.file2)
    
    # 打印统计信息
    print_merge_summary(stats)
    
    if stats['merged_count'] > 0:
        print(f"\n🎉 成功合并 {stats['merged_count']} 个节点的clerp字段!")
        print(f"📄 两个文件都已更新:")
        print(f"   • {args.file1}")
        print(f"   • {args.file2}")
    else:
        print(f"\nℹ️  没有需要合并的clerp字段")
    
    if args.backup:
        print(f"\n💾 备份文件已保存:")
        print(f"   • {args.file1}.backup")
        print(f"   • {args.file2}.backup")


if __name__ == "__main__":
    main()
