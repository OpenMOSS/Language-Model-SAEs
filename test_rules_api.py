#!/usr/bin/env python3
"""测试规则分析API功能"""

import requests
import json

def test_rules_api():
    """测试规则分析API"""
    
    # 测试用的FEN字符串
    test_fens = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",  # 开局
        "rnbqkb1r/pppp1ppp/5n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",  # 普通局面
        "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",  # 意大利开局
    ]
    
    base_url = "http://localhost:8000"
    
    print("测试规则分析API...")
    print("=" * 50)
    
    for i, fen in enumerate(test_fens, 1):
        print(f"\n测试 {i}: {fen[:30]}...")
        
        try:
            # 测试规则分析API
            response = requests.post(
                f"{base_url}/analyze/rules",
                json={"fen": fen},
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                print("✅ 规则分析成功")
                print(f"状态: {result.get('status')}")
                
                rules = result.get('rules', {})
                if 'error' in rules:
                    print(f"❌ 规则分析错误: {rules['error']}")
                else:
                    print(f"分析结果: {len(rules)} 项规则")
                    
                    # 显示一些关键规则
                    key_rules = [
                        'is_king_in_check', 'is_checkmate', 'is_stalemate',
                        'has_isolated_pawns', 'has_doubled_pawns', 'has_passed_pawns',
                        'is_in_fork', 'has_pinned_pieces'
                    ]
                    
                    for rule in key_rules:
                        if rule in rules:
                            print(f"  {rule}: {rules[rule]}")
                            
                    # 显示中心控制
                    if 'center_control' in rules:
                        center = rules['center_control']
                        if isinstance(center, dict):
                            print(f"  中心控制: 白方 {center.get('white', {})}, 黑方 {center.get('black', {})}")
            else:
                print(f"❌ API请求失败: {response.status_code}")
                print(f"错误信息: {response.text}")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ 网络错误: {e}")
        except Exception as e:
            print(f"❌ 解析错误: {e}")
        
        print("-" * 30)

if __name__ == "__main__":
    test_rules_api() 