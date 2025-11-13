#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试配置修复脚本
验证 NUM_LABELS 配置是否正确更新
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import get_config

def test_config_update():
    """测试配置更新功能"""
    print("=== 测试配置更新功能 ===")
    
    # 1. 测试分类任务配置
    print("\n1. 测试分类任务配置:")
    config = get_config("classification", "bert")
    print(f"初始 NUM_LABELS: {config.NUM_LABELS}")
    print(f"get_config_dict() 中的 NUM_LABELS: {config.get_config_dict()['NUM_LABELS']}")
    
    # 模拟 train.py 中的更新过程
    config.__class__.update_config({
        'NUM_LABELS': 1
    })
    
    print(f"更新后 NUM_LABELS: {config.NUM_LABELS}")
    print(f"更新后 get_config_dict() 中的 NUM_LABELS: {config.get_config_dict()['NUM_LABELS']}")
    
    # 2. 测试回归任务配置
    print("\n2. 测试回归任务配置:")
    config_reg = get_config("regression", "bert")
    print(f"初始 NUM_LABELS: {config_reg.NUM_LABELS}")
    print(f"get_config_dict() 中的 NUM_LABELS: {config_reg.get_config_dict()['NUM_LABELS']}")
    
    # 3. 测试Pairwise任务配置
    print("\n3. 测试Pairwise任务配置:")
    config_pairwise = get_config("pairwise", "bert")
    print(f"初始 NUM_LABELS: {config_pairwise.NUM_LABELS}")
    print(f"get_config_dict() 中的 NUM_LABELS: {config_pairwise.get_config_dict()['NUM_LABELS']}")
    
    # 4. 测试BERT-CNN模型配置
    print("\n4. 测试BERT-CNN分类任务配置:")
    config_cnn = get_config("classification", "bert_cnn")
    print(f"初始 NUM_LABELS: {config_cnn.NUM_LABELS}")
    print(f"get_config_dict() 中的 NUM_LABELS: {config_cnn.get_config_dict()['NUM_LABELS']}")
    
    # 模拟更新
    config_cnn.__class__.update_config({
        'NUM_LABELS': 3
    })
    
    print(f"更新后 NUM_LABELS: {config_cnn.NUM_LABELS}")
    print(f"更新后 get_config_dict() 中的 NUM_LABELS: {config_cnn.get_config_dict()['NUM_LABELS']}")
    
    print("\n=== 测试完成 ===")

if __name__ == "__main__":
    test_config_update()