#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试配置修复脚本
验证新的配置系统是否正常工作
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import get_config

def test_config_update():
    """测试配置更新功能"""
    print("=== 测试新的配置系统 ===")
    
    # 1. 测试分类任务配置
    print("\n1. 测试分类任务配置:")
    config = get_config("classification", "bert")
    print(f"初始 NUM_LABELS: {config.NUM_LABELS}")
    print(f"get_config_dict() 中的 NUM_LABELS: {config.get_config_dict()['NUM_LABELS']}")
    
    # 模拟 train.py 中的更新过程
    config.update_config({
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
    print(f"模型类型: {config_cnn.MODEL_TYPE}")
    print(f"CNN滤波器: {config_cnn.CNN_FILTERS}")
    
    # 模拟更新
    config_cnn.update_config({
        'NUM_LABELS': 3,
        'BATCH_SIZE': 64
    })
    
    print(f"更新后 NUM_LABELS: {config_cnn.NUM_LABELS}")
    print(f"更新后 BATCH_SIZE: {config_cnn.BATCH_SIZE}")
    print(f"更新后 get_config_dict() 中的 NUM_LABELS: {config_cnn.get_config_dict()['NUM_LABELS']}")
    
    # 5. 测试属性访问
    print("\n5. 测试属性访问:")
    print(f"TASK_TYPE: {config.TASK_TYPE}")
    print(f"MODEL_NAME: {config.MODEL_NAME}")
    print(f"MAX_LENGTH: {config.MAX_LENGTH}")
    
    # 6. 测试直接属性设置
    print("\n6. 测试直接属性设置:")
    config.TEST_ATTR = "test_value"
    print(f"TEST_ATTR: {config.TEST_ATTR}")
    print(f"get_config_dict() 中的 TEST_ATTR: {config.get_config_dict().get('TEST_ATTR')}")
    
    print("\n=== 测试完成 ===")

if __name__ == "__main__":
    test_config_update()