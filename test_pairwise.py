#!/usr/bin/env python3
"""
测试pairwise训练功能
"""

import pandas as pd
import os
import sys
import logging

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import get_config
from utils.data_utils import create_data_loaders, create_test_dataloader
from utils.bert_utils import ModelTrainer, ModelPredictor, load_model
from models.bert import create_bert_model
from transformers import AutoTokenizer

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_sample_pairwise_data():
    """创建示例pairwise数据"""
    # 创建训练数据
    train_data = {
        'query': [
            '搜索手机', '搜索手机', '搜索手机', '搜索手机',
            '搜索电脑', '搜索电脑', '搜索电脑', '搜索电脑'
        ],
        'text1': [
            '苹果手机很好用', '苹果手机很好用', '苹果手机很好用', '苹果手机很好用',
            '联想电脑性能强', '联想电脑性能强', '联想电脑性能强', '联想电脑性能强'
        ],
        'text2': [
            '华为手机也不错', '华为手机也不错', '华为手机也不错', '华为手机也不错',
            '戴尔电脑很稳定', '戴尔电脑很稳定', '戴尔电脑很稳定', '戴尔电脑很稳定'
        ],
        'label': [1, 1, 0, 0, 1, 1, 0, 0]  # 1表示text1优于text2，0表示text2优于text1
    }
    
    train_df = pd.DataFrame(train_data)
    train_df.to_csv('data/train.csv', sep='\t', index=False)
    logger.info("创建训练数据: data/train.csv")
    
    # 创建测试数据
    test_data = {
        'query': [
            '搜索手机', '搜索电脑'
        ],
        'text1': [
            '小米手机性价比高', '华硕电脑游戏强'
        ],
        'text2': [
            'OPPO手机拍照好', '惠普电脑办公佳'
        ],
        'label': [1, 0]
    }
    
    test_df = pd.DataFrame(test_data)
    test_df.to_csv('data/test.csv', sep='\t', index=False)
    logger.info("创建测试数据: data/test.csv")

def test_pairwise_training():
    """测试pairwise训练"""
    logger.info("开始测试pairwise训练...")
    
    # 创建配置
    config = get_config("pairwise", "bert")
    config.update_config({
        'MODEL_NAME': 'bert-base-chinese',
        'NUM_EPOCHS': 2,  # 减少训练轮数用于测试
        'BATCH_SIZE': 2,  # 小批次用于测试
        'LOSS_FUNCTION': 'RankNetLoss',
        'BEST_MODEL_CRITERION': 'accuracy'
    })
    
    logger.info(f"配置: {config.get_config_dict()}")
    
    # 创建示例数据
    create_sample_pairwise_data()
    
    # 创建分词器
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    
    # 创建数据加载器
    logger.info("创建数据加载器...")
    train_dataloader, val_dataloader = create_data_loaders(config, tokenizer)
    test_dataloader = create_test_dataloader(config, tokenizer)
    
    logger.info(f"训练数据批次: {len(train_dataloader)}")
    logger.info(f"验证数据批次: {len(val_dataloader)}")
    logger.info(f"测试数据批次: {len(test_dataloader)}")
    
    # 创建模型
    logger.info("创建模型...")
    model = create_bert_model(config)
    
    # 训练模型
    logger.info("开始训练...")
    trainer = ModelTrainer(
        model=model,
        config=config,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader
    )
    
    train_history = trainer.train()
    logger.info("训练完成")
    
    # 评估模型
    logger.info("开始评估...")
    model.eval()
    val_loss, val_metrics = trainer.evaluate()
    logger.info(f"验证损失: {val_loss:.4f}")
    logger.info(f"验证指标: {val_metrics}")
    
    # 测试预测
    logger.info("开始预测...")
    predictor = ModelPredictor(model, tokenizer, config)
    
    # 测试单个文本预测
    test_texts = [
        "[CLS]搜索手机[SEP]苹果手机很好用[SEP]",
        "[CLS]搜索电脑[SEP]联想电脑性能强[SEP]"
    ]
    
    predictions = predictor.predict(test_texts)
    logger.info("预测结果:")
    for i, (text, pred) in enumerate(zip(test_texts, predictions)):
        logger.info(f"文本 {i+1}: {text} -> 预测得分: {pred:.4f}")
    
    # 测试从文件预测
    logger.info("测试从文件预测...")
    from train import predict_texts
    predictions = predict_texts(
        config=config,
        model=model,
        input_file='data/test.csv',
        output_file='pairwise_predictions.csv'
    )
    
    logger.info("pairwise训练测试完成！")

if __name__ == "__main__":
    test_pairwise_training()