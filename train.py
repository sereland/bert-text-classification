import argparse
import os
import logging
import json
import torch
from transformers import AutoTokenizer
from typing import Optional

# 导入自定义模块
from config import get_config
from utils.data_utils import create_data_loaders, create_test_dataloader, DataProcessor
from utils.bert_utils import ModelTrainer, ModelPredictor, load_model
from models.bert import create_bert_model
from models.bert_cnn import create_bert_cnn_model

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_args():
    """设置命令行参数"""
    parser = argparse.ArgumentParser(description="BERT文本分类/回归训练脚本")
    
    # 基本参数
    parser.add_argument("--task_type", type=str, default="classification", 
                       choices=["classification", "regression"],
                       help="任务类型：分类或回归")
    parser.add_argument("--model_type", type=str, default="bert",
                       choices=["bert", "bert_cnn"],
                       help="模型类型：BERT或BERT-CNN")
    parser.add_argument("--model_name", type=str, default="bert-base-chinese",
                       help="预训练模型名称")
    
    # 数据参数
    parser.add_argument("--train_file", type=str, default="data/train.csv",
                       help="训练数据文件路径")
    parser.add_argument("--test_file", type=str, default="data/test.csv",
                       help="测试数据文件路径")
    parser.add_argument("--text_columns", type=str, default="query,text",
                       help="文本列名，用逗号分隔")
    parser.add_argument("--label_column", type=str, default="label",
                       help="标签列名")
    
    # 训练参数
    parser.add_argument("--batch_size", type=int, default=32,
                       help="批次大小")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                       help="学习率")
    parser.add_argument("--num_epochs", type=int, default=3,
                       help="训练轮数")
    parser.add_argument("--max_length", type=int, default=128,
                       help="文本最大长度")
    parser.add_argument("--num_labels", type=int, default=2,
                       help="分类任务类别数")
    
    # 设备参数
    parser.add_argument("--device", type=str, default="cuda",
                       choices=["cuda", "cpu"],
                       help="设备类型")
    
    # 保存参数
    parser.add_argument("--save_dir", type=str, default="checkpoints",
                        help="模型保存目录")
    parser.add_argument("--save_steps", type=int, default=500,
                        help="保存步数")
    parser.add_argument("--eval_steps", type=int, default=100,
                        help="评估步数")
    parser.add_argument("--logging_steps", type=int, default=50,
                        help="日志记录步数")
    
    # 早停参数
    parser.add_argument("--early_stopping", action="store_true",
                       help="是否使用早停")
    parser.add_argument("--patience", type=int, default=3,
                       help="早停耐心值")
    
    # 模型选择参数
    parser.add_argument("--best_model_criterion", type=str, default="loss",
                       choices=["loss", "f1", "accuracy", "r2", "mse", "mae", "rmse"],
                       help="最优模型选择标准")
    
    # 其他参数
    parser.add_argument("--seed", type=int, default=42,
                       help="随机种子")
    parser.add_argument("--do_train", action="store_true",
                       help="是否进行训练")
    parser.add_argument("--do_eval", action="store_true",
                       help="是否进行评估")
    parser.add_argument("--do_predict", action="store_true",
                       help="是否进行预测")
    parser.add_argument("--model_path", type=str, default=None,
                       help="加载模型路径")
    
    # 预测参数
    parser.add_argument("--predict_input_file", type=str, default=None,
                       help="预测输入文件路径（CSV格式）")
    parser.add_argument("--predict_output_file", type=str, default="predictions.csv",
                       help="预测输出文件路径")
    
    return parser.parse_args()

def set_seed(seed: int):
    """设置随机种子"""
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def create_model(config):
    """创建模型"""
    if config.MODEL_TYPE == "bert":
        return create_bert_model(config)
    elif config.MODEL_TYPE == "bert_cnn":
        return create_bert_cnn_model(config)
    else:
        raise ValueError(f"不支持的模型类型: {config.MODEL_TYPE}")

def train_model(config, model, train_dataloader, val_dataloader=None):
    """训练模型"""
    # 创建训练器
    trainer = ModelTrainer(
        model=model,
        config=config,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader
    )
    
    # 训练模型
    train_history = trainer.train()
    
    # 保存训练历史
    trainer.save_training_history("training_history.json")
    
    return trainer, train_history

def evaluate_model(config, model, test_dataloader):
    """评估模型"""
    # 创建预测器
    predictor = ModelPredictor(model, AutoTokenizer.from_pretrained(config.MODEL_NAME), config)
    
    # 评估模型
    model.eval()
    all_predictions = []
    all_labels = []
    
    for batch in test_dataloader:
        texts = []  # 这里需要从batch中提取文本，但我们的数据集没有保存原始文本
        labels = batch['labels'].cpu().numpy()
        
        # 由于数据集没有保存原始文本，我们需要重新构建
        # 这里简化处理，直接使用模型的输出
        batch = {k: v.to(config.DEVICE) for k, v in batch.items() if k != 'labels'}
        
        with torch.no_grad():
            outputs = model(**batch)
            
            if config.TASK_TYPE == "classification":
                predictions = torch.argmax(outputs['logits'], dim=-1).cpu().numpy()
            else:  # regression
                predictions = outputs['logits'].squeeze().cpu().numpy()
        
        all_predictions.extend(predictions)
        all_labels.extend(labels)
    
    # 计算指标
    from utils.bert_utils import MetricsCalculator
    
    if config.TASK_TYPE == "classification":
        metrics = MetricsCalculator.calculate_classification_metrics(
            all_labels, all_predictions
        )
    else:  # regression
        metrics = MetricsCalculator.calculate_regression_metrics(
            all_labels, all_predictions
        )
    
    logger.info(f"评估指标: {metrics}")
    
    return metrics

def predict_texts(config, model, texts=None, input_file=None, output_file=None):
    """预测文本"""
    import pandas as pd
    
    # 创建预测器
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    predictor = ModelPredictor(model, tokenizer, config)
    
    # 如果指定了输入文件，从文件加载数据
    if input_file is not None:
        logger.info(f"从文件加载数据: {input_file}")
        df = pd.read_csv(input_file)
        
        # 获取文本列
        text_columns = config.TEXT_COLUMNS
        if isinstance(text_columns, str):
            text_columns = [text_columns]
        
        # 合并文本列 - 使用与训练时完全相同的方式
        texts_to_predict = []
        for _, row in df.iterrows():
            # 使用与训练时相同的合并方式：" [SEP] "作为分隔符
            combined_text = " [SEP] ".join([str(row[col]) for col in text_columns if col in row])
            texts_to_predict.append(combined_text)
        
        logger.info(f"加载了 {len(texts_to_predict)} 条数据进行预测")
    else:
        # 使用提供的文本列表
        texts_to_predict = texts
    
    # 预测
    predictions = predictor.predict(texts_to_predict)
    
    # 如果是分类任务，也可以获取概率
    if config.TASK_TYPE == "classification":
        probabilities = predictor.predict_proba(texts_to_predict)
        
        # 保存预测结果
        if output_file is not None:
            logger.info(f"保存预测结果到: {output_file}")
            
            # 创建结果DataFrame
            if input_file is not None:
                # 保留原始数据并添加预测结果
                result_df = df.copy()
                result_df['prediction'] = predictions
                
                # 添加概率列
                for i in range(probabilities[0].shape[0]):
                    result_df[f'probability_class_{i}'] = [prob[i] for prob in probabilities]
            else:
                # 只有文本和预测结果
                result_df = pd.DataFrame({
                    'text': texts_to_predict,
                    'prediction': predictions
                })
                
                # 添加概率列
                for i in range(probabilities[0].shape[0]):
                    result_df[f'probability_class_{i}'] = [prob[i] for prob in probabilities]
            
            result_df.to_csv(output_file, index=False)
            logger.info(f"预测结果已保存到 {output_file}")
        
        return predictions, probabilities
    else:
        # 回归任务
        # 保存预测结果
        if output_file is not None:
            logger.info(f"保存预测结果到: {output_file}")
            
            # 创建结果DataFrame
            if input_file is not None:
                # 保留原始数据并添加预测结果
                result_df = df.copy()
                result_df['prediction'] = predictions
            else:
                # 只有文本和预测结果
                result_df = pd.DataFrame({
                    'text': texts_to_predict,
                    'prediction': predictions
                })
            
            result_df.to_csv(output_file, index=False)
            logger.info(f"预测结果已保存到 {output_file}")
        
        return predictions

def main():
    """主函数"""
    # 解析命令行参数
    args = setup_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 更新配置
    config = get_config(args.task_type, args.model_type)
    config.update_config({
        'MODEL_NAME': args.model_name,
        'TRAIN_FILE': args.train_file,
        'TEST_FILE': args.test_file,
        'TEXT_COLUMNS': args.text_columns.split(','),
        'LABEL_COLUMN': args.label_column,
        'BATCH_SIZE': args.batch_size,
        'LEARNING_RATE': args.learning_rate,
        'NUM_EPOCHS': args.num_epochs,
        'MAX_LENGTH': args.max_length,
        'NUM_LABELS': args.num_labels,
        'DEVICE': args.device,
        'SAVE_DIR': args.save_dir,
        'SAVE_STEPS': args.save_steps,
        'EVAL_STEPS': args.eval_steps,
        'LOGGING_STEPS': args.logging_steps,
        'EARLY_STOPPING': args.early_stopping,
        'PATIENCE': args.patience,
        'BEST_MODEL_CRITERION': args.best_model_criterion
    })
    
    # 创建保存目录
    os.makedirs(config.SAVE_DIR, exist_ok=True)
    
    # 保存配置
    with open(os.path.join(config.SAVE_DIR, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(config.get_config_dict(), f, ensure_ascii=False, indent=2)
    
    logger.info(f"配置: {config.get_config_dict()}")
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    
    # 创建数据加载器
    if args.do_train:
        logger.info("创建训练和验证数据加载器...")
        train_dataloader, val_dataloader = create_data_loaders(config, tokenizer)
        logger.info(f"训练数据批次: {len(train_dataloader)}")
        logger.info(f"验证数据批次: {len(val_dataloader)}")
    
    if args.do_eval or args.do_predict:
        logger.info("创建测试数据加载器...")
        test_dataloader = create_test_dataloader(config, tokenizer)
        logger.info(f"测试数据批次: {len(test_dataloader)}")
    
    # 创建模型
    if args.model_path:
        logger.info(f"加载模型: {args.model_path}")
        model = load_model(args.model_path, create_model, config)
    else:
        logger.info("创建新模型...")
        model = create_model(config)
    
    # 训练模型
    if args.do_train:
        logger.info("开始训练...")
        trainer, train_history = train_model(config, model, train_dataloader, val_dataloader)
        logger.info("训练完成")
        
        # 保存最终模型
        trainer.save_model("final_model.pt")
    
    # 评估模型
    if args.do_eval:
        logger.info("开始评估...")
        metrics = evaluate_model(config, model, test_dataloader)
        logger.info("评估完成")
        
        # 保存评估结果
        with open(os.path.join(config.SAVE_DIR, 'evaluation_results.json'), 'w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
    
    # 预测
    if args.do_predict:
        logger.info("开始预测...")
        
        # 如果指定了输入文件，从文件加载并预测
        if args.predict_input_file is not None:
            predictions = predict_texts(
                config, model,
                input_file=args.predict_input_file,
                output_file=args.predict_output_file
            )
            logger.info(f"预测完成，结果已保存到: {args.predict_output_file}")
        else:
            # 使用示例文本进行预测
            sample_texts = [
                "这是一个测试文本",
                "另一个测试文本"
            ]
            
            predictions = predict_texts(config, model, sample_texts)
            
            logger.info("预测结果:")
            if config.TASK_TYPE == "classification":
                predictions, probabilities = predictions
                for i, (text, pred) in enumerate(zip(sample_texts, predictions)):
                    logger.info(f"文本 {i+1}: {text} -> 预测类别: {pred}")
                    if probabilities:
                        prob_str = ", ".join([f"类别{j}: {prob:.4f}" for j, prob in enumerate(probabilities[i])])
                        logger.info(f"  概率: {prob_str}")
            else:
                for i, (text, pred) in enumerate(zip(sample_texts, predictions)):
                    logger.info(f"文本 {i+1}: {text} -> 预测值: {pred:.4f}")

if __name__ == "__main__":
    main()