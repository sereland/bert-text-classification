import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from transformers import get_cosine_schedule_with_warmup, get_constant_schedule_with_warmup
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, mean_squared_error, mean_absolute_error, r2_score
import logging
from tqdm import tqdm
import os
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EarlyStopping:
    """早停机制"""
    
    def __init__(self, patience: int = 3, min_delta: float = 0.0):
        """
        初始化早停机制
        
        Args:
            patience: 容忍的epoch数
            min_delta: 最小改进量
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        """
        调用早停机制
        
        Args:
            score: 当前得分
            
        Returns:
            是否应该早停
        """
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
        
        return self.early_stop

class MetricsCalculator:
    """指标计算器"""
    
    @staticmethod
    def calculate_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        计算分类指标
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            
        Returns:
            指标字典
        """
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted'
        )
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    @staticmethod
    def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        计算回归指标
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            
        Returns:
            指标字典
        """
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r2': r2
        }

class ModelTrainer:
    """模型训练器"""
    
    def __init__(self, 
                 model: nn.Module,
                 config: Any,
                 train_dataloader: DataLoader,
                 val_dataloader: Optional[DataLoader] = None):
        """
        初始化训练器
        
        Args:
            model: 模型
            config: 配置对象
            train_dataloader: 训练数据加载器
            val_dataloader: 验证数据加载器
        """
        self.model = model
        self.config = config
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # 初始化优化器
        self.optimizer = self._create_optimizer()
        
        # 初始化学习率调度器
        self.scheduler = self._create_scheduler()
        
        # 初始化损失函数
        self.criterion = self._create_criterion()
        
        # 初始化早停机制
        self.early_stopping = EarlyStopping(patience=config.PATIENCE) if config.EARLY_STOPPING else None
        
        # 训练历史
        self.train_history = {
            'train_loss': [],
            'val_loss': [],
            'val_metrics': []
        }
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """创建优化器"""
        if self.config.OPTIMIZER == "AdamW":
            return AdamW(
                self.model.parameters(),
                lr=self.config.LEARNING_RATE,
                weight_decay=self.config.WEIGHT_DECAY
            )
        elif self.config.OPTIMIZER == "Adam":
            return torch.optim.Adam(
                self.model.parameters(),
                lr=self.config.LEARNING_RATE,
                weight_decay=self.config.WEIGHT_DECAY
            )
        elif self.config.OPTIMIZER == "SGD":
            return torch.optim.SGD(
                self.model.parameters(),
                lr=self.config.LEARNING_RATE,
                weight_decay=self.config.WEIGHT_DECAY,
                momentum=0.9
            )
        else:
            raise ValueError(f"不支持的优化器: {self.config.OPTIMIZER}")
    
    def _create_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """创建学习率调度器"""
        if self.config.SCHEDULER == "linear":
            total_steps = len(self.train_dataloader) * self.config.NUM_EPOCHS
            warmup_steps = int(total_steps * self.config.WARMUP_RATIO)
            return get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps
            )
        elif self.config.SCHEDULER == "cosine":
            total_steps = len(self.train_dataloader) * self.config.NUM_EPOCHS
            warmup_steps = int(total_steps * self.config.WARMUP_RATIO)
            return get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps
            )
        elif self.config.SCHEDULER == "constant":
            warmup_steps = int(len(self.train_dataloader) * self.config.WARMUP_RATIO)
            return get_constant_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps
            )
        else:
            return None
    
    def _create_criterion(self) -> nn.Module:
        """创建损失函数"""
        if self.config.TASK_TYPE == "classification":
            return nn.CrossEntropyLoss()
        elif self.config.TASK_TYPE == "regression":
            return nn.MSELoss()
        else:
            raise ValueError(f"不支持的任务类型: {self.config.TASK_TYPE}")
    
    def train_epoch(self) -> float:
        """
        训练一个epoch
        
        Returns:
            平均训练损失
        """
        self.model.train()
        total_loss = 0.0
        
        progress_bar = tqdm(self.train_dataloader, desc="训练")
        
        for batch in progress_bar:
            # 将数据移动到设备
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # 前向传播
            outputs = self.model(**batch)
            loss = outputs['loss'] if 'loss' in outputs else self.criterion(outputs['logits'], batch['labels'])
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # 优化器步骤
            self.optimizer.step()
            
            # 调度器步骤
            if self.scheduler is not None:
                self.scheduler.step()
            
            # 清零梯度
            self.optimizer.zero_grad()
            
            # 累计损失
            total_loss += loss.item()
            
            # 更新进度条
            progress_bar.set_postfix({'loss': loss.item()})
        
        return total_loss / len(self.train_dataloader)
    
    def evaluate(self) -> Tuple[float, Dict[str, float]]:
        """
        评估模型
        
        Returns:
            平均验证损失和指标字典的元组
        """
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="验证"):
                # 将数据移动到设备
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # 前向传播
                outputs = self.model(**batch)
                loss = outputs['loss'] if 'loss' in outputs else self.criterion(outputs['logits'], batch['labels'])
                
                # 累计损失
                total_loss += loss.item()
                
                # 收集预测和标签
                if self.config.TASK_TYPE == "classification":
                    predictions = torch.argmax(outputs['logits'], dim=-1)
                else:  # regression
                    predictions = outputs['logits'].squeeze()
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(batch['labels'].cpu().numpy())
        
        # 计算平均损失
        avg_loss = total_loss / len(self.val_dataloader)
        
        # 计算指标
        if self.config.TASK_TYPE == "classification":
            metrics = MetricsCalculator.calculate_classification_metrics(
                np.array(all_labels), np.array(all_predictions)
            )
        else:  # regression
            metrics = MetricsCalculator.calculate_regression_metrics(
                np.array(all_labels), np.array(all_predictions)
            )
        
        return avg_loss, metrics
    
    def train(self) -> Dict[str, List[float]]:
        """
        训练模型
        
        Returns:
            训练历史
        """
        logger.info("开始训练...")
        
        best_val_loss = float('inf')
        
        for epoch in range(self.config.NUM_EPOCHS):
            logger.info(f"Epoch {epoch + 1}/{self.config.NUM_EPOCHS}")
            
            # 训练
            train_loss = self.train_epoch()
            self.train_history['train_loss'].append(train_loss)
            
            # 验证
            if self.val_dataloader is not None:
                val_loss, val_metrics = self.evaluate()
                self.train_history['val_loss'].append(val_loss)
                self.train_history['val_metrics'].append(val_metrics)
                
                logger.info(f"训练损失: {train_loss:.4f}, 验证损失: {val_loss:.4f}")
                logger.info(f"验证指标: {val_metrics}")
                
                # 保存最佳模型
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_model(f"best_model_epoch_{epoch + 1}.pt")
                
                # 早停检查
                if self.early_stopping is not None:
                    if self.early_stopping(-val_loss):  # 使用负损失，因为我们要最大化
                        logger.info(f"早停触发，在第 {epoch + 1} 个epoch停止训练")
                        break
            else:
                logger.info(f"训练损失: {train_loss:.4f}")
        
        logger.info("训练完成")
        return self.train_history
    
    def save_model(self, filename: str):
        """
        保存模型
        
        Args:
            filename: 文件名
        """
        os.makedirs(self.config.SAVE_DIR, exist_ok=True)
        filepath = os.path.join(self.config.SAVE_DIR, filename)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config.get_config_dict(),
            'train_history': self.train_history
        }, filepath)
        
        logger.info(f"模型已保存到: {filepath}")
    
    def save_training_history(self, filename: str):
        """
        保存训练历史
        
        Args:
            filename: 文件名
        """
        os.makedirs(self.config.SAVE_DIR, exist_ok=True)
        filepath = os.path.join(self.config.SAVE_DIR, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.train_history, f, ensure_ascii=False, indent=2)
        
        logger.info(f"训练历史已保存到: {filepath}")

class ModelPredictor:
    """模型预测器"""
    
    def __init__(self, model: nn.Module, tokenizer: AutoTokenizer, config: Any):
        """
        初始化预测器
        
        Args:
            model: 模型
            tokenizer: 分词器
            config: 配置对象
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
    
    def predict(self, texts: List[str]) -> List[Any]:
        """
        预测文本
        
        Args:
            texts: 文本列表
            
        Returns:
            预测结果列表
        """
        predictions = []
        
        for text in texts:
            # 分词和编码
            inputs = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.config.MAX_LENGTH,
                return_tensors='pt'
            )
            
            # 将数据移动到设备
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 预测
            with torch.no_grad():
                outputs = self.model(**inputs)
                
                if self.config.TASK_TYPE == "classification":
                    pred = torch.argmax(outputs['logits'], dim=-1).item()
                else:  # regression
                    pred = outputs['logits'].squeeze().item()
                
                predictions.append(pred)
        
        return predictions
    
    def predict_proba(self, texts: List[str]) -> List[np.ndarray]:
        """
        预测概率（仅适用于分类任务）
        
        Args:
            texts: 文本列表
            
        Returns:
            概率数组列表
        """
        if self.config.TASK_TYPE != "classification":
            raise ValueError("概率预测仅适用于分类任务")
        
        probabilities = []
        
        for text in texts:
            # 分词和编码
            inputs = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.config.MAX_LENGTH,
                return_tensors='pt'
            )
            
            # 将数据移动到设备
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 预测
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs['logits'], dim=-1).cpu().numpy()[0]
                probabilities.append(probs)
        
        return probabilities

def load_model(model_path: str, model_class: nn.Module, config: Any) -> nn.Module:
    """
    加载模型
    
    Args:
        model_path: 模型路径
        model_class: 模型类
        config: 配置对象
        
    Returns:
        加载的模型
    """
    checkpoint = torch.load(model_path, map_location='cpu')
    
    model = model_class(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model

if __name__ == "__main__":
    # 测试工具函数
    from config import get_config
    
    config = get_config("classification", "bert")
    
    # 测试指标计算
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1])
    
    metrics = MetricsCalculator.calculate_classification_metrics(y_true, y_pred)
    print("分类指标:", metrics)
    
    y_true_reg = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred_reg = np.array([1.1, 2.1, 2.9, 4.1])
    
    metrics_reg = MetricsCalculator.calculate_regression_metrics(y_true_reg, y_pred_reg)
    print("回归指标:", metrics_reg)