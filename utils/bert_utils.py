import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from transformers import get_cosine_schedule_with_warmup, get_constant_schedule_with_warmup
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, mean_squared_error, mean_absolute_error, r2_score, classification_report, roc_auc_score
from sklearn import metrics
import logging
from tqdm import tqdm
import os
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PairwiseLoss(nn.Module):
    """成对损失函数基类"""
    
    def __init__(self, loss_type: str = "ranknet"):
        """
        初始化成对损失函数
        
        Args:
            loss_type: 损失函数类型 (ranknet, margin_ranking, bpr)
        """
        super().__init__()
        self.loss_type = loss_type
        
    def forward(self, scores1: torch.Tensor, scores2: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        计算成对损失
        
        Args:
            scores1: 第一个文本的得分 [batch_size]
            scores2: 第二个文本的得分 [batch_size]
            labels: 标签 [batch_size] (1表示text1优于text2，0表示text2优于text1)
            
        Returns:
            损失值
        """
        if self.loss_type == "ranknet":
            return self._ranknet_loss(scores1, scores2, labels)
        elif self.loss_type == "margin_ranking":
            return self._margin_ranking_loss(scores1, scores2, labels)
        elif self.loss_type == "bpr":
            return self._bpr_loss(scores1, scores2, labels)
        else:
            raise ValueError(f"不支持的损失函数类型: {self.loss_type}")
    
    def _ranknet_loss(self, scores1: torch.Tensor, scores2: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        RankNet损失函数
        
        Args:
            scores1: 第一个文本的得分
            scores2: 第二个文本的得分
            labels: 标签
            
        Returns:
            RankNet损失
        """
        # 计算得分差
        score_diff = scores1 - scores2
        
        # 将标签转换为-1或1
        target = 2 * labels - 1
        
        # 计算交叉熵损失
        loss = torch.log(1 + torch.exp(-target * score_diff))
        
        return loss.mean()
    
    def _margin_ranking_loss(self, scores1: torch.Tensor, scores2: torch.Tensor, labels: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
        """
        Margin Ranking损失函数
        
        Args:
            scores1: 第一个文本的得分
            scores2: 第二个文本的得分
            labels: 标签
            margin: 边界值
            
        Returns:
            Margin Ranking损失
        """
        # 将标签转换为-1或1
        target = 2 * labels - 1
        
        # 计算损失
        loss = torch.relu(-target * (scores1 - scores2) + margin)
        
        return loss.mean()
    
    def _bpr_loss(self, scores1: torch.Tensor, scores2: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Bayesian Personalized Ranking (BPR) 损失函数
        
        Args:
            scores1: 第一个文本的得分
            scores2: 第二个文本的得分
            labels: 标签
            
        Returns:
            BPR损失
        """
        # 只处理正样本对（labels=1）
        mask = labels == 1
        if not mask.any():
            return torch.tensor(0.0, device=scores1.device, requires_grad=True)
        
        # 计算得分差
        score_diff = scores1[mask] - scores2[mask]
        
        # 计算BPR损失
        loss = -torch.log(torch.sigmoid(score_diff))
        
        return loss.mean()

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
    def calculate_auc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
        """
        计算AUC指标
        
        Args:
            y_true: 真实标签
            y_scores: 预测得分
            
        Returns:
            AUC值
        """
        auc = roc_auc_score(y_true, y_scores)
        print(f'auc: {auc}')
        return auc

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
        print(classification_report(y_true, y_pred))
        
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
        # 初始化早停机制
        self.early_stopping = EarlyStopping(patience=config.PATIENCE) if config.EARLY_STOPPING else None
        
        # 训练历史
        self.train_history = {
            'train_loss': [],
            'val_loss': [],
            'val_metrics': [],
            'train_steps': [],      # 记录训练步数
            'val_steps': [],        # 记录验证步数
            'learning_rates': [],    # 记录学习率
            'best_step': None,     # 记录最佳模型的步数
            'best_epoch': None,    # 记录最佳模型的epoch
            'best_metrics': None,  # 记录最佳指标
            'best_score': float('-inf'),  # 记录最佳综合评分
            'best_criterion': None  # 记录最佳模型的选择标准
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
        loss_function = getattr(self.config, 'LOSS_FUNCTION', 'auto')
        
        if loss_function == "auto":
            # 自动选择：根据任务类型
            if self.config.TASK_TYPE == "classification":
                return nn.CrossEntropyLoss()
            elif self.config.TASK_TYPE == "regression":
                return nn.MSELoss()
            elif self.config.TASK_TYPE == "pairwise":
                return PairwiseLoss("ranknet")  # 默认使用RankNet损失
            else:
                raise ValueError(f"不支持的任务类型: {self.config.TASK_TYPE}")
        else:
            # 使用指定的损失函数
            if loss_function == "CrossEntropyLoss":
                return nn.CrossEntropyLoss()
            elif loss_function == "MSELoss":
                return nn.MSELoss()
            elif loss_function == "L1Loss":
                return nn.L1Loss()
            elif loss_function == "SmoothL1Loss":
                return nn.SmoothL1Loss()
            elif loss_function == "BCEWithLogitsLoss":
                return nn.BCEWithLogitsLoss()
            elif loss_function == "KLDivLoss":
                return nn.KLDivLoss(reduction='batchmean')
            elif loss_function == "RankNetLoss":
                return PairwiseLoss("ranknet")
            elif loss_function == "MarginRankingLoss":
                return PairwiseLoss("margin_ranking")
            elif loss_function == "BPRLoss":
                return PairwiseLoss("bpr")
            else:
                raise ValueError(f"不支持的损失函数: {loss_function}")
    
    def evaluate(self) -> Tuple[float, Dict[str, float]]:
        """
        评估模型
        
        Returns:
            平均验证损失和指标字典的元组
        """
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_prediction_pos_scores = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="验证"):
                # 将数据移动到设备
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # 前向传播
                if self.config.TASK_TYPE == "pairwise":
                    # pairwise任务：分别处理两个文本
                    text1_inputs = {k.replace('text1_', ''): v for k, v in batch.items() if k.startswith('text1_')}
                    text2_inputs = {k.replace('text2_', ''): v for k, v in batch.items() if k.startswith('text2_')}
                    
                    scores1 = self.model(**text1_inputs)['logits'].squeeze()
                    scores2 = self.model(**text2_inputs)['logits'].squeeze()
                    
                    # 计算pairwise损失
                    loss = self.criterion(scores1, scores2, batch['labels'])
                    
                    # 收集预测和标签
                    # 对于pairwise任务，预测为text1的得分是否大于text2的得分
                    predictions = (scores1 > scores2).long()  # 使用long类型确保是整数
                    labels = batch['labels']
                    
                    all_predictions.extend(predictions.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                else:
                    # 分类或回归任务
                    outputs = self.model(**batch)
                    loss = outputs['loss'] if 'loss' in outputs else self.criterion(outputs['logits'], batch['labels'])
                    
                    # 收集预测和标签
                    if self.config.TASK_TYPE == "classification":
                        predictions = torch.argmax(outputs['logits'], dim=-1)
                    else:  # regression
                        predictions = outputs['logits'].squeeze()
                    
                    all_predictions.extend(predictions.cpu().numpy())
                    all_prediction_pos_scores.extend(outputs['logits'][:, 1].cpu().numpy())
                    all_labels.extend(batch['labels'].cpu().numpy())
                
                # 累计损失
                total_loss += loss.item()
        
        # 计算平均损失
        avg_loss = total_loss / len(self.val_dataloader)
        
        # 计算指标
        if self.config.TASK_TYPE == "pairwise":
            # pairwise任务使用分类指标（实际上是二分类）
            metrics = MetricsCalculator.calculate_classification_metrics(
                np.array(all_labels), np.array(all_predictions)
            )
        elif self.config.TASK_TYPE == "classification":
            metrics = MetricsCalculator.calculate_classification_metrics(
                np.array(all_labels), np.array(all_predictions)
            )
            auc = MetricsCalculator.calculate_auc(
                np.array(all_labels), np.array(all_prediction_pos_scores)
            )
        else:  # regression
            metrics = MetricsCalculator.calculate_regression_metrics(
                np.array(all_labels), np.array(all_predictions)
            )
        
        return avg_loss, metrics
    def _calculate_model_score(self, val_loss: float, val_metrics: Dict[str, float]) -> float:
        """
        根据配置的标准计算模型综合评分
        
        Args:
            val_loss: 验证损失
            val_metrics: 验证指标字典
            
        Returns:
            综合评分（越高越好）
        """
        criterion = getattr(self.config, 'BEST_MODEL_CRITERION', 'loss')
        
        if criterion == 'loss':
            # 使用验证损失作为评分（损失越小越好，所以用负值）
            return -val_loss
        elif criterion in ['f1', 'accuracy', 'precision', 'recall']:
            # 分类指标：越大越好
            if self.config.TASK_TYPE in ['classification', 'pairwise']:
                return val_metrics.get(criterion, 0.0)
            else:
                logger.warning(f"指标 {criterion} 不适用于回归任务，将使用loss")
                return -val_loss
        elif criterion in ['r2']:
            # R2分数：越大越好
            if self.config.TASK_TYPE == 'regression':
                return val_metrics.get(criterion, 0.0)
            else:
                logger.warning(f"指标 {criterion} 不适用于分类任务，将使用loss")
                return -val_loss
        elif criterion in ['mse', 'mae', 'rmse']:
            # 回归损失指标：越小越好，所以用负值
            if self.config.TASK_TYPE == 'regression':
                return -val_metrics.get(criterion, 0.0)
            else:
                logger.warning(f"指标 {criterion} 不适用于分类或pairwise任务，将使用loss")
                return -val_loss
        elif criterion in ['auc', 'ndcg']:
            # pairwise指标：越大越好
            if self.config.TASK_TYPE == 'pairwise':
                return val_metrics.get(criterion, 0.0)
            else:
                logger.warning(f"指标 {criterion} 不适用于分类或回归任务，将使用loss")
                return -val_loss
        else:
            logger.warning(f"未知的指标标准: {criterion}，将使用loss")
            return -val_loss
    
    def train(self) -> Dict[str, List[float]]:
        """
        训练模型
        
        Returns:
            训练历史
        """
        logger.info("开始训练...")
        
        global_step = 0
        
        for epoch in range(self.config.NUM_EPOCHS):
            logger.info(f"Epoch {epoch + 1}/{self.config.NUM_EPOCHS}")
            
            self.model.train()
            epoch_train_loss = 0.0
            step_count = 0
            
            progress_bar = tqdm(self.train_dataloader, desc="训练")
            
            for batch_idx, batch in enumerate(progress_bar):
                # 将数据移动到设备
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # 前向传播
                if self.config.TASK_TYPE == "pairwise":
                    # pairwise任务：分别处理两个文本
                    text1_inputs = {k.replace('text1_', ''): v for k, v in batch.items() if k.startswith('text1_')}
                    text2_inputs = {k.replace('text2_', ''): v for k, v in batch.items() if k.startswith('text2_')}
                    
                    scores1 = self.model(**text1_inputs)['logits'].squeeze()
                    scores2 = self.model(**text2_inputs)['logits'].squeeze()
                    
                    # 计算pairwise损失
                    loss = self.criterion(scores1, scores2, batch['labels'])
                else:
                    # 分类或回归任务
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
                epoch_train_loss += loss.item()
                step_count += 1
                global_step += 1
                
                # 更新进度条
                progress_bar.set_postfix({'loss': loss.item()})
                
                # 定期日志记录
                if hasattr(self.config, 'LOGGING_STEPS') and global_step % self.config.LOGGING_STEPS == 0:
                    avg_loss = epoch_train_loss / step_count if step_count > 0 else loss.item()
                    current_lr = self.optimizer.param_groups[0]['lr']
                    logger.info(f"Global Step {global_step} - 训练损失: {avg_loss:.4f}, 学习率: {current_lr:.2e}")
                    
                    # 记录训练历史详细信息
                    self.train_history['train_steps'].append(global_step)
                    self.train_history['learning_rates'].append(current_lr)
                
                # 定期保存检查点
                if hasattr(self.config, 'SAVE_STEPS') and global_step % self.config.SAVE_STEPS == 0:
                    self.save_model(f"checkpoint_step_{global_step}.pt")
                    logger.info(f"检查点已保存: checkpoint_step_{global_step}.pt")
                
                # 定期验证
                if self.val_dataloader is not None and hasattr(self.config, 'EVAL_STEPS') and global_step % self.config.EVAL_STEPS == 0:
                    # 保存当前训练损失，避免验证过程影响
                    current_train_loss = epoch_train_loss / step_count if step_count > 0 else loss.item()
                    
                    val_loss, val_metrics = self.evaluate()
                    self.train_history['val_loss'].append(val_loss)
                    self.train_history['val_metrics'].append(val_metrics)
                    self.train_history['val_steps'].append(global_step)
                    
                    logger.info(f"Global Step {global_step} - 训练损失: {current_train_loss:.4f}, 验证损失: {val_loss:.4f}")
                    logger.info(f"验证指标: {val_metrics}")
                    
                    # 计算模型综合评分
                    model_score = self._calculate_model_score(val_loss, val_metrics)
                    
                    # 保存最佳模型
                    if model_score > self.train_history['best_score']:
                        self.train_history['best_score'] = model_score
                        self.train_history['best_step'] = global_step
                        self.train_history['best_epoch'] = epoch + 1
                        self.train_history['best_metrics'] = val_metrics
                        self.train_history['best_criterion'] = getattr(self.config, 'BEST_MODEL_CRITERION', 'loss')
                        self.save_model(f"best_model_step_{global_step}.pt")
                        
                        # 根据选择的标准显示合适的值
                        criterion = getattr(self.config, 'BEST_MODEL_CRITERION', 'loss')
                        if criterion == 'loss':
                            display_score = val_loss
                        elif criterion in ['mse', 'mae', 'rmse']:
                            display_score = val_metrics.get(criterion, 0.0)
                        else:
                            display_score = model_score
                        
                        logger.info(f"最佳模型已保存: best_model_step_{global_step}.pt (Step: {global_step}, Epoch: {epoch + 1}, {criterion}: {display_score:.4f})")
                    
                    # 早停检查
                    if self.early_stopping is not None:
                        if self.early_stopping(model_score):  # 使用综合评分
                            logger.info(f"早停触发，在第 {global_step} 步停止训练 (Best Score: {self.train_history['best_score']:.4f})")
                            return self.train_history
                    
                    # 恢复训练模式
                    self.model.train()
            
            # 计算平均训练损失
            avg_train_loss = epoch_train_loss / step_count if step_count > 0 else 0.0
            self.train_history['train_loss'].append(avg_train_loss)
            
            # 如果没有定期验证或最后一个epoch，进行最终验证
            if self.val_dataloader is not None and (not hasattr(self.config, 'EVAL_STEPS') or global_step % self.config.EVAL_STEPS != 0):
                val_loss, val_metrics = self.evaluate()
                self.train_history['val_loss'].append(val_loss)
                self.train_history['val_metrics'].append(val_metrics)
                self.train_history['val_steps'].append(global_step)
                
                logger.info(f"Epoch {epoch + 1} 完成 - 训练损失: {avg_train_loss:.4f}, 验证损失: {val_loss:.4f}")
                logger.info(f"验证指标: {val_metrics}")
                
                # 计算模型综合评分
                model_score = self._calculate_model_score(val_loss, val_metrics)
                
                # 保存最佳模型
                if model_score > self.train_history['best_score']:
                    self.train_history['best_score'] = model_score
                    self.train_history['best_step'] = global_step
                    self.train_history['best_epoch'] = epoch + 1
                    self.train_history['best_metrics'] = val_metrics
                    self.train_history['best_criterion'] = getattr(self.config, 'BEST_MODEL_CRITERION', 'loss')
                    self.save_model(f"best_model_epoch_{epoch + 1}.pt")
                    
                    # 根据选择的标准显示合适的值
                    criterion = getattr(self.config, 'BEST_MODEL_CRITERION', 'loss')
                    if criterion == 'loss':
                        display_score = val_loss
                    elif criterion in ['mse', 'mae', 'rmse']:
                        display_score = val_metrics.get(criterion, 0.0)
                    else:
                        display_score = model_score
                    
                    logger.info(f"最佳模型已保存: best_model_epoch_{epoch + 1}.pt (Step: {global_step}, Epoch: {epoch + 1}, {criterion}: {display_score:.4f})")
                
                # 早停检查
                if self.early_stopping is not None:
                    if self.early_stopping(model_score):  # 使用综合评分
                        logger.info(f"早停触发，在第 {epoch + 1} 个epoch停止训练 (Best Score: {self.train_history['best_score']:.4f})")
                        break
            else:
                logger.info(f"Epoch {epoch + 1} 完成 - 训练损失: {avg_train_loss:.4f}")
        
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
                elif self.config.TASK_TYPE == "pairwise":
                    # pairwise任务：输出单个得分
                    pred = outputs['logits'].squeeze().item()
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