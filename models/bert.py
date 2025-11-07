import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from typing import Optional, Tuple, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BertClassifier(nn.Module):
    """BERT分类器"""
    
    def __init__(self, config: Any):
        """
        初始化BERT分类器
        
        Args:
            config: 配置对象
        """
        super(BertClassifier, self).__init__()
        self.config = config
        
        # 加载预训练BERT模型
        self.bert = AutoModel.from_pretrained(config.MODEL_NAME)
        
        # 获取BERT配置
        bert_config = AutoConfig.from_pretrained(config.MODEL_NAME)
        self.hidden_size = bert_config.hidden_size
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, config.NUM_LABELS)
        )
        
        # 损失函数
        self.criterion = self._get_criterion()
        
        logger.info(f"BERT分类器初始化完成 - 模型: {config.MODEL_NAME}, 类别数: {config.NUM_LABELS}")
    
    def _get_criterion(self) -> nn.Module:
        """获取损失函数"""
        if self.config.TASK_TYPE == "classification":
            return nn.CrossEntropyLoss()
        elif self.config.TASK_TYPE == "regression":
            return nn.MSELoss()
        else:
            raise ValueError(f"不支持的任务类型: {self.config.TASK_TYPE}")
    
    def forward(self, 
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                token_type_ids: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None) -> dict:
        """
        前向传播
        
        Args:
            input_ids: 输入token IDs
            attention_mask: 注意力掩码
            token_type_ids: token类型IDs
            labels: 标签
            
        Returns:
            包含logits和loss的字典
        """
        # BERT编码
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # 获取[CLS] token的表示
        pooled_output = outputs.pooler_output
        
        # 分类
        logits = self.classifier(pooled_output)
        
        # 计算损失
        loss = None
        if labels is not None:
            loss = self.criterion(logits, labels)
        
        return {
            'loss': loss,
            'logits': logits,
            'hidden_states': outputs.hidden_states if hasattr(outputs, 'hidden_states') else None,
            'attentions': outputs.attentions if hasattr(outputs, 'attentions') else None
        }

class BertRegressor(nn.Module):
    """BERT回归器"""
    
    def __init__(self, config: Any):
        """
        初始化BERT回归器
        
        Args:
            config: 配置对象
        """
        super(BertRegressor, self).__init__()
        self.config = config
        
        # 加载预训练BERT模型
        self.bert = AutoModel.from_pretrained(config.MODEL_NAME)
        
        # 获取BERT配置
        bert_config = AutoConfig.from_pretrained(config.MODEL_NAME)
        self.hidden_size = bert_config.hidden_size
        
        # 回归头
        self.regressor = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, 1)  # 回归任务输出1个值
        )
        
        # 损失函数
        self.criterion = nn.MSELoss()
        
        logger.info(f"BERT回归器初始化完成 - 模型: {config.MODEL_NAME}")
    
    def forward(self, 
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                token_type_ids: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None) -> dict:
        """
        前向传播
        
        Args:
            input_ids: 输入token IDs
            attention_mask: 注意力掩码
            token_type_ids: token类型IDs
            labels: 标签
            
        Returns:
            包含logits和loss的字典
        """
        # BERT编码
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # 获取[CLS] token的表示
        pooled_output = outputs.pooler_output
        
        # 回归
        logits = self.regressor(pooled_output).squeeze(-1)
        
        # 计算损失
        loss = None
        if labels is not None:
            loss = self.criterion(logits, labels)
        
        return {
            'loss': loss,
            'logits': logits,
            'hidden_states': outputs.hidden_states if hasattr(outputs, 'hidden_states') else None,
            'attentions': outputs.attentions if hasattr(outputs, 'attentions') else None
        }

class BertMultiTask(nn.Module):
    """BERT多任务模型"""
    
    def __init__(self, config: Any):
        """
        初始化BERT多任务模型
        
        Args:
            config: 配置对象
        """
        super(BertMultiTask, self).__init__()
        self.config = config
        
        # 加载预训练BERT模型
        self.bert = AutoModel.from_pretrained(config.MODEL_NAME)
        
        # 获取BERT配置
        bert_config = AutoConfig.from_pretrained(config.MODEL_NAME)
        self.hidden_size = bert_config.hidden_size
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, config.NUM_LABELS)
        )
        
        # 回归头
        self.regressor = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, 1)
        )
        
        # 损失函数
        self.classification_criterion = nn.CrossEntropyLoss()
        self.regression_criterion = nn.MSELoss()
        
        logger.info(f"BERT多任务模型初始化完成 - 模型: {config.MODEL_NAME}")
    
    def forward(self, 
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                token_type_ids: Optional[torch.Tensor] = None,
                classification_labels: Optional[torch.Tensor] = None,
                regression_labels: Optional[torch.Tensor] = None) -> dict:
        """
        前向传播
        
        Args:
            input_ids: 输入token IDs
            attention_mask: 注意力掩码
            token_type_ids: token类型IDs
            classification_labels: 分类标签
            regression_labels: 回归标签
            
        Returns:
            包含logits和loss的字典
        """
        # BERT编码
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # 获取[CLS] token的表示
        pooled_output = outputs.pooler_output
        
        # 分类
        classification_logits = self.classifier(pooled_output)
        
        # 回归
        regression_logits = self.regressor(pooled_output).squeeze(-1)
        
        # 计算损失
        classification_loss = None
        regression_loss = None
        total_loss = None
        
        if classification_labels is not None:
            classification_loss = self.classification_criterion(classification_logits, classification_labels)
        
        if regression_labels is not None:
            regression_loss = self.regression_criterion(regression_logits, regression_labels)
        
        # 总损失（如果有多个任务）
        if classification_loss is not None and regression_loss is not None:
            total_loss = classification_loss + regression_loss
        elif classification_loss is not None:
            total_loss = classification_loss
        elif regression_loss is not None:
            total_loss = regression_loss
        
        return {
            'loss': total_loss,
            'classification_loss': classification_loss,
            'regression_loss': regression_loss,
            'classification_logits': classification_logits,
            'regression_logits': regression_logits,
            'hidden_states': outputs.hidden_states if hasattr(outputs, 'hidden_states') else None,
            'attentions': outputs.attentions if hasattr(outputs, 'attentions') else None
        }

def create_bert_model(config: Any) -> nn.Module:
    """
    创建BERT模型
    
    Args:
        config: 配置对象
        
    Returns:
        BERT模型
    """
    if config.TASK_TYPE == "classification":
        return BertClassifier(config)
    elif config.TASK_TYPE == "regression":
        return BertRegressor(config)
    else:
        raise ValueError(f"不支持的任务类型: {config.TASK_TYPE}")

def get_model_size(model: nn.Module) -> int:
    """
    获取模型参数数量
    
    Args:
        model: 模型
        
    Returns:
        参数数量
    """
    return sum(p.numel() for p in model.parameters())

def get_trainable_params(model: nn.Module) -> int:
    """
    获取可训练参数数量
    
    Args:
        model: 模型
        
    Returns:
        可训练参数数量
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def freeze_bert_layers(model: nn.Module, num_layers_to_freeze: int = 0):
    """
    冻结BERT层
    
    Args:
        model: 模型
        num_layers_to_freeze: 要冻结的层数
    """
    if hasattr(model, 'bert'):
        bert_model = model.bert
        
        # 冻结embedding层
        if num_layers_to_freeze > 0:
            for param in bert_model.embeddings.parameters():
                param.requires_grad = False
        
        # 冻结transformer层
        for i in range(min(num_layers_to_freeze, len(bert_model.encoder.layer))):
            for param in bert_model.encoder.layer[i].parameters():
                param.requires_grad = False
        
        logger.info(f"冻结了 {num_layers_to_freeze} 个BERT层")

if __name__ == "__main__":
    # 测试BERT模型
    from config import get_config
    
    # 测试分类模型
    config = get_config("classification", "bert")
    model = create_bert_model(config)
    
    print(f"模型参数数量: {get_model_size(model)}")
    print(f"可训练参数数量: {get_trainable_params(model)}")
    
    # 测试前向传播
    import torch
    batch_size = 2
    seq_length = 128
    input_ids = torch.randint(0, 1000, (batch_size, seq_length))
    attention_mask = torch.ones(batch_size, seq_length)
    labels = torch.randint(0, config.NUM_LABELS, (batch_size,))
    
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    print(f"输出logits形状: {outputs['logits'].shape}")
    print(f"损失: {outputs['loss']}")
    
    # 测试回归模型
    config_reg = get_config("regression", "bert")
    model_reg = create_bert_model(config_reg)
    
    labels_reg = torch.randn(batch_size)
    outputs_reg = model_reg(input_ids=input_ids, attention_mask=attention_mask, labels=labels_reg)
    print(f"回归输出logits形状: {outputs_reg['logits'].shape}")
    print(f"回归损失: {outputs_reg['loss']}")