import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from typing import Optional, Tuple, Any, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CNNLayer(nn.Module):
    """CNN层"""
    
    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 kernel_sizes: List[int],
                 dropout: float = 0.1):
        """
        初始化CNN层
        
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            kernel_sizes: 卷积核大小列表
            dropout: dropout率
        """
        super(CNNLayer, self).__init__()
        
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2)
            for kernel_size in kernel_sizes
        ])
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 (batch_size, seq_len, hidden_size)
            
        Returns:
            输出张量 (batch_size, len(kernel_sizes) * out_channels)
        """
        # 转换维度以适应Conv1d: (batch_size, hidden_size, seq_len)
        x = x.transpose(1, 2)
        
        # 对每个卷积核进行卷积
        conv_outputs = []
        for conv in self.convs:
            conv_out = conv(x)  # (batch_size, out_channels, seq_len)
            conv_out = F.relu(conv_out)
            # 全局最大池化
            conv_out = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
            conv_outputs.append(conv_out)
        
        # 拼接所有卷积输出
        output = torch.cat(conv_outputs, dim=1)
        output = self.dropout(output)
        
        return output

class BertCNNClassifier(nn.Module):
    """BERT-CNN分类器"""
    
    def __init__(self, config: Any):
        """
        初始化BERT-CNN分类器
        
        Args:
            config: 配置对象
        """
        super(BertCNNClassifier, self).__init__()
        self.config = config
        
        # 加载预训练BERT模型
        self.bert = AutoModel.from_pretrained(config.MODEL_NAME)
        
        # 获取BERT配置
        bert_config = AutoConfig.from_pretrained(config.MODEL_NAME)
        self.hidden_size = bert_config.hidden_size
        
        # CNN层
        self.cnn = CNNLayer(
            in_channels=self.hidden_size,
            out_channels=config.CNN_FILTERS[0],  # 使用第一个滤波器数量
            kernel_sizes=config.CNN_KERNEL_SIZES,
            dropout=config.CNN_DROPOUT
        )
        
        # 计算CNN输出维度
        cnn_output_dim = len(config.CNN_KERNEL_SIZES) * config.CNN_FILTERS[0]
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(cnn_output_dim, cnn_output_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(cnn_output_dim // 2, config.NUM_LABELS)
        )
        
        # 损失函数
        self.criterion = self._get_criterion()
        
        logger.info(f"BERT-CNN分类器初始化完成 - 模型: {config.MODEL_NAME}, "
                   f"CNN滤波器: {config.CNN_FILTERS}, 卷积核: {config.CNN_KERNEL_SIZES}")
    
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
        
        # 获取最后一层隐藏状态
        last_hidden_state = outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)
        
        # 应用注意力掩码（将padding位置的输出设为0）
        if attention_mask is not None:
            # 扩展注意力掩码维度
            attention_mask = attention_mask.unsqueeze(-1)  # (batch_size, seq_len, 1)
            last_hidden_state = last_hidden_state * attention_mask
        
        # CNN特征提取
        cnn_features = self.cnn(last_hidden_state)
        
        # 分类
        logits = self.classifier(cnn_features)
        
        # 计算损失
        loss = None
        if labels is not None:
            loss = self.criterion(logits, labels)
        
        return {
            'loss': loss,
            'logits': logits,
            'cnn_features': cnn_features,
            'hidden_states': outputs.hidden_states if hasattr(outputs, 'hidden_states') else None,
            'attentions': outputs.attentions if hasattr(outputs, 'attentions') else None
        }

class BertCNNRegressor(nn.Module):
    """BERT-CNN回归器"""
    
    def __init__(self, config: Any):
        """
        初始化BERT-CNN回归器
        
        Args:
            config: 配置对象
        """
        super(BertCNNRegressor, self).__init__()
        self.config = config
        
        # 加载预训练BERT模型
        self.bert = AutoModel.from_pretrained(config.MODEL_NAME)
        
        # 获取BERT配置
        bert_config = AutoConfig.from_pretrained(config.MODEL_NAME)
        self.hidden_size = bert_config.hidden_size
        
        # CNN层
        self.cnn = CNNLayer(
            in_channels=self.hidden_size,
            out_channels=config.CNN_FILTERS[0],
            kernel_sizes=config.CNN_KERNEL_SIZES,
            dropout=config.CNN_DROPOUT
        )
        
        # 计算CNN输出维度
        cnn_output_dim = len(config.CNN_KERNEL_SIZES) * config.CNN_FILTERS[0]
        
        # 回归头
        self.regressor = nn.Sequential(
            nn.Linear(cnn_output_dim, cnn_output_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(cnn_output_dim // 2, 1)
        )
        
        # 损失函数
        self.criterion = nn.MSELoss()
        
        logger.info(f"BERT-CNN回归器初始化完成 - 模型: {config.MODEL_NAME}")
    
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
        
        # 获取最后一层隐藏状态
        last_hidden_state = outputs.last_hidden_state
        
        # 应用注意力掩码
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(-1)
            last_hidden_state = last_hidden_state * attention_mask
        
        # CNN特征提取
        cnn_features = self.cnn(last_hidden_state)
        
        # 回归
        logits = self.regressor(cnn_features).squeeze(-1)
        
        # 计算损失
        loss = None
        if labels is not None:
            loss = self.criterion(logits, labels)
        
        return {
            'loss': loss,
            'logits': logits,
            'cnn_features': cnn_features,
            'hidden_states': outputs.hidden_states if hasattr(outputs, 'hidden_states') else None,
            'attentions': outputs.attentions if hasattr(outputs, 'attentions') else None
        }

class BertCNNGated(nn.Module):
    """带门控机制的BERT-CNN模型"""
    
    def __init__(self, config: Any):
        """
        初始化带门控机制的BERT-CNN模型
        
        Args:
            config: 配置对象
        """
        super(BertCNNGated, self).__init__()
        self.config = config
        
        # 加载预训练BERT模型
        self.bert = AutoModel.from_pretrained(config.MODEL_NAME)
        
        # 获取BERT配置
        bert_config = AutoConfig.from_pretrained(config.MODEL_NAME)
        self.hidden_size = bert_config.hidden_size
        
        # CNN层
        self.cnn = CNNLayer(
            in_channels=self.hidden_size,
            out_channels=config.CNN_FILTERS[0],
            kernel_sizes=config.CNN_KERNEL_SIZES,
            dropout=config.CNN_DROPOUT
        )
        
        # 门控机制
        cnn_output_dim = len(config.CNN_KERNEL_SIZES) * config.CNN_FILTERS[0]
        self.gate = nn.Sequential(
            nn.Linear(self.hidden_size, cnn_output_dim),
            nn.Sigmoid()
        )
        
        # [CLS] token的表示
        self.cls_linear = nn.Linear(self.hidden_size, cnn_output_dim)
        
        # 分类/回归头
        if config.TASK_TYPE == "classification":
            self.head = nn.Sequential(
                nn.Linear(cnn_output_dim, cnn_output_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(cnn_output_dim // 2, config.NUM_LABELS)
            )
            self.criterion = nn.CrossEntropyLoss()
        else:  # regression
            self.head = nn.Sequential(
                nn.Linear(cnn_output_dim, cnn_output_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(cnn_output_dim // 2, 1)
            )
            self.criterion = nn.MSELoss()
        
        logger.info(f"BERT-CNN门控模型初始化完成 - 模型: {config.MODEL_NAME}")
    
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
        
        # 获取最后一层隐藏状态和[CLS] token
        last_hidden_state = outputs.last_hidden_state
        cls_output = outputs.pooler_output
        
        # 应用注意力掩码
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(-1)
            last_hidden_state = last_hidden_state * attention_mask
        
        # CNN特征提取
        cnn_features = self.cnn(last_hidden_state)
        
        # [CLS]特征
        cls_features = self.cls_linear(cls_output)
        
        # 门控机制
        gate_values = self.gate(cls_output)
        
        # 融合特征
        fused_features = gate_values * cnn_features + (1 - gate_values) * cls_features
        
        # 预测
        logits = self.head(fused_features)
        
        # 处理回归任务的输出
        if self.config.TASK_TYPE == "regression":
            logits = logits.squeeze(-1)
        
        # 计算损失
        loss = None
        if labels is not None:
            loss = self.criterion(logits, labels)
        
        return {
            'loss': loss,
            'logits': logits,
            'cnn_features': cnn_features,
            'cls_features': cls_features,
            'gate_values': gate_values,
            'fused_features': fused_features,
            'hidden_states': outputs.hidden_states if hasattr(outputs, 'hidden_states') else None,
            'attentions': outputs.attentions if hasattr(outputs, 'attentions') else None
        }

def create_bert_cnn_model(config: Any, use_gate: bool = False) -> nn.Module:
    """
    创建BERT-CNN模型
    
    Args:
        config: 配置对象
        use_gate: 是否使用门控机制
        
    Returns:
        BERT-CNN模型
    """
    if use_gate:
        return BertCNNGated(config)
    elif config.TASK_TYPE == "classification":
        return BertCNNClassifier(config)
    elif config.TASK_TYPE == "regression":
        return BertCNNRegressor(config)
    else:
        raise ValueError(f"不支持的任务类型: {config.TASK_TYPE}")

if __name__ == "__main__":
    # 测试BERT-CNN模型
    from config import get_config
    
    # 测试分类模型
    config = get_config("classification", "bert_cnn")
    model = create_bert_cnn_model(config)
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")
    
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
    
    # 测试门控模型
    gate_model = create_bert_cnn_model(config, use_gate=True)
    gate_outputs = gate_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    print(f"门控模型输出logits形状: {gate_outputs['logits'].shape}")
    print(f"门控值形状: {gate_outputs['gate_values'].shape}")
    
    # 测试回归模型
    config_reg = get_config("regression", "bert_cnn")
    model_reg = create_bert_cnn_model(config_reg)
    
    labels_reg = torch.randn(batch_size)
    outputs_reg = model_reg(input_ids=input_ids, attention_mask=attention_mask, labels=labels_reg)
    print(f"回归输出logits形状: {outputs_reg['logits'].shape}")
    print(f"回归损失: {outputs_reg['loss']}")