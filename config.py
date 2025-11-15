import os
from typing import Dict, Any, Union

class Config:
    """项目配置类 - 使用字典合并方式，避免复杂的类继承"""
    
    def __init__(self, base_config: Dict[str, Any] = None):
        """初始化配置"""
        # 基础配置
        self._config = {
            # 数据配置
            'DATA_DIR': "data",
            'TRAIN_FILE': os.path.join("data", "train.csv"),
            'TEST_FILE': os.path.join("data", "test.csv"),
            
            # 模型配置
            'MODEL_TYPE': "bert",  # 可选: bert, bert_cnn, bert_xlnet
            'MODEL_NAME': "pretrained/google-bert/bert-base-chinese",  # 预训练模型名称
            'MAX_LENGTH': 128,  # 文本最大长度
            'NUM_LABELS': 1,  # 默认分类任务类别数，回归任务设为1
            
            # 训练配置
            'TASK_TYPE': "classification",  # 可选: classification, regression, pairwise
            'BATCH_SIZE': 32,
            'VALID_BATCH_SIZE': 256,
            'TEST_BATCH_SIZE': 256,
            'LEARNING_RATE': 2e-5,
            'NUM_EPOCHS': 3,
            'WEIGHT_DECAY': 0.01,
            'WARMUP_RATIO': 0.1,
            
            # 损失函数配置
            'LOSS_FUNCTION': "auto",  # 可选: auto, CrossEntropyLoss, MSELoss, L1Loss, SmoothL1Loss, BCEWithLogitsLoss, KLDivLoss, RankNetLoss, MarginRankingLoss, BPRLoss
            
            # 验证集配置
            'USE_VALIDATION_SET': True,  # 是否使用独立的验证集
            
            # 优化器配置
            'OPTIMIZER': "AdamW",  # 可选: AdamW, Adam, SGD
            
            # 学习率调度器配置
            'SCHEDULER': "linear",  # 可选: linear, cosine, constant
            
            # 设备配置
            'DEVICE': "cuda",  # 可选: cuda, cpu
            
            # 保存配置
            'SAVE_DIR': "checkpoints",
            'SAVE_STEPS': 500,
            'EVAL_STEPS': 100,
            'LOGGING_STEPS': 50,
            
            # 早停配置
            'EARLY_STOPPING': True,
            'PATIENCE': 3,
            
            # 模型选择配置
            'BEST_MODEL_CRITERION': "loss",  # 可选: "loss", "f1", "accuracy", "r2", "mse", "mae", "rmse", "auc", "ndcg", "gauc", "ndcg5", "ndcg10"
            
            # 数据处理配置
            'TEXT_COLUMNS': ["query", "text"],  # 文本列名
            'TEXT_COLUMNS_PAIRWISE': ["query", "text1", "text2"],  # pairwise任务文本列名
            'WEIGHT_COLUMN': "weight",  # 样本权重列名
            'LABEL_COLUMN': "label",  # 标签列名
            
            # BERT-CNN 特定配置
            'CNN_FILTERS': [100, 100, 100],  # CNN滤波器数量
            'CNN_KERNEL_SIZES': [3, 4, 5],  # CNN卷积核大小
            'CNN_DROPOUT': 0.1,
            
            # BERT-XLNet 特定配置
            'XLNET_MEM_LEN': 512,  # XLNet记忆长度

            'TRAIN_WITH_WEIGHT': False,  # 是否在训练时使用样本权重
        }
        
        # 如果提供了基础配置，合并它
        if base_config:
            self._config.update(base_config)
    
    def get_config_dict(self) -> Dict[str, Any]:
        """获取配置字典"""
        return self._config.copy()
    
    def update_config(self, config_dict: Dict[str, Any]):
        """更新配置"""
        self._config.update(config_dict)
    
    def __getattr__(self, name: str) -> Any:
        """获取配置属性"""
        if name in self._config:
            return self._config[name]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
    
    def __setattr__(self, name: str, value: Any):
        """设置配置属性"""
        if name.startswith('_'):
            super().__setattr__(name, value)
        else:
            self._config[name] = value

# 任务特定配置
TASK_CONFIGS = {
    "classification": {
        "TASK_TYPE": "classification",
        "LOSS_FUNCTION": "BCEWithLogitsLoss",
        "NUM_LABELS": 1,  # 分类任务输出1个值
    },
    "regression": {
        "TASK_TYPE": "regression",
        "LOSS_FUNCTION": "MSELoss",
        "NUM_LABELS": 1,  # 回归任务输出1个值
    },
    "pairwise": {
        "TASK_TYPE": "pairwise",
        "LOSS_FUNCTION": "RankNetLoss",
        "TEXT_COLUMNS": ["query", "text1", "text2"],  # pairwise任务需要三个文本列
        "NUM_LABELS": 1,  # pairwise任务输出单个得分
        "BEST_MODEL_CRITERION": "auc",  # pairwise任务默认使用AUC作为评估指标
    }
}

# 模型特定配置
MODEL_CONFIGS = {
    "bert": {
        "MODEL_TYPE": "bert",
        "MODEL_NAME": "bert-base-chinese",
    },
    "bert_cnn": {
        "MODEL_TYPE": "bert_cnn",
        "MODEL_NAME": "bert-base-chinese",
        "CNN_FILTERS": [100, 100, 100],
        "CNN_KERNEL_SIZES": [3, 4, 5],
        "CNN_DROPOUT": 0.1,
    },
    "bert_xlnet": {
        "MODEL_TYPE": "bert_xlnet",
        "MODEL_NAME": "xlnet-base-chinese",
        "XLNET_MEM_LEN": 512,
    }
}

def get_config(task_type: str = "classification", model_type: str = "bert") -> Config:
    """根据任务类型和模型类型获取配置"""
    # 验证参数
    if task_type not in TASK_CONFIGS:
        raise ValueError(f"不支持的任务类型: {task_type}")
    if model_type not in MODEL_CONFIGS:
        raise ValueError(f"不支持的模型类型: {model_type}")
    
    # 创建基础配置
    config = Config()
    
    # 合并任务配置
    config.update_config(TASK_CONFIGS[task_type])
    
    # 合并模型配置
    config.update_config(MODEL_CONFIGS[model_type])
    
    return config

if __name__ == "__main__":
    # 测试配置
    config = get_config("classification", "bert")
    print("当前配置:")
    for key, value in config.get_config_dict().items():
        print(f"{key}: {value}")