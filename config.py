import os
from typing import Dict, Any

class Config:
    """项目配置类"""
    
    # 数据配置
    DATA_DIR = "data"
    TRAIN_FILE = os.path.join(DATA_DIR, "train.csv")
    TEST_FILE = os.path.join(DATA_DIR, "test.csv")
    
    # 模型配置
    MODEL_TYPE = "bert"  # 可选: bert, bert_cnn, bert_xlnet
    MODEL_NAME = "bert-base-chinese"  # 预训练模型名称
    MAX_LENGTH = 128  # 文本最大长度
    NUM_LABELS = 2  # 分类任务类别数，回归任务设为1
    
    # 训练配置
    TASK_TYPE = "classification"  # 可选: classification, regression
    BATCH_SIZE = 32
    LEARNING_RATE = 2e-5
    NUM_EPOCHS = 3
    WEIGHT_DECAY = 0.01
    WARMUP_RATIO = 0.1
    
    # 优化器配置
    OPTIMIZER = "AdamW"  # 可选: AdamW, Adam, SGD
    
    # 学习率调度器配置
    SCHEDULER = "linear"  # 可选: linear, cosine, constant
    
    # 设备配置
    DEVICE = "cuda"  # 可选: cuda, cpu
    
    # 保存配置
    SAVE_DIR = "checkpoints"
    SAVE_STEPS = 500
    EVAL_STEPS = 100
    LOGGING_STEPS = 50
    
    # 早停配置
    EARLY_STOPPING = True
    PATIENCE = 3
    
    # 数据处理配置
    TEXT_COLUMNS = ["query", "text"]  # 文本列名
    LABEL_COLUMN = "label"  # 标签列名
    
    # BERT-CNN 特定配置
    CNN_FILTERS = [100, 100, 100]  # CNN滤波器数量
    CNN_KERNEL_SIZES = [3, 4, 5]  # CNN卷积核大小
    CNN_DROPOUT = 0.1
    
    # BERT-XLNet 特定配置
    XLNET_MEM_LEN = 512  # XLNet记忆长度
    
    @classmethod
    def get_config_dict(cls) -> Dict[str, Any]:
        """获取配置字典"""
        return {
            attr: getattr(cls, attr)
            for attr in dir(cls)
            if not attr.startswith('_') and not callable(getattr(cls, attr))
        }
    
    @classmethod
    def update_config(cls, config_dict: Dict[str, Any]):
        """更新配置"""
        for key, value in config_dict.items():
            if hasattr(cls, key):
                setattr(cls, key, value)

# 不同任务的配置
class ClassificationConfig(Config):
    """分类任务配置"""
    TASK_TYPE = "classification"
    LOSS_FUNCTION = "CrossEntropyLoss"
    
class RegressionConfig(Config):
    """回归任务配置"""
    TASK_TYPE = "regression"
    LOSS_FUNCTION = "MSELoss"
    NUM_LABELS = 1

# 模型特定配置
class BertConfig(Config):
    """BERT模型配置"""
    MODEL_TYPE = "bert"
    MODEL_NAME = "bert-base-chinese"

class BertCNNConfig(Config):
    """BERT-CNN模型配置"""
    MODEL_TYPE = "bert_cnn"
    MODEL_NAME = "bert-base-chinese"
    CNN_FILTERS = [100, 100, 100]
    CNN_KERNEL_SIZES = [3, 4, 5]
    CNN_DROPOUT = 0.1

class BertXLNetConfig(Config):
    """BERT-XLNet模型配置"""
    MODEL_TYPE = "bert_xlnet"
    MODEL_NAME = "xlnet-base-chinese"
    XLNET_MEM_LEN = 512

def get_config(task_type: str = "classification", model_type: str = "bert") -> Config:
    """根据任务类型和模型类型获取配置"""
    if task_type == "classification":
        base_config = ClassificationConfig
    elif task_type == "regression":
        base_config = RegressionConfig
    else:
        raise ValueError(f"不支持的任务类型: {task_type}")
    
    if model_type == "bert":
        config_class = type("BertClassificationConfig" if task_type == "classification" else "BertRegressionConfig", 
                           (base_config, BertConfig), {})
    elif model_type == "bert_cnn":
        config_class = type("BertCNNClassificationConfig" if task_type == "classification" else "BertCNNRegressionConfig", 
                           (base_config, BertCNNConfig), {})
    elif model_type == "bert_xlnet":
        config_class = type("BertXLNetClassificationConfig" if task_type == "classification" else "BertXLNetRegressionConfig", 
                           (base_config, BertXLNetConfig), {})
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
    
    return config_class()

if __name__ == "__main__":
    # 测试配置
    config = get_config("classification", "bert")
    print("当前配置:")
    for key, value in config.get_config_dict().items():
        print(f"{key}: {value}")