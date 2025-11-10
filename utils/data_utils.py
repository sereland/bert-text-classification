import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from typing import List, Dict, Any, Optional, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextDataset(Dataset):
    """文本数据集类"""
    
    def __init__(self, 
                 texts: List[str], 
                 labels: List[float], 
                 tokenizer: AutoTokenizer,
                 max_length: int = 128,
                 is_regression: bool = False):
        """
        初始化数据集
        
        Args:
            texts: 文本列表
            labels: 标签列表
            tokenizer: 分词器
            max_length: 最大长度
            is_regression: 是否为回归任务
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_regression = is_regression
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # 分词和编码
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # 移除batch维度
        encoding = {key: val.squeeze(0) for key, val in encoding.items()}
        
        # 添加标签
        if self.is_regression:
            encoding['labels'] = torch.tensor(label, dtype=torch.float)
        else:
            encoding['labels'] = torch.tensor(label, dtype=torch.long)
        
        return encoding

class DataProcessor:
    """数据处理器"""
    
    def __init__(self, 
                 text_columns: List[str] = ["query", "text"],
                 label_column: str = "label",
                 task_type: str = "classification"):
        """
        初始化数据处理器
        
        Args:
            text_columns: 文本列名列表
            label_column: 标签列名
            task_type: 任务类型 (classification/regression)
        """
        self.text_columns = text_columns
        self.label_column = label_column
        self.task_type = task_type
        self.label_encoder = None
        self.is_regression = task_type == "regression"
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        加载数据文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            DataFrame
        """
        try:
            df = pd.read_csv(file_path, delimiter='\t')
            logger.info(f"成功加载数据文件: {file_path}, 形状: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"加载数据文件失败: {e}")
            raise
    
    def preprocess_data(self, df: pd.DataFrame) -> Tuple[List[str], List[float]]:
        """
        预处理数据
        
        Args:
            df: 数据框
            
        Returns:
            文本列表和标签列表的元组
        """
        # 检查必要的列是否存在
        # missing_cols = [col for col in self.text_columns + [self.label_column] 
        #                 if col not in df.columns]
        # if missing_cols:
        #     raise ValueError(f"数据框中缺少以下列: {missing_cols}")
        
        # 合并文本列
        texts = []
        for _, row in df.iterrows():
            # 将多个文本列合并为一个字符串
            combined_text = "[SEP]".join([str(row[col]) for col in self.text_columns])
            combined_text = "[CLS]" + combined_text 
            texts.append(combined_text)
        
        # 处理标签
        labels = df[self.label_column].tolist()
        
        # 如果是分类任务，对标签进行编码
        if not self.is_regression:
            if self.label_encoder is None:
                self.label_encoder = LabelEncoder()
                labels = self.label_encoder.fit_transform(labels)
            else:
                labels = self.label_encoder.transform(labels)
        
        logger.info(f"预处理完成，文本数量: {len(texts)}, 标签数量: {len(labels)}")
        return texts, labels
    
    def split_data(self, 
                   texts: List[str], 
                   labels: List[float],
                   test_size: float = 0.2,
                   random_state: int = 42) -> Tuple[List[str], List[str], List[float], List[float]]:
        """
        分割数据为训练集和验证集
        
        Args:
            texts: 文本列表
            labels: 标签列表
            test_size: 测试集比例
            random_state: 随机种子
            
        Returns:
            训练文本、验证文本、训练标签、验证标签的元组
        """
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=test_size, random_state=random_state, 
            stratify=labels if not self.is_regression else None
        )
        
        logger.info(f"数据分割完成 - 训练集: {len(train_texts)}, 验证集: {len(val_texts)}")
        return train_texts, val_texts, train_labels, val_labels
    
    def create_dataloader(self,
                         texts: List[str],
                         labels: List[float],
                         tokenizer: AutoTokenizer,
                         batch_size: int = 32,
                         max_length: int = 128,
                         shuffle: bool = True) -> DataLoader:
        """
        创建数据加载器
        
        Args:
            texts: 文本列表
            labels: 标签列表
            tokenizer: 分词器
            batch_size: 批次大小
            max_length: 最大长度
            shuffle: 是否打乱数据
            
        Returns:
            DataLoader
        """
        dataset = TextDataset(
            texts=texts,
            labels=labels,
            tokenizer=tokenizer,
            max_length=max_length,
            is_regression=self.is_regression
        )
        
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=4,
            pin_memory=True
        )
        
        return dataloader
    
    def get_num_classes(self) -> int:
        """
        获取分类任务的类别数
        
        Returns:
            类别数
        """
        if self.is_regression:
            return 1
        elif self.label_encoder is not None:
            return len(self.label_encoder.classes_)
        else:
            raise ValueError("标签编码器未初始化")
    
    def get_class_names(self) -> List[str]:
        """
        获取分类任务的类别名称
        
        Returns:
            类别名称列表
        """
        if self.is_regression:
            return ["regression"]
        elif self.label_encoder is not None:
            return self.label_encoder.classes_.tolist()
        else:
            raise ValueError("标签编码器未初始化")

def create_data_loaders(config,
                        tokenizer: AutoTokenizer) -> Tuple[DataLoader, DataLoader]:
    """
    创建训练和验证数据加载器
    
    Args:
        config: 配置对象
        tokenizer: 分词器
        
    Returns:
        训练数据加载器和验证数据加载器的元组
    """
    # 初始化数据处理器
    processor = DataProcessor(
        text_columns=config.TEXT_COLUMNS,
        label_column=config.LABEL_COLUMN,
        task_type=config.TASK_TYPE
    )
    
    # 加载训练数据
    train_df = processor.load_data(config.TRAIN_FILE)
    
    # 预处理数据
    texts, labels = processor.preprocess_data(train_df)
    
    # 分割数据
    train_texts, val_texts, train_labels, val_labels = processor.split_data(
        texts, labels, test_size=0.2
    )
    
    # 创建数据加载器
    train_dataloader = processor.create_dataloader(
        train_texts, train_labels, tokenizer,
        batch_size=config.BATCH_SIZE,
        max_length=config.MAX_LENGTH,
        shuffle=True
    )
    
    val_dataloader = processor.create_dataloader(
        val_texts, val_labels, tokenizer,
        batch_size=config.BATCH_SIZE,
        max_length=config.MAX_LENGTH,
        shuffle=False
    )
    
    # 更新配置中的类别数
    if config.TASK_TYPE == "classification":
        config.NUM_LABELS = processor.get_num_classes()
    
    return train_dataloader, val_dataloader

def create_test_dataloader(config,
                          tokenizer: AutoTokenizer,
                          processor: Optional[DataProcessor] = None) -> DataLoader:
    """
    创建测试数据加载器
    
    Args:
        config: 配置对象
        tokenizer: 分词器
        processor: 数据处理器（可选）
        
    Returns:
        测试数据加载器
    """
    # 如果没有提供处理器，创建一个新的
    if processor is None:
        processor = DataProcessor(
            text_columns=config.TEXT_COLUMNS,
            label_column=config.LABEL_COLUMN,
            task_type=config.TASK_TYPE
        )
    
    # 加载测试数据
    test_df = processor.load_data(config.TEST_FILE)
    
    # 预处理数据
    texts, labels = processor.preprocess_data(test_df)
    
    # 创建测试数据加载器
    test_dataloader = processor.create_dataloader(
        texts, labels, tokenizer,
        batch_size=config.BATCH_SIZE,
        max_length=config.MAX_LENGTH,
        shuffle=False
    )
    
    return test_dataloader

if __name__ == "__main__":
    # 测试数据处理
    from config import get_config
    
    config = get_config("classification", "bert")
    
    # 创建示例数据
    sample_data = {
        'query': ['查询1', '查询2', '查询3', '查询4'],
        'text': ['文本1', '文本2', '文本3', '文本4'],
        'label': [0, 1, 0, 1]
    }
    
    df = pd.DataFrame(sample_data)
    df.to_csv(config.TRAIN_FILE, index=False)
    
    # 测试数据处理器
    processor = DataProcessor(
        text_columns=config.TEXT_COLUMNS,
        label_column=config.LABEL_COLUMN,
        task_type=config.TASK_TYPE
    )
    
    texts, labels = processor.preprocess_data(df)
    print(f"文本数量: {len(texts)}")
    print(f"标签数量: {len(labels)}")
    print(f"类别数: {processor.get_num_classes()}")
    print(f"类别名称: {processor.get_class_names()}")