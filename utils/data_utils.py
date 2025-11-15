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
                 max_length: int = 256,
                 is_regression: bool = False,
                 queries: List[str] = None,
                 weights: List[float] = None):
        """
        初始化数据集
        
        Args:
            texts: 文本列表
            labels: 标签列表
            tokenizer: 分词器
            max_length: 最大长度
            is_regression: 是否为回归任务
            queries: query列表（用于GAUC计算）
        """
        self.texts = texts
        self.labels = labels
        self.queries = queries or [f"query_{i}" for i in range(len(texts))]
        self.weights = weights
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_regression = is_regression
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = str(self.texts[idx])
        label = self.labels[idx]
        query = str(self.queries[idx])
        weight = float(self.weights[idx])
        
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
        # if self.is_regression:
        #     encoding['labels'] = torch.tensor(label, dtype=torch.float)
        # elif label != 0 and label != 1:
        #     encoding['labels'] = torch.tensor(label, dtype=torch.float)
        # else:
        #     encoding['labels'] = torch.tensor(label, dtype=torch.long)

        # 暂时放弃01分类了
        encoding['labels'] = torch.tensor(label, dtype=torch.float)
        
        # 添加query信息（用于GAUC计算）
        encoding['query'] = query
        encoding['weight'] = torch.tensor(weight, dtype=torch.float)
        
        return encoding

class PairwiseTextDataset(Dataset):
    """成对文本数据集类"""
    
    def __init__(self,
                 texts1: List[str],
                 texts2: List[str],
                 labels: List[float],
                 tokenizer: AutoTokenizer,
                 queries: List[str] = None,
                 max_length: int = 256):
        """
        初始化成对文本数据集
        
        Args:
            texts1: 第一个文本列表
            texts2: 第二个文本列表
            labels: 标签列表（1表示text1优于text2，0表示text2优于text1）
            queries: query文本列表（用于GAUC和NDCG计算）
            tokenizer: 分词器
            max_length: 最大长度
        """
        self.texts1 = texts1
        self.texts2 = texts2
        self.labels = labels
        self.queries = queries or [f"query_{i}" for i in range(len(texts1))]
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.texts1)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text1 = str(self.texts1[idx])
        text2 = str(self.texts2[idx])
        label = self.labels[idx]
        query = str(self.queries[idx])
        
        # 分别对两个文本进行分词和编码
        encoding1 = self.tokenizer(
            text1,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        encoding2 = self.tokenizer(
            text2,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # 移除batch维度
        encoding1 = {f"text1_{key}": val.squeeze(0) for key, val in encoding1.items()}
        encoding2 = {f"text2_{key}": val.squeeze(0) for key, val in encoding2.items()}
        
        # 合并编码
        encoding = {**encoding1, **encoding2}
        
        # 添加标签（pairwise标签为浮点数）
        encoding['labels'] = torch.tensor(label, dtype=torch.float)
        
        # 添加query信息（用于GAUC和NDCG计算）
        encoding['query'] = query
        
        return encoding

class PairwiseDataProcessor:
    """成对数据处理器"""
    
    def __init__(self,
                 text_columns: List[str] = ["query", "text1", "text2"],
                 label_column: str = "label"):
        """
        初始化成对数据处理器
        
        Args:
            text_columns: 文本列名列表，应该包含query, text1, text2
            label_column: 标签列名
        """
        self.text_columns = text_columns
        self.label_column = label_column
        if len(text_columns) != 3:
            raise ValueError("Pairwise任务需要3个文本列：query, text1, text2")
    
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
    
    def preprocess_data(self, df: pd.DataFrame) -> Tuple[List[str], List[str], List[float], List[str]]:
        """
        预处理成对数据
        
        Args:
            df: 数据框
            
        Returns:
            文本1列表、文本2列表、标签列表和query列表的元组
        """
        # 检查必要的列是否存在
        missing_cols = [col for col in self.text_columns + [self.label_column]
                        if col not in df.columns]
        if missing_cols:
            raise ValueError(f"数据框中缺少以下列: {missing_cols}")
        
        # 处理文本
        query_col, text1_col, text2_col = self.text_columns
        
        texts1 = []
        texts2 = []
        queries = []
        
        for _, row in df.iterrows():
            # 构建text1: [CLS]query[SEP]text1[SEP]
            text1 = f"[CLS]{row[query_col]}[SEP]{row[text1_col]}[SEP]"
            # 构建text2: [CLS]query[SEP]text2[SEP]
            text2 = f"[CLS]{row[query_col]}[SEP]{row[text2_col]}[SEP]"
        
            texts1.append(text1)
            texts2.append(text2)
            queries.append(str(row[query_col]))  # 保存原始query文本
        
        # 处理标签
        labels = df[self.label_column].tolist()
        
        logger.info(f"预处理完成，文本对数量: {len(texts1)}, 标签数量: {len(labels)}")
        return texts1, texts2, labels, queries
    
    def split_data(self,
                   texts1: List[str],
                   texts2: List[str],
                   labels: List[float],
                   queries: List[str] = None,
                   test_size: float = 0.2,
                   random_state: int = 42) -> Tuple[List[str], List[str], List[str], List[str], List[float], List[float], List[str], List[str]]:
        """
        分割成对数据为训练集和验证集
        
        Args:
            texts1: 第一个文本列表
            texts2: 第二个文本列表
            labels: 标签列表
            queries: query列表
            test_size: 测试集比例
            random_state: 随机种子
            
        Returns:
            训练文本1、训练文本2、验证文本1、验证文本2、训练标签、验证标签、训练query、验证query的元组
        """
        # 使用相同的索引分割所有数据
        indices = list(range(len(texts1)))
        train_indices, val_indices = train_test_split(
            indices, test_size=test_size, random_state=random_state, stratify=labels
        )
        
        train_texts1 = [texts1[i] for i in train_indices]
        train_texts2 = [texts2[i] for i in train_indices]
        train_labels = [labels[i] for i in train_indices]
        train_queries = [queries[i] for i in train_indices] if queries else [f"query_{i}" for i in train_indices]
        
        val_texts1 = [texts1[i] for i in val_indices]
        val_texts2 = [texts2[i] for i in val_indices]
        val_labels = [labels[i] for i in val_indices]
        val_queries = [queries[i] for i in val_indices] if queries else [f"query_{i}" for i in val_indices]
        
        logger.info(f"数据分割完成 - 训练集: {len(train_texts1)}, 验证集: {len(val_texts1)}")
        return train_texts1, train_texts2, val_texts1, val_texts2, train_labels, val_labels, train_queries, val_queries
    
    def create_dataloader(self,
                         texts1: List[str],
                         texts2: List[str],
                         labels: List[float],
                         tokenizer: AutoTokenizer,
                         queries: List[str] = None,
                         batch_size: int = 32,
                         max_length: int = 128,
                         shuffle: bool = True) -> DataLoader:
        """
        创建成对数据加载器
        
        Args:
            texts1: 第一个文本列表
            texts2: 第二个文本列表
            labels: 标签列表
            tokenizer: 分词器
            queries: query文本列表（用于GAUC和NDCG计算）
            batch_size: 批次大小
            max_length: 最大长度
            shuffle: 是否打乱数据
            
        Returns:
            DataLoader
        """
        dataset = PairwiseTextDataset(
            texts1=texts1,
            texts2=texts2,
            labels=labels,
            queries=queries,
            tokenizer=tokenizer,
            max_length=max_length
        )
        
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=4,
            pin_memory=True
        )
        
        return dataloader

class DataProcessor:
    """数据处理器"""
    
    def __init__(self, 
                 text_columns: List[str] = ["query", "text"],
                 label_column: str = "label",
                 weight_column: str = None,
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
        self.weight_column = weight_column
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
    
    def preprocess_data(self, df: pd.DataFrame) -> Tuple[List[str], List[float], List[str]]:
        """
        预处理数据
        
        Args:
            df: 数据框
            
        Returns:
            文本列表、标签列表和query列表的元组
        """
        # 检查必要的列是否存在
        missing_cols = [col for col in self.text_columns + [self.label_column]
                        if col not in df.columns]
        if missing_cols:
            raise ValueError(f"数据框中缺少以下列: {missing_cols}")
        
        # 合并文本列
        texts = []
        queries = []
        
        # 如果text_columns包含query列，则提取query信息
        if "query" in self.text_columns:
            for _, row in df.iterrows():
                # 提取query信息
                query = str(row["query"])
                queries.append(query)
                
                # 将其他文本列合并为一个字符串（排除query列）
                text_cols = [col for col in self.text_columns if col != "query"]
                combined_text = "[SEP]".join([str(row[col]) for col in text_cols])
                texts.append(combined_text)
        else:
            # 如果没有query列，生成默认query
            for _, row in df.iterrows():
                # 将所有文本列合并为一个字符串
                logger.warn('没有query列，生成默认query')
                combined_text = "[SEP]".join([str(row[col]) for col in self.text_columns])
                texts.append(combined_text)
                queries.append(f"query_{len(queries)}")
        
        if self.weight_column:
            if self.weight_column in df.columns:
                weights = df[self.weight_column].tolist()
            else:
                logger.warning(f"指定的权重列 {self.weight_column} 不存在，权重全设为1")
                weights = [1.0] * len(texts)
        else:
            logger.warning("没有提供权重列，权重全设为1")
            weights = [1.0] * len(texts)
        # 处理标签
        labels = df[self.label_column].tolist()
        
        # 如果是分类任务，对标签进行编码
        # if not self.is_regression:
        #     if self.label_encoder is None:
        #         self.label_encoder = LabelEncoder()
        #         labels = self.label_encoder.fit_transform(labels)
        #     else:
        #         labels = self.label_encoder.transform(labels)
        
        logger.info(f"预处理完成，文本数量: {len(texts)}, 标签数量: {len(labels)}, query数量: {len(queries)}")
        return texts, labels, queries, weights
    
    def split_data(self,
                   texts: List[str],
                   labels: List[float],
                   queries: List[str] = None,
                   test_size: float = 0.2,
                   random_state: int = 42) -> Tuple[List[str], List[str], List[float], List[float], List[str], List[str]]:
        """
        分割数据为训练集和验证集
        
        Args:
            texts: 文本列表
            labels: 标签列表
            queries: query列表
            test_size: 测试集比例
            random_state: 随机种子
            
        Returns:
            训练文本、验证文本、训练标签、验证标签、训练query、验证query的元组
        """
        if queries is None:
            queries = [f"query_{i}" for i in range(len(texts))]
            
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=test_size, random_state=random_state,
            stratify=labels if not self.is_regression else None
        )
        
        # 分割queries
        train_queries = [queries[i] for i in range(len(texts)) if i in range(len(train_texts))]
        val_queries = [queries[i] for i in range(len(texts)) if i in range(len(train_texts), len(train_texts) + len(val_texts))]
        
        logger.info(f"数据分割完成 - 训练集: {len(train_texts)}, 验证集: {len(val_texts)}")
        return train_texts, val_texts, train_labels, val_labels, train_queries, val_queries
    
    def create_dataloader(self,
                          texts: List[str],
                          labels: List[float],
                          tokenizer: AutoTokenizer,
                          queries: List[str] = None,
                          weights: List[float] = None,
                          batch_size: int = 32,
                          max_length: int = 128,
                          shuffle: bool = True) -> DataLoader:
        """
        创建数据加载器
        
        Args:
            texts: 文本列表
            labels: 标签列表
            tokenizer: 分词器
            queries: query列表（用于GAUC计算）
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
            is_regression=self.is_regression,
            queries=queries,
            weights=weights,
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
    # 根据任务类型选择数据处理器
    if config.TASK_TYPE == "pairwise":
        processor = PairwiseDataProcessor(
            text_columns=config.TEXT_COLUMNS_PAIRWISE,
            label_column=config.LABEL_COLUMN
        )
        
        # 加载训练数据
        train_df = processor.load_data(config.TRAIN_FILE)
        
        # 预处理数据
        texts1, texts2, labels, queries = processor.preprocess_data(train_df)
        
        # 根据配置决定是否使用独立验证集
        if getattr(config, 'USE_VALIDATION_SET', True):
            # 使用独立验证集：从训练集中划分验证集
            train_texts1, train_texts2, val_texts1, val_texts2, train_labels, val_labels, train_queries, val_queries = processor.split_data(
                texts1, texts2, labels, queries, test_size=0.2
            )
    
            # 创建数据加载器
            train_dataloader = processor.create_dataloader(
                train_texts1, train_texts2, train_labels, train_queries, tokenizer,
                batch_size=config.BATCH_SIZE,
                max_length=config.MAX_LENGTH,
                shuffle=True
            )
    
            val_dataloader = processor.create_dataloader(
                val_texts1, val_texts2, val_labels, val_queries, tokenizer,
                batch_size=config.BATCH_SIZE,
                max_length=config.MAX_LENGTH,
                shuffle=False
            )
    
            logger.info("使用独立验证集：从训练集中划分20%作为验证集")
        else:
            # 不使用独立验证集：直接使用测试集作为验证集
            train_dataloader = processor.create_dataloader(
                texts1, texts2, labels, queries, tokenizer,
                batch_size=config.BATCH_SIZE,
                max_length=config.MAX_LENGTH,
                shuffle=True
            )
    
            # 使用测试集作为验证集
            val_dataloader = create_test_dataloader(config, tokenizer, processor)
    
            logger.info("不使用独立验证集：将直接使用测试集作为验证集")
        
        # pairwise任务类别数固定为1（输出单个得分）
        config.NUM_LABELS = 1
        
    else:
        # 分类或回归任务
        processor = DataProcessor(
            text_columns=config.TEXT_COLUMNS,
            label_column=config.LABEL_COLUMN,
            weight_column = config.WEIGHT_COLUMN,
            task_type=config.TASK_TYPE
        )
        
        # 加载训练数据
        train_df = processor.load_data(config.TRAIN_FILE)
        
        # 预处理数据
        texts, labels, queries, weights = processor.preprocess_data(train_df)
        if not config.TRAIN_WITH_WEIGHTS:
            logger.info('不使用权重训练，权重全设置为1')
            weights = [1.0] * len(labels)
        else:
            logger.info(f'使用权重训练, weights样例: {weights[:5]}')
        
        # 根据配置决定是否使用独立验证集
        if getattr(config, 'USE_VALIDATION_SET', True):
            # 使用独立验证集：从训练集中划分验证集
            train_texts, val_texts, train_labels, val_labels, train_queries, val_queries = processor.split_data(
                texts, labels, queries, test_size=0.2
            )
            
            # 创建数据加载器
            train_dataloader = processor.create_dataloader(
                train_texts, train_labels, tokenizer, train_queries,
                batch_size=config.BATCH_SIZE,
                max_length=config.MAX_LENGTH,
                shuffle=True
            )
            
            val_dataloader = processor.create_dataloader(
                val_texts, val_labels, tokenizer, val_queries,
                batch_size=config.VALID_BATCH_SIZE,
                max_length=config.MAX_LENGTH,
                shuffle=False
            )
            
            logger.info("使用独立验证集：从训练集中划分20%作为验证集")
        else:
            # 不使用独立验证集：直接使用测试集作为验证集
            train_dataloader = processor.create_dataloader(
                texts, labels, tokenizer, queries, weights,
                batch_size=config.BATCH_SIZE,
                max_length=config.MAX_LENGTH,
                shuffle=True
            )
            
            # 使用测试集作为验证集
            val_dataloader = create_test_dataloader(config, tokenizer, processor)
            
            logger.info("不使用独立验证集：将直接使用测试集作为验证集")
        
        # 更新配置中的类别数
        # if config.TASK_TYPE == "classification":
        #     config.NUM_LABELS = processor.get_num_classes()
    
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
    # 根据任务类型选择数据处理器
    if config.TASK_TYPE == "pairwise":
        if processor is None or not isinstance(processor, PairwiseDataProcessor):
            processor = PairwiseDataProcessor(
                text_columns=config.TEXT_COLUMNS,
                label_column=config.LABEL_COLUMN
            )
        
        # 加载测试数据
        test_df = processor.load_data(config.TEST_FILE)
        
        # 预处理数据
        texts1, texts2, labels, queries = processor.preprocess_data(test_df)
        
        # 创建测试数据加载器
        test_dataloader = processor.create_dataloader(
            texts1, texts2, labels, queries, tokenizer,
            batch_size=config.TEST_BATCH_SIZE,
            max_length=config.MAX_LENGTH,
            shuffle=False
        )
    else:
        # 分类或回归任务
        if processor is None or not isinstance(processor, DataProcessor):
            processor = DataProcessor(
                text_columns=config.TEXT_COLUMNS,
                label_column=config.LABEL_COLUMN,
                weight_column=config.WEIGHT_COLUMN,
                task_type=config.TASK_TYPE
            )
        
        # 加载测试数据
        test_df = processor.load_data(config.TEST_FILE)
        
        # 预处理数据
        texts, labels, queries, weights = processor.preprocess_data(test_df)
        
        # 创建测试数据加载器
        test_dataloader = processor.create_dataloader(
            texts, labels, tokenizer, queries, weights,
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