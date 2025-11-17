import pandas as pd
import numpy as np
from collections import defaultdict
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
        # logger.info(f'queries sample: {self.queries[:5]}, type: {type(self.queries)}')
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
        logger.info(f'[debug] queries sample: {queries[:5]}, type: {type(queries)}, shape: {len(queries)}')  # debug
        
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

class ListwiseDataset(Dataset):
    """
    Listwise 数据集
    每个 item = 某一个 query 下的所有候选文档
    """
    def __init__(self,
                 grouped_data: List[Tuple[str, List[str], List[float], List[float]]],
                 texts: List[str],
                 labels: List[float],
                 tokenizer: AutoTokenizer,
                 queries: List[str],
                 weights: List[float],
                 max_length: int = 256,
                 max_candidates: int = 5,
                 sampling_strategy: str = "topk"):
        
        self.grouped_data = grouped_data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_candidates = max_candidates
        self.sampling_strategy = sampling_strategy
        logger.info(f'[listwise]max_candidates={max_candidates}, sampling_strategy={sampling_strategy}')

        self.processed_data = self._preprocess_data()
    

    def _preprocess_data(self):
        """预处理数据，处理候选数超过限制的情况"""
        processed_data = []
        
        skip_query = 0
        for query, answers, labels, weights in self.grouped_data:
            current_answers = answers.copy()
            current_labels = labels.copy()
            current_weights = weights.copy()

            if len(current_answers) < 2:
                skip_query += 1
                continue
            
            # 如果候选数超过限制，进行采样
            if self.max_candidates and len(current_answers) > self.max_candidates:
                current_answers, current_labels, current_weights = self._sample_candidates(
                    current_answers, current_labels, current_weights
                )
            
            processed_data.append((query, current_answers, current_labels, current_weights))
        logger.info(f'[listwise] 预处理完成，跳过的query数量: {skip_query}, 有效query数量: {len(processed_data)}，总query数量: {len(self.grouped_data)}')
        
        return processed_data
    
    def _sample_candidates(self, answers, labels, weights):
        """采样候选答案"""
        n_candidates = len(answers)
        
        if self.sampling_strategy == "topk":
            indices = np.argsort(labels)[-self.max_candidates:][::-1]
        elif self.sampling_strategy == "random":
            indices = np.random.choice(n_candidates, min(self.max_candidates, n_candidates), replace=False)
        elif self.sampling_strategy == "stratified":
            sorted_indices = np.argsort(labels)
            segment_size = max(1, self.max_candidates // 3)
            
            head_indices = sorted_indices[-min(segment_size, len(sorted_indices)):]
            tail_indices = sorted_indices[:min(segment_size, len(sorted_indices))]
            
            remaining = self.max_candidates - len(head_indices) - len(tail_indices)
            if remaining > 0 and len(sorted_indices) > segment_size * 2:
                mid_start = (len(sorted_indices) - remaining) // 2
                mid_indices = sorted_indices[mid_start:mid_start + remaining]
            else:
                mid_indices = np.array([], dtype=int)
            
            indices = np.concatenate([head_indices, mid_indices, tail_indices])
            indices = indices[:self.max_candidates]
        else:
            indices = np.argsort(labels)[-self.max_candidates:][::-1]
        
        indices = np.unique(indices)
        sampled_answers = [answers[i] for i in indices]
        sampled_labels = [labels[i] for i in indices]
        sampled_weights = [weights[i] for i in indices]
        
        return sampled_answers, sampled_labels, sampled_weights
    
    def _encode_query_answer(self, query: str, answer: str) -> Dict[str, torch.Tensor]:
        """编码query+answer对"""
        # 构造输入文本：query + [SEP] + answer
        text = f"{query}[SEP]{answer}"
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {key: val.squeeze(0) for key, val in encoding.items()}
    
    def __len__(self) -> int:
        return len(self.processed_data)
    
    def __getitem__(self, idx: int) -> Tuple[List[Dict[str, torch.Tensor]], List[float], List[float], int]:
        """
        返回一个组的数据
        
        Returns:
            encodings: 所有query-answer对的编码列表
            labels: CTR标签列表
            weights: 权重列表
            list_size: 实际候选数
        """
        query, answers, labels, weights = self.processed_data[idx]
        
        # 编码所有query-answer对
        encodings = []
        for answer in answers:
            encoding = self._encode_query_answer(query, answer)
            encodings.append(encoding)
        
        return encodings, labels, weights, len(encodings)

def exponential_rank_weight(scores, alpha=0.8):
    ranks = torch.argsort(torch.argsort(-scores))
    weights = torch.exp(-alpha * ranks.float())
    return weights / weights.sum()

import torch.nn.functional as F
def convert_to_rank_distribution(true_scores):
    """
    将真实CTR转换为排序概率分布
    """
    # 第一步：获取排序序号（0表示最高）
    ranks = torch.argsort(torch.argsort(-true_scores))
    # 第二步：将排序转换为概率分布（排名越靠前概率越高）
    true_probs = F.softmax(-ranks.float(), dim=0)
    return true_probs
def listwise_collate_fn(batch: List[Tuple], max_length: int = 256):
    """
    Listwise数据的collate函数 - 单塔BERT版本
    
    Args:
        batch: 一个batch的数据，每个元素是__getitem__的返回值
    """
    if not batch:
        return {}
    batch = [b for b in batch if b[3] >= 2]
    if len(batch) == 0:
        return {}
    
    # 找到当前batch中最长的列表长度
    list_sizes = [list_size for _, _, _, list_size in batch]
    max_list_size = max(list_sizes)
    max_list_size = max(5, max_list_size)  # TODO：这里先写死5了，后面可以改成动态的
    batch_size = len(batch)
    
    # 初始化padding后的tensor
    padded_input_ids = torch.zeros(batch_size, max_list_size, max_length, dtype=torch.long)
    padded_attention_mask = torch.zeros(batch_size, max_list_size, max_length, dtype=torch.long)
    padded_token_type_ids = torch.zeros(batch_size, max_list_size, max_length, dtype=torch.long)
    padded_labels = torch.zeros(batch_size, max_list_size, dtype=torch.float)
    padded_weights = torch.zeros(batch_size, max_list_size, dtype=torch.float)
    mask = torch.zeros(batch_size, max_list_size, dtype=torch.bool)
    
    # 填充数据
    for i, (encodings, labels, weights, list_size) in enumerate(batch):
        # 填充编码
        if list_size < 2:
            continue
        for j in range(list_size):
            encoding = encodings[j]
            padded_input_ids[i, j] = encoding['input_ids']
            padded_attention_mask[i, j] = encoding['attention_mask']
            if 'token_type_ids' in encoding:
                padded_token_type_ids[i, j] = encoding['token_type_ids']
        
        # 填充标签和权重
        padded_labels[i, :list_size] = torch.tensor(labels, dtype=torch.float)
        # padded_labels[i, :list_size] = convert_to_rank_distribution(torch.tensor(labels, dtype=torch.float))
        padded_weights[i, :list_size] = torch.tensor(weights, dtype=torch.float)
        mask[i, :list_size] = 1
    
    # 构造返回字典
    result = {
        'input_ids': padded_input_ids,
        'attention_mask': padded_attention_mask,
        'labels': padded_labels,
        'weights': padded_weights,
        'mask': mask
    }
    
    # 只在有token_type_ids时返回
    if torch.any(padded_token_type_ids != 0):
        result['token_type_ids'] = padded_token_type_ids
    
    return result

class ListwiseDataProcessor:
    """Listwise数据处理器"""
    
    def __init__(self,
                 text_columns: List[str] = ["query", "text"],
                 label_column: str = "label",
                 weight_column: str = None):
        """
        初始化Listwise数据处理器
        
        Args:
            text_columns: 文本列名列表
            label_column: 标签列名
        """
        self.text_columns = text_columns
        self.label_column = label_column
        self.weight_column = weight_column
    
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
            raise ValueError("加载数据文件失败")
    
    def create_dataloader(self,
                          df: pd.DataFrame,
                          tokenizer: AutoTokenizer,
                          batch_size: int = 8,
                          max_length: int = 256,
                          max_candidates: int = 5,
                          sampling_strategy: str = "random",
                          shuffle: bool = True) -> DataLoader:
        """
        创建Listwise数据加载器
        
        Args:
            df: 数据框
            tokenizer: 分词器
            batch_size: 批次大小
            max_length: 最大长度
            max_candidates: 每个query的最大候选数
            sampling_strategy: 采样策略
            shuffle: 是否打乱数据
            
        Returns:
            DataLoader
        """
        # 分组数据
        logger.info('[listwise] 开始按query分组数据...')
        grouped = df.groupby('query')
        grouped_data = []
        
        for query, group in grouped:
            answers = []
            labels = []
            weights = []
            
            for _, row in group.iterrows():
                # 合并文本列
                text_cols = [str(row[col]) for col in self.text_columns if col != "query"]
                combined_text = "[SEP]".join(text_cols)
                answers.append(combined_text)
                labels.append(row[self.label_column])
                if self.weight_column and self.weight_column in row:
                    weights.append(row[self.weight_column])
                else:
                    weights.append(1.0)
            
            grouped_data.append((query, answers, labels, weights))
        logger.info(f'[listwise] 分组完成，query数量: {len(grouped_data)}')
        dataset = ListwiseDataset(
            grouped_data=grouped_data,
            texts=None,
            labels=None,
            tokenizer=tokenizer,
            queries=None,
            weights=None,
            max_length=max_length,
            max_candidates=max_candidates,
            sampling_strategy=sampling_strategy
        )
        
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=lambda x: listwise_collate_fn(x, max_length=max_length),
            num_workers=4,
            pin_memory=True
        )
        
        return dataloader


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
        logger.info('创建pairwise数据加载器')
        processor = PairwiseDataProcessor(
            text_columns=config.TEXT_COLUMNS_PAIRWISE,
            label_column=config.LABEL_COLUMN
        )
        
        # 加载训练数据
        train_df = processor.load_data(config.TRAIN_FILE)
        
        # 预处理数据
        texts1, texts2, labels, queries = processor.preprocess_data(train_df)
        logger.info(f'[debug] queries sample after preprocess: {queries[:5]}, type: {type(queries)}, shape: {len(queries)}')  # debug
        
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
        
    elif config.TASK_TYPE in ['classification', 'regression']:
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
        if not config.TRAIN_WITH_WEIGHT:
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
    else:
        logger.info('创建listwise数据加载器，不使用独立验证集')
        listwise_processor = ListwiseDataProcessor(
            text_columns=config.TEXT_COLUMNS,
            label_column=config.LABEL_COLUMN,
            weight_column=config.WEIGHT_COLUMN)
        
        # 返回一个dataframe: query、text、weight、label
        train_df = listwise_processor.load_data(config.TRAIN_FILE)
        train_dataloader = listwise_processor.create_dataloader(
            train_df,
            tokenizer,
            batch_size=config.BATCH_SIZE,
            max_length=config.MAX_LENGTH,
            shuffle=True
        )
        val_df = listwise_processor.load_data(config.TEST_FILE)
        val_dataloader = listwise_processor.create_dataloader(
            val_df,
            tokenizer,
            batch_size=config.BATCH_SIZE,
            max_length=config.MAX_LENGTH,
            shuffle=False
        )
        logger.info('listwise数据加载器创建完成')
    
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
    elif config.TASK_TYPE in ['classification', 'regression']:
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
    else:
        if processor is None or not isinstance(processor, ListwiseDataProcessor):
            processor = ListwiseDataProcessor(
                text_columns=config.TEXT_COLUMNS,
                label_column=config.LABEL_COLUMN,
                weight_column=config.WEIGHT_COLUMN)
            test_df = processor.load_data(config.TEST_FILE)
            test_dataloader = processor.create_dataloader(
                test_df,
                tokenizer,
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