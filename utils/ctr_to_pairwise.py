"""
CTR数据转换为Pairwise数据的工具
用于将query,text,ctr格式的数据转换为query,text1,text2,label格式的pairwise数据
"""

import pandas as pd
import numpy as np
from typing import List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CTRToPairwiseConverter:
    """CTR数据转换为Pairwise数据的转换器"""
    
    def __init__(self, 
                 text_columns: List[str] = ["query", "text"],
                 label_column: str = "label",
                 sampling_strategy: str = "random",
                 min_ctr_diff: float = 0.05,  # 最小CTR差异阈值
                 max_pairs_per_query: int = 100,  # 每query最大pair数
                 balance_ratio: float = 1.0):  # 正负样本比例):
        """
        初始化转换器
        
        Args:
            text_columns: 文本列名列表
            label_column: 标签列名
            sampling_strategy: 采样策略 (random, positive_negative, all_pairs)
        """
        self.text_columns = text_columns
        self.label_column = label_column
        self.sampling_strategy = sampling_strategy
        self.min_ctr_diff = min_ctr_diff
        self.max_pairs_per_query = max_pairs_per_query
        self.balance_ratio = balance_ratio
    
    def convert_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        将CTR数据转换为Pairwise数据
        
        Args:
            df: 原始CTR数据框，包含query, text, ctr列
            
        Returns:
            转换后的Pairwise数据框，包含query, text1, text2, label列
        """
        logger.info(f"开始转换数据，原始数据形状: {df.shape}")
        
        # 检查必要的列是否存在
        required_cols = self.text_columns + [self.label_column]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"数据框中缺少以下列: {missing_cols}")
        
        # 按CTR值排序，便于采样
        df_sorted = df.sort_values(by=self.label_column, ascending=False)
        
        # 根据采样策略生成pairwise数据
        if self.sampling_strategy == "random":
            pairwise_data = self._random_sampling(df_sorted)
        elif self.sampling_strategy == "positive_negative":
            pairwise_data = self._positive_negative_sampling(df_sorted)
        elif self.sampling_strategy == "all_pairs":
            pairwise_data = self._all_pairs_sampling(df_sorted)
        elif self.sampling_strategy == "query_aware":
            pairwise_data = self._query_aware_sampling(df_sorted)
        else:
            raise ValueError(f"不支持的采样策略: {self.sampling_strategy}")
        
        logger.info(f"转换完成，Pairwise数据形状: {pairwise_data.shape}")
        return pairwise_data
    
    def _random_sampling(self, df: pd.DataFrame) -> pd.DataFrame:
        """随机采样策略"""
        pairwise_data = []
        
        # 为每个样本随机选择另一个样本进行比较
        for i in range(len(df)):
            # 随机选择另一个样本
            j = np.random.randint(0, len(df))
            if i != j:  # 确保不是同一个样本
                row_i = df.iloc[i]
                row_j = df.iloc[j]
                
                # 构建pairwise数据
                pairwise_row = {
                    'query': row_i[self.text_columns[0]],
                    'text1': row_i[self.text_columns[1]],
                    'text2': row_j[self.text_columns[1]],
                    'label': 1 if row_i[self.label_column] > row_j[self.label_column] else 0
                }
                pairwise_data.append(pairwise_row)
        
        return pd.DataFrame(pairwise_data)
    
    def _positive_negative_sampling(self, df: pd.DataFrame) -> pd.DataFrame:
        """正负样本采样策略"""
        # 按CTR中位数分割正负样本
        median_ctr = df[self.label_column].median()
        positive_samples = df[df[self.label_column] > median_ctr]
        negative_samples = df[df[self.label_column] <= median_ctr]
        
        logger.info(f"正样本数量: {len(positive_samples)}, 负样本数量: {len(negative_samples)}")
        
        pairwise_data = []
        
        # 为每个正样本匹配一个负样本
        min_samples = min(len(positive_samples), len(negative_samples))
        
        for i in range(min_samples):
            pos_row = positive_samples.iloc[i]
            neg_row = negative_samples.iloc[i]
            
            # 构建正样本对 (正样本 > 负样本)
            pairwise_row_pos = {
                'query': pos_row[self.text_columns[0]],
                'text1': pos_row[self.text_columns[1]],
                'text2': neg_row[self.text_columns[1]],
                'label': 1  # text1优于text2
            }
            pairwise_data.append(pairwise_row_pos)
            
            # 构建负样本对 (负样本 > 正样本) - 交换顺序
            pairwise_row_neg = {
                'query': neg_row[self.text_columns[0]],
                'text1': neg_row[self.text_columns[1]],
                'text2': pos_row[self.text_columns[1]],
                'label': 0  # text1不优于text2
            }
            pairwise_data.append(pairwise_row_neg)
        
        return pd.DataFrame(pairwise_data)
    
    def _all_pairs_sampling(self, df: pd.DataFrame) -> pd.DataFrame:
        """所有可能对采样策略（计算量较大，谨慎使用）"""
        pairwise_data = []
        
        # 生成所有可能的对
        for i in range(len(df)):
            for j in range(i + 1, len(df)):
                row_i = df.iloc[i]
                row_j = df.iloc[j]
                
                # 构建pairwise数据
                pairwise_row = {
                    'query': row_i[self.text_columns[0]],
                    'text1': row_i[self.text_columns[1]],
                    'text2': row_j[self.text_columns[1]],
                    'label': 1 if row_i[self.label_column] > row_j[self.label_column] else 0
                }
                pairwise_data.append(pairwise_row)
        
        return pd.DataFrame(pairwise_data)
    
    def _query_aware_sampling(self, df: pd.DataFrame) -> pd.DataFrame:
        """query内采样"""
        pairwise_data = []
        grouped = df.groupby('query')
        
        for query, group in grouped:
            if len(group) < 2:
                continue
            
            pairs = []
            group_sorted = group.sort_values(self.label_column, ascending=False)
            
            for i in range(len(group_sorted)):
                for j in range(i + 1, len(group_sorted)):
                    row_i = group_sorted.iloc[i]
                    row_j = group_sorted.iloc[j]
                    ctr_diff = row_i[self.label_column] - row_j[self.label_column]
                    
                    # 过滤CTR差异太小的pair
                    if abs(ctr_diff) < self.min_ctr_diff:
                        continue
                    
                    # 正样本
                    pairs.append({
                        'query': query,
                        'text1': row_i['text'],
                        'text2': row_j['text'],
                        'label': 1
                    })
                    
                    # 负样本（交换顺序）
                    pairs.append({
                        'query': query,
                        'text1': row_j['text'],
                        'text2': row_i['text'],
                        'label': 0
                    })
            
            # 限制每个query的样本数
            if len(pairs) > self.max_pairs_per_query:
                pairs = np.random.choice(pairs, self.max_pairs_per_query, replace=False).tolist()
            
            pairwise_data.extend(pairs)
        
        return pd.DataFrame(pairwise_data)
    
    def save_pairwise_data(self, df: pd.DataFrame, output_path: str):
        """
        保存转换后的Pairwise数据
        
        Args:
            df: Pairwise数据框
            output_path: 输出文件路径
        """
        df.to_csv(output_path, sep='\t', index=False)
        logger.info(f"Pairwise数据已保存到: {output_path}")

def create_sample_ctr_data():
    """创建示例CTR数据"""
    # 创建示例CTR数据
    ctr_data = {
        'query': [
            '搜索手机', '搜索手机', '搜索手机', '搜索手机',
            '搜索电脑', '搜索电脑', '搜索电脑', '搜索电脑',
            '搜索手机', '搜索手机', '搜索电脑', '搜索电脑'
        ],
        'text': [
            '苹果手机很好用', '华为手机也不错', '小米手机性价比高', 'OPPO手机拍照好',
            '联想电脑性能强', '戴尔电脑很稳定', '华硕电脑游戏强', '惠普电脑办公佳',
            'vivo手机设计美', '三星手机屏幕好', '宏碁电脑轻薄', '苹果电脑系统好'
        ],
        'label': [
            0.85, 0.78, 0.65, 0.72,  # 手机相关，CTR较高
            0.45, 0.38, 0.52, 0.41,  # 电脑相关，CTR中等
            0.58, 0.63, 0.35, 0.48   # 其他产品，CTR各异
        ]
    }
    
    df = pd.DataFrame(ctr_data)
    df.to_csv('data/ctr_train.csv', sep='\t', index=False)
    logger.info("创建示例CTR数据: data/ctr_train.csv")
    
    return df

def demo_conversion():
    """演示CTR到Pairwise的转换"""
    logger.info("=== CTR到Pairwise数据转换演示 ===")
    
    # 创建示例CTR数据
    # ctr_df = create_sample_ctr_data()
    ctr_df = pd.read_csv('data/test.csv', sep='\t')
    
    # 转换为Pairwise数据
    converter = CTRToPairwiseConverter(
        text_columns=["query", "text"],
        label_column="label",
        sampling_strategy="query_aware"  
    )
    
    pairwise_df = converter.convert_data(ctr_df)
    converter.save_pairwise_data(pairwise_df, 'data/train.csv')
    
    # 显示转换结果统计
    logger.info(f"转换统计:")
    logger.info(f"  原始数据样本数: {len(ctr_df)}")
    logger.info(f"  Pairwise数据样本数: {len(pairwise_df)}")
    logger.info(f"  正样本对数量: {len(pairwise_df[pairwise_df['label'] == 1])}")
    logger.info(f"  负样本对数量: {len(pairwise_df[pairwise_df['label'] == 0])}")
    
    # 显示前几行数据
    logger.info(f"原始数据前5行:")
    logger.info(f"\n{ctr_df.head()}")
    
    logger.info(f"Pairwise数据前5行:")
    logger.info(f"\n{pairwise_df.head()}")

if __name__ == "__main__":
    demo_conversion()