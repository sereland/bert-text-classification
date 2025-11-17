"""
高级指标计算工具
包含GAUC和NDCG@K等排序相关指标
"""

import numpy as np
from sklearn.metrics import roc_auc_score, ndcg_score
from typing import List, Union, Dict, Any
import logging

logger = logging.getLogger(__name__)

def compute_gauc(all_labels: np.ndarray, 
                all_preds: np.ndarray, 
                all_queries: List[str]) -> float:
    """
    计算Group AUC (GAUC)
    
    Args:
        all_labels: 所有样本的标签 [n_samples]
        all_preds: 所有样本的预测分数 [n_samples]
        all_queries: 所有样本的query标识 [n_samples]
        
    Returns:
        GAUC值
    """
    if len(all_labels) != len(all_preds) or len(all_labels) != len(all_queries):
        raise ValueError("labels, preds, queries长度必须相同")
    
    # 构建query到样本索引的映射
    query_to_indices = {}
    for i, q in enumerate(all_queries):
        if q not in query_to_indices:
            query_to_indices[q] = []
        query_to_indices[q].append(i)
    
    total_gauc = 0.0
    total_weight = 0
    valid_queries = 0
    
    for q, indices in query_to_indices.items():
        mask = np.array(indices)
        query_labels = all_labels[mask]
        query_preds = all_preds[mask]
        
        # 检查该query是否有正负样本
        unique_labels = np.unique(query_labels)
        if len(unique_labels) < 2:
            # logger.warning(f"Query {q} 只有一个类别，跳过AUC计算")
            continue
        
        try:
            # 计算该query的AUC
            auc_q = roc_auc_score(query_labels, query_preds)
            weight = len(mask)  # 使用样本数作为权重
            
            total_gauc += weight * auc_q
            total_weight += weight
            valid_queries += 1
            
        except Exception as e:
            logger.warning(f"Query {q} AUC计算失败: {e}")
            continue
    
    if total_weight == 0:
        logger.warning("没有有效的query用于计算GAUC")
        return 0.0
    
    gauc = total_gauc / total_weight
    logger.info(f"GAUC计算完成，GAUC: {gauc}，有效query数: {valid_queries}, 总query数: {len(query_to_indices)}")
    
    return gauc

def compute_ndcg_k(labels: np.ndarray,
                  scores: np.ndarray,
                  k: int = 10) -> float:
    """
    计算NDCG@K
    
    Args:
        labels: 真实相关性分数 [n_samples]
        scores: 预测分数 [n_samples]
        k: 截断位置
        
    Returns:
        NDCG@K值，如果样本数<2则返回None
    """
    if len(labels) != len(scores):
        raise ValueError("labels和scores长度必须相同")
    
    if len(labels) == 0:
        return 0.0
    
    # 确保k不超过样本数
    k = min(k, len(labels))
    
    # sklearn的ndcg_score要求至少有2个文档
    if len(labels) < 2:
        # 返回None表示跳过此query的计算
        return None
    
    try:
        # sklearn的ndcg_score需要2D数组
        # logger.info(f'Computing NDCG@{k} for labels: {labels}, scores: {scores}')
        ndcg = ndcg_score([labels], [scores], k=k)
        # logger.info(f'Computed NDCG@{k}: {ndcg}')
        return ndcg
    except Exception as e:
        logger.warning(f"NDCG@{k}计算失败: {e}")
        return None

def compute_group_ndcg_k(all_labels: List[np.ndarray],
                       all_scores: List[np.ndarray],
                       all_queries: List[str],
                       k: int = 10) -> float:
    """
    计算Group NDCG@K（按query分组计算NDCG然后平均）
    
    Args:
        all_labels: 每个query的标签列表
        all_scores: 每个query的预测分数列表
        all_queries: query标识列表
        k: 截断位置
        
    Returns:
        平均NDCG@K值
    """
    if len(all_labels) != len(all_scores) or len(all_labels) != len(all_queries):
        raise ValueError("all_labels, all_scores, all_queries长度必须相同")
    
    total_ndcg = 0.0
    valid_queries = 0
    skipped_queries = 0
    
    for i, q in enumerate(all_queries):
        labels = all_labels[i]
        scores = all_scores[i]
        
        try:
            ndcg = compute_ndcg_k(labels, scores, k)
            if ndcg is not None:
                total_ndcg += ndcg
                valid_queries += 1
            else:
                skipped_queries += 1
        except Exception as e:
            logger.warning(f"Query {q} NDCG@{k}计算失败: {e}")
            skipped_queries += 1
            continue
    
    if valid_queries == 0:
        logger.warning("没有有效的query用于计算NDCG@K")
        return 0.0
    
    avg_ndcg = total_ndcg / valid_queries
    logger.info(f"NDCG@{k}计算完成，有效query数: {valid_queries}, 跳过query数: {skipped_queries}, 总query数: {len(all_queries)}")
    
    return avg_ndcg

def compute_precision_at_1(labels: np.ndarray, scores: np.ndarray) -> float:
    """计算单个 query 的 Precision@1（基于真实和预测的 Top-1 一致性）"""
    if len(labels) < 2:
        return None  # 跳过样本数不足的query

    true_top = np.argmax(labels)     #真实最高分索引
    pred_top = np.argmax(scores)     #预测最高分索引

    logger.info(f'[debug]p@1: {labels}\n, scores: {scores}\n, true_top: {true_top}\n, pred_top: {pred_top}')

    return 1.0 if true_top == pred_top else 0.0

def compute_mrr(labels: np.ndarray, scores: np.ndarray) -> float:
    """计算单个 query 的 MRR（真实 top 文档在预测排序中的 reciprocal rank）"""
    if len(labels) < 2:
        return None

    true_top = np.argmax(labels)

    # 预测排序
    pred_rank = np.argsort(-scores)

    # 找到真实 Top 文档在预测排序中的位置（从 1 开始计数）
    for rank, idx in enumerate(pred_rank, start=1):
        if idx == true_top:
            return 1.0 / rank

    return 0.0  # 理论上不会发生

def compute_group_precision_at_1(group_labels, group_scores):
    total = 0
    valid = 0

    for labels, scores in zip(group_labels, group_scores):
        p1 = compute_precision_at_1(labels, scores)
        if p1 is not None:
            total += p1
            valid += 1

    if valid == 0:
        return 0.0
    
    logger.info(f'Precision@1计算完成，有效query数: {valid}, 总query数: {len(group_labels)}')
    return total / valid


def compute_group_mrr(group_labels, group_scores):
    total = 0
    valid = 0

    for labels, scores in zip(group_labels, group_scores):
        mrr = compute_mrr(labels, scores)
        if mrr is not None:
            total += mrr
            valid += 1

    if valid == 0:
        return 0.0
    
    logger.info(f'MRR计算完成，有效query数: {valid}, 总query数: {len(group_labels)}')
    return total / valid


class RankingMetricsCalculator:
    """排序指标计算器"""
    
    @staticmethod
    def calculate_pairwise_metrics(all_labels: np.ndarray,
                                all_predictions: np.ndarray,
                                all_score_diffs: np.ndarray = None,
                                all_queries: List[str] = None,
                                k_values: List[int] = [5, 10],
                                label_01: bool = False) -> Dict[str, float]:
        """
        计算pairwise任务的指标
        
        Args:
            all_labels: 真实标签
            all_predictions: 预测标签（0或1）
            all_score_diffs: 得分差（用于GAUC）
            all_queries: query标识（用于GAUC和NDCG）
            k_values: NDCG的K值列表
            
        Returns:
            指标字典
        """
        metrics = {}
        if label_01:        
            # 基础分类指标
            from sklearn.metrics import accuracy_score, precision_recall_fscore_support
            accuracy = accuracy_score(all_labels, all_predictions)
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_labels, all_predictions, average='weighted'
            )
            
            metrics.update({
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            })
            
            # 计算GAUC（如果有query信息和得分差）
            if all_queries is not None and all_score_diffs is not None:
                try:
                    gauc = compute_gauc(all_labels, all_score_diffs, all_queries)
                    metrics['gauc'] = gauc
                except Exception as e:
                    logger.warning(f"GAUC计算失败: {e}")
                    metrics['gauc'] = 0.0
        
        # 计算NDCG@K（如果有query信息）
        logger.info(f'k_values for NDCG: {k_values}')
        if all_queries is not None and all_score_diffs is not None:
            for k in k_values:
                logger.info(f"计算NDCG@{k}...")
                try:
                    # 对于pairwise任务，使用得分差作为排序分数
                    # 按query分组计算NDCG
                    query_to_indices = {}
                    for i, q in enumerate(all_queries):
                        if q not in query_to_indices:
                            query_to_indices[q] = []
                        query_to_indices[q].append(i)
                    
                    # 准备compute_group_ndcg_k需要的参数格式
                    group_labels = []
                    group_scores = []
                    group_queries = []
                    
                    for q, indices in query_to_indices.items():
                        mask = np.array(indices)
                        query_labels = all_labels[mask]
                        query_scores = all_score_diffs[mask]
                        
                        if len(query_labels) > 0:
                            group_labels.append(query_labels)
                            group_scores.append(query_scores)
                            group_queries.append(q)
                    
                    # 使用compute_group_ndcg_k计算
                    avg_ndcg = compute_group_ndcg_k(group_labels, group_scores, group_queries, k)
                    metrics[f'ndcg@{k}'] = avg_ndcg
                        
                except Exception as e:
                    logger.warning(f"NDCG@{k}计算失败: {e}")
                    metrics[f'ndcg@{k}'] = 0.0

            # Precision@1 和 MRR（只计算一次，不随 k 变化）
            if 'precision@1' not in metrics:
                metrics['precision@1'] = compute_group_precision_at_1(group_labels, group_scores)
                metrics['mrr'] = compute_group_mrr(group_labels, group_scores)

        
        return metrics
    
    @staticmethod
    def calculate_ctr_metrics(all_labels: np.ndarray,
                          all_scores: np.ndarray,
                          all_queries: List[str] = None,
                          k_values: List[int] = [5, 10]) -> Dict[str, float]:
        """
        计算CTR预测任务的指标
        
        Args:
            all_labels: 真实CTR值
            all_scores: 预测CTR值
            all_queries: query标识
            k_values: NDCG的K值列表
            
        Returns:
            指标字典
        """
        metrics = {}
        
        # 回归指标
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        mse = mean_squared_error(all_labels, all_scores)
        mae = mean_absolute_error(all_labels, all_scores)
        rmse = np.sqrt(mse)
        r2 = r2_score(all_labels, all_scores)
        
        metrics.update({
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r2': r2
        })
        
        # 计算GAUC（如果有query信息）
        if all_queries is not None:
            try:
                gauc = compute_gauc(all_labels, all_scores, all_queries)
                metrics['gauc'] = gauc
            except Exception as e:
                logger.warning(f"GAUC计算失败: {e}")
                metrics['gauc'] = 0.0
        
        # 计算NDCG@K（如果有query信息）
        if all_queries is not None:
            for k in k_values:
                try:
                    # 按query分组计算NDCG
                    query_to_indices = {}
                    for i, q in enumerate(all_queries):
                        if q not in query_to_indices:
                            query_to_indices[q] = []
                        query_to_indices[q].append(i)
                    
                    # 准备compute_group_ndcg_k需要的参数格式
                    group_labels = []
                    group_scores = []
                    group_queries = []
                    
                    for q, indices in query_to_indices.items():
                        mask = np.array(indices)
                        query_labels = all_labels[mask]
                        query_scores = all_scores[mask]
                        
                        if len(query_labels) > 0:
                            group_labels.append(query_labels)
                            group_scores.append(query_scores)
                            group_queries.append(q)
                    
                    # 使用compute_group_ndcg_k计算
                    avg_ndcg = compute_group_ndcg_k(group_labels, group_scores, group_queries, k)
                    metrics[f'ndcg@{k}'] = avg_ndcg
                    
                except Exception as e:
                    logger.warning(f"NDCG@{k}计算失败: {e}")
                    metrics[f'ndcg@{k}'] = 0.0
        
        return metrics

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, mean_squared_error, mean_absolute_error, r2_score, classification_report, roc_auc_score
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
        print(classification_report(y_true, y_pred, digits=4))
        
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

if __name__ == "__main__":
    # 测试指标计算
    print("测试GAUC和NDCG@K计算...")
    
    # 模拟数据
    np.random.seed(42)
    n_samples = 1000
    n_queries = 10
    
    all_labels = np.random.randint(0, 2, n_samples)
    all_preds = np.random.random(n_samples)
    all_queries = [f"query_{i % n_queries}" for i in range(n_samples)]
    
    # 测试GAUC
    gauc = compute_gauc(all_labels, all_preds, all_queries)
    print(f"GAUC: {gauc:.4f}")
    
    # 测试NDCG@K
    labels = np.array([3, 2, 1, 0, 2, 3, 1, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2])
    ndcg5 = compute_ndcg_k(labels, scores, k=5)
    ndcg10 = compute_ndcg_k(labels, scores, k=10)
    print(f"NDCG@5: {ndcg5:.4f}")
    print(f"NDCG@10: {ndcg10:.4f}")
    
    # 测试排序指标计算器
    metrics = RankingMetricsCalculator.calculate_pairwise_metrics(
        all_labels, 
        (all_preds > 0.5).astype(int), 
        all_preds, 
        all_queries,
        k_values=[5, 10]
    )
    print("Pairwise指标:", metrics)