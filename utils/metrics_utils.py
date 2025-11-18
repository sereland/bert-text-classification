"""
高级指标计算工具
包含GAUC和NDCG@K等排序相关指标
"""

import numpy as np
from sklearn.metrics import roc_auc_score, ndcg_score
from typing import List, Union, Dict, Any, Optional
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

def smart_sample_indices(
    labels: np.ndarray,
    sample_size: int,
    seed: Optional[int] = 42
) -> np.ndarray:
    """
    智能抽样：保证最大CTR样本一定被选中，其余随机补齐，并打乱顺序（关键！）
    """
    n = len(labels)
    if sample_size >= n:
        return np.arange(n)

    rng = np.random.default_rng(seed)
    max_idx = np.argmax(labels)
    remaining = np.setdiff1d(np.arange(n), [max_idx])

    need = sample_size - 1
    if need <= 0:
        selected = np.array([max_idx])
    else:
        chosen = rng.choice(remaining, size=need, replace=False, shuffle=False)
        selected = np.concatenate([[max_idx], chosen])

    # 必须打乱！否则Top1会被严重高估
    rng.shuffle(selected)
    return selected


# ====================== 单query指标（支持抽样） ======================
def compute_precision_at_1(
    labels: np.ndarray,
    scores: np.ndarray,
    sample_size: Optional[int] = None,
    seed: Optional[int] = 42
) -> Optional[float]:
    """Top-1 Accuracy（真实Top1是否被模型排到第一）"""
    if len(labels) < 2:
        return None

    if sample_size is not None and len(labels) > sample_size:
        idx = smart_sample_indices(labels, sample_size, seed)
        labels = labels[idx]
        scores = scores[idx]

    if len(labels) < 2:   # 抽样后可能变成1个
        return None

    true_top = np.argmax(labels)
    pred_top = np.argmax(scores)
    return 1.0 if true_top == pred_top else 0.0


def compute_top2_set_accuracy(
    labels: np.ndarray,
    scores: np.ndarray,
    sample_size: Optional[int] = None,
    seed: Optional[int] = 42
) -> Optional[float]:
    """Top2 Set Accuracy：预测的Top2集合是否和真实完全一致"""
    if len(labels) < 2:
        return None

    if sample_size is not None and len(labels) > sample_size:
        idx = smart_sample_indices(labels, sample_size, seed)
        labels = labels[idx]
        scores = scores[idx]

    if len(labels) < 2:
        return None

    true_top2 = set(np.argsort(-labels)[:2])
    pred_top2 = set(np.argsort(-scores)[:2])
    return 1.0 if true_top2 == pred_top2 else 0.0


def compute_mrr(
    labels: np.ndarray,
    scores: np.ndarray,
    sample_size: Optional[int] = None,
    seed: Optional[int] = 42
) -> Optional[float]:
    if len(labels) < 2:
        return None

    if sample_size is not None and len(labels) > sample_size:
        idx = smart_sample_indices(labels, sample_size, seed)
        labels = labels[idx]
        scores = scores[idx]

    true_top = np.argmax(labels)
    pred_ranks = np.argsort(-scores)
    for rank, idx in enumerate(pred_ranks, 1):
        if idx == true_top:
            return 1.0 / rank
    return 0.0


def compute_ndcg_k_single(
    labels: np.ndarray,
    scores: np.ndarray,
    k: int
) -> float:
    """单个query的NDCG@K（不抽样，直接用sklearn）"""
    k = min(k, len(labels))
    if k == 0:
        return 0.0
    try:
        return ndcg_score([labels], [scores], k=k)
    except:
        return 0.0


# ====================== Group 级别指标 ======================
def compute_group_metrics(
    group_labels: List[np.ndarray],
    group_scores: List[np.ndarray],
    sample_size: Optional[int] = None,
    sample_ndcg: bool = False,          # 新增开关！
    ndcg_ks: List[int] = None,
    seed: Optional[int] = 2020
) -> Dict[str, float]:
    """
    统一计算所有分组指标
    """
    if ndcg_ks is None:
        ndcg_ks = [5, 10]

    results = {}
    total_queries = len(group_labels)

    # --- Precision@1 & Top2 Set Accuracy & MRR ---
    p1_total, top2_total, mrr_total = 0.0, 0.0, 0.0
    valid_cnt = 0

    for labels, scores in zip(group_labels, group_scores):
        p1 = compute_precision_at_1(labels, scores, sample_size, seed)
        top2 = compute_top2_set_accuracy(labels, scores, sample_size, seed)
        mrr = compute_mrr(labels, scores, sample_size, seed)

        if p1 is not None:
            p1_total += p1
            top2_total += top2
            mrr_total += mrr
            valid_cnt += 1

    if valid_cnt > 0:
        results['precision@1'] = p1_total / valid_cnt
        results['top2_set_accuracy'] = top2_total / valid_cnt
        results['mrr'] = mrr_total / valid_cnt
    else:
        results['precision@1'] = results['top2_set_accuracy'] = results['mrr'] = 0.0

    # --- NDCG@K（开关决定是否抽样）---
    ndcg_totals = {k: 0.0 for k in ndcg_ks}
    ndcg_valid = {k: 0 for k in ndcg_ks}

    for labels, scores in zip(group_labels, group_scores):
        cur_labels = labels
        cur_scores = scores
        if sample_ndcg and sample_size is not None and len(labels) > sample_size:
            idx = smart_sample_indices(labels, sample_size, seed)
            cur_labels = labels[idx]
            cur_scores = scores[idx]

        for k in ndcg_ks:
            ndcg = compute_ndcg_k_single(cur_labels, cur_scores, k)
            ndcg_totals[k] += ndcg
            ndcg_valid[k] += 1

    for k in ndcg_ks:
        if ndcg_valid[k] > 0:
            results[f'ndcg@{k}'] = ndcg_totals[k] / ndcg_valid[k]
        else:
            results[f'ndcg@{k}'] = 0.0

    logger.info(f"指标计算完成 | 总query数: {total_queries} | 有效query数(Top1等): {valid_cnt} | "
                f"sample_size={sample_size} | sample_ndcg={sample_ndcg}")
    return results


class RankingMetricsCalculator:
    @staticmethod
    def calculate_pairwise_metrics(
        all_labels: np.ndarray,
        all_predictions: np.ndarray,          # pairwise 0/1 预测结果
        all_score_diffs: np.ndarray = None,   # 模型给的排序分数（越高越好）
        all_queries: Optional[List[str]] = None,
        k_values: List[int] = None,
        label_01: bool = False,
        sample_size: Optional[int] = 5,       # None 表示不对Top1/Top2/MRR抽样
        sample_ndcg: bool = False,            # <<<< 新增：是否对NDCG也抽样
        seed: Optional[int] = 2020
    ) -> Dict[str, float]:
        if k_values is None:
            k_values = [5, 10]

        metrics = {}

        # 分类指标（可选）
        if label_01:
            accuracy = accuracy_score(all_labels, all_predictions)
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_labels, all_predictions, average='weighted', zero_division=0
            )
            metrics.update({'accuracy': accuracy, 'precision': precision,
                            'recall': recall, 'f1': f1})

            if all_queries is not None and all_score_diffs is not None:
                try:
                    gauc = compute_gauc(all_labels, all_score_diffs, all_queries)
                    metrics['gauc'] = gauc
                except Exception as e:
                    logger.warning(f"GAUC计算失败: {e}")
                    metrics['gauc'] = 0.0

        # 需要按query分组的情况
        if all_queries is not None and all_score_diffs is not None:
            # 分组
            query_to_indices = {}
            for i, q in enumerate(all_queries):
                query_to_indices.setdefault(q, []).append(i)

            group_labels = []
            group_scores = []

            for q, indices in query_to_indices.items():
                idx = np.array(indices)
                group_labels.append(all_labels[idx])
                group_scores.append(all_score_diffs[idx])

            # 统一计算所有排序指标
            group_metrics = compute_group_metrics(
                group_labels=group_labels,
                group_scores=group_scores,
                sample_size=sample_size,
                sample_ndcg=sample_ndcg,      # 这里控制NDCG是否抽样
                ndcg_ks=k_values,
                seed=seed
            )
            metrics.update(group_metrics)
        else:
            logger.warning("缺少 all_queries 或 all_score_diffs，无法计算排序指标")

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