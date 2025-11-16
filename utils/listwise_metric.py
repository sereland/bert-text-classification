import numpy as np
from sklearn.metrics import ndcg_score


def compute_ndcg_k_listwise(labels: np.ndarray,
                            scores: np.ndarray,
                            mask: np.ndarray,
                            k: int = 10) -> float:
    """
    labels, scores, mask: [N, L]
    """
    N = labels.shape[0]
    ndcgs = []

    for i in range(N):
        l = labels[i]
        s = scores[i]
        m = mask[i]

        # 取有效文档
        valid_l = l[m == 1]
        valid_s = s[m == 1]

        if len(valid_l) < 2:
            continue

        ndcg = ndcg_score([valid_l], [valid_s], k=min(k, len(valid_l)))
        ndcgs.append(ndcg)

    return float(np.mean(ndcgs)) if ndcgs else 0.0


def compute_precision_at_1_listwise(labels: np.ndarray,
                                    scores: np.ndarray,
                                    mask: np.ndarray) -> float:
    N = labels.shape[0]
    acc = []

    for i in range(N):
        l = labels[i]
        s = scores[i]
        m = mask[i]

        valid_l = l[m == 1]
        valid_s = s[m == 1]

        if len(valid_l) < 2:
            continue

        true_top = np.argmax(valid_l)
        pred_top = np.argmax(valid_s)

        acc.append(1.0 if true_top == pred_top else 0.0)

    return float(np.mean(acc)) if acc else 0.0


def compute_mrr_listwise(labels: np.ndarray,
                         scores: np.ndarray,
                         mask: np.ndarray) -> float:
    N = labels.shape[0]
    mrrs = []

    for i in range(N):
        l = labels[i]
        s = scores[i]
        m = mask[i]

        valid_l = l[m == 1]
        valid_s = s[m == 1]

        if len(valid_l) < 2:
            continue

        true_top = np.argmax(valid_l)
        pred_rank = np.argsort(-valid_s)

        rank = np.where(pred_rank == true_top)[0][0] + 1
        mrrs.append(1.0 / rank)

    return float(np.mean(mrrs)) if mrrs else 0.0

def compute_topk_set_match(labels: np.ndarray,
                           scores: np.ndarray,
                           mask: np.ndarray,
                           k: int = 2) -> float:
    """
    计算Top-k集合匹配指标（顺序不敏感）
    
    labels, scores, mask: [N, L]
    k: Top-k
    """
    N = labels.shape[0]
    matches = []

    for i in range(N):
        l = labels[i]
        s = scores[i]
        m = mask[i]

        valid_l = l[m == 1]
        valid_s = s[m == 1]

        if len(valid_l) < 2:
            continue

        # 取预测前k和真实前k的索引
        topk_pred = set(np.argsort(-valid_s)[:k])
        topk_true = set(np.argsort(-valid_l)[:k])

        # 判断集合是否完全一致
        matches.append(1.0 if topk_pred == topk_true else 0.0)

    return float(np.mean(matches)) if matches else 0.0

class RankingMetricsCalculatorListwise:

    @staticmethod
    def calculate(
        all_labels: np.ndarray,     # [N, L]
        all_scores: np.ndarray,     # [N, L]
        all_masks: np.ndarray,       # [N, L]
        k_values=[5, 10]
    ):
        metrics = {}

        # NDCG
        for k in k_values:
            metrics[f'ndcg@{k}'] = compute_ndcg_k_listwise(
                all_labels, all_scores, all_masks, k
            )

        # precision@1
        metrics['precision@1'] = compute_precision_at_1_listwise(
            all_labels, all_scores, all_masks
        )

        # MRR
        metrics['mrr'] = compute_mrr_listwise(
            all_labels, all_scores, all_masks
        )
        
        # Top-2 Set Match
        metrics['top2_set_match'] = compute_topk_set_match(
            all_labels, all_scores, all_masks, k=2
        )

        return metrics