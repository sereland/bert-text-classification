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


class RankingMetricsCalculatorListwise:

    @staticmethod
    def calculate(
        all_labels: np.ndarray,     # [N, L]
        all_scores: np.ndarray,     # [N, L]
        all_mask: np.ndarray,       # [N, L]
        k_values=[5, 10]
    ):
        metrics = {}

        # NDCG
        for k in k_values:
            metrics[f'ndcg@{k}'] = compute_ndcg_k_listwise(
                all_labels, all_scores, all_mask, k
            )

        # precision@1
        metrics['precision@1'] = compute_precision_at_1_listwise(
            all_labels, all_scores, all_mask
        )

        # MRR
        metrics['mrr'] = compute_mrr_listwise(
            all_labels, all_scores, all_mask
        )

        return metrics