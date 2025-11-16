import numpy as np
from sklearn.metrics import ndcg_score


def compute_ndcg_k_listwise(labels: np.ndarray,
                            scores: np.ndarray,
                            k: int = 10) -> float:
    """
    labels: [B, L]
    scores: [B, L]
    """
    batch = labels.shape[0]
    ndcgs = []

    for i in range(batch):
        l = labels[i]
        s = scores[i]

        if len(l) < 2:
            continue

        ndcg = ndcg_score([l], [s], k=min(k, len(l)))
        ndcgs.append(ndcg)

    if len(ndcgs) == 0:
        return 0.0

    return float(np.mean(ndcgs))


def compute_precision_at_1_listwise(labels: np.ndarray,
                                    scores: np.ndarray) -> float:
    """
    labels: [B, L]
    scores: [B, L]
    """
    batch = labels.shape[0]
    acc = []

    for i in range(batch):
        l = labels[i]
        s = scores[i]

        if len(l) < 2:
            continue

        true_top = np.argmax(l)
        pred_top = np.argmax(s)

        acc.append(1.0 if true_top == pred_top else 0.0)

    return float(np.mean(acc)) if acc else 0.0


def compute_mrr_listwise(labels: np.ndarray,
                         scores: np.ndarray) -> float:
    """
    labels: [B, L]
    scores: [B, L]
    """
    batch = labels.shape[0]
    mrrs = []

    for i in range(batch):
        l = labels[i]
        s = scores[i]

        if len(l) < 2:
            continue

        true_top = np.argmax(l)
        pred_rank = np.argsort(-scores[i])

        # find rank
        rank = np.where(pred_rank == true_top)[0][0] + 1
        mrrs.append(1.0 / rank)

    return float(np.mean(mrrs)) if mrrs else 0.0

class RankingMetricsCalculatorListwise:

    @staticmethod
    def calculate(
        all_labels: np.ndarray,    # [B, L]
        all_scores: np.ndarray,    # [B, L]
        k_values=[5, 10]
    ):
        metrics = {}

        # NDCG
        for k in k_values:
            metrics[f'ndcg@{k}'] = compute_ndcg_k_listwise(all_labels, all_scores, k)

        # precision@1
        metrics['precision@1'] = compute_precision_at_1_listwise(all_labels, all_scores)

        # MRR
        metrics['mrr'] = compute_mrr_listwise(all_labels, all_scores)

        return metrics
