import torch
from torch import nn
import torch.nn.functional as F
# class MaskedListNetLoss(nn.Module):
#     def forward(self, predicted_scores, true_labels, mask):
#         """
#         predicted_scores: (batch_size, max_list_size)
#         true_labels: (batch_size, max_list_size) 
#         mask: (batch_size, max_list_size)
#         """
#         batch_size = predicted_scores.size(0)
        
#         # 对padding部分用极小的值替换，使得softmax时权重为0
#         masked_scores = predicted_scores.masked_fill(~mask, -1e9)
#         masked_labels = true_labels.masked_fill(~mask, -1e9)
        
#         # 计算softmax概率分布（只对实际候选有效）
#         P_pred = F.softmax(masked_scores, dim=-1)
#         P_true = F.softmax(masked_labels, dim=-1)
        
#         # 只计算实际候选的损失
#         loss = - (P_true * torch.log(P_pred + 1e-9)) * mask
#         loss = loss.sum(dim=1) / mask.sum(dim=1)  # 按实际候选数平均
#         return loss.mean()

# class MaskedListNetLoss(nn.Module):
#     def __init__(self, tau: float = 1.0):
#         super().__init__()
#         self.tau = tau

#     def forward(self, predicted_scores, true_labels, mask):
#         # predicted_scores: (B, L)
#         # true_labels: (B, L)
#         # mask: (B, L) bool
#         mask = mask.bool()
#         # 屏蔽 pad 位置
#         neg_inf = -1e9
#         # 将 masked positions 置为 -inf
#         masked_scores = predicted_scores.masked_fill(~mask, neg_inf)
#         masked_labels = true_labels.masked_fill(~mask, neg_inf)

#         # 预测分布
#         P_pred = F.softmax(masked_scores, dim=-1)

#         # 目标分布：对 labels 做 softmax（可加温度）
#         P_true = F.softmax(masked_labels / self.tau, dim=-1)

#         # stable log
#         log_P_pred = torch.log(P_pred + 1e-12)

#         # cross entropy per row (sum over valid positions)
#         per_row_loss = - (P_true * log_P_pred).sum(dim=1)  # already sums only over valid positions because P_true at pad is ~0

#         # normalize by number of valid items to make loss scale invariant
#         denom = mask.sum(dim=1).float().clamp_min(1.0)  # 防止除0
#         per_row_loss = per_row_loss / denom

#         return per_row_loss.mean()

class MaskedListNetLoss(nn.Module):
    def forward(self, predicted_scores, true_labels, mask):
        """
        predicted_scores: (B, L) raw scores from model
        true_labels: (B, L) CTR values
        mask: (B, L) bool
        """
        batch_size, list_size = predicted_scores.shape

        # Step 1: 对每个 query 内的 true_labels 做 per-query 归一化 → 排序分数
        true_scores = torch.zeros_like(true_labels)
        for i in range(batch_size):
            valid_labels = true_labels[i][mask[i]]
            if len(valid_labels) > 1:
                # 方案1：z-score
                mean = valid_labels.mean()
                std = valid_labels.std(unbiased=False)
                normalized = (valid_labels - mean) / (std + 1e-8)
                true_scores[i][mask[i]] = normalized
            else:
                true_scores[i][mask[i]] = 0.0

        # Step 2: mask padding
        masked_pred = predicted_scores.masked_fill(~mask, -1e9)
        masked_true = true_scores.masked_fill(~mask, 0.0)  # 0 for padding

        # Step 3: softmax to probability
        P_pred = F.softmax(masked_pred, dim=-1)
        P_true = F.softmax(masked_true, dim=-1)

        # Step 4: cross entropy
        loss = - P_true * torch.log(P_pred + 1e-10)
        loss = loss.sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return loss.mean()


class PairwiseLoss(nn.Module):
    """成对损失函数基类"""
    
    def __init__(self, loss_type: str = "ranknet"):
        """
        初始化成对损失函数
        
        Args:
            loss_type: 损失函数类型 (ranknet, margin_ranking, bpr)
        """
        super().__init__()
        self.loss_type = loss_type
        
    def forward(self, scores1: torch.Tensor, scores2: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        计算成对损失
        
        Args:
            scores1: 第一个文本的得分 [batch_size]
            scores2: 第二个文本的得分 [batch_size]
            labels: 标签 [batch_size] (1表示text1优于text2，0表示text2优于text1)
            
        Returns:
            损失值
        """
        if self.loss_type == "ranknet":
            return self._ranknet_loss(scores1, scores2, labels)
        elif self.loss_type == "margin_ranking":
            return self._margin_ranking_loss(scores1, scores2, labels)
        elif self.loss_type == "bpr":
            return self._bpr_loss(scores1, scores2, labels)
        else:
            raise ValueError(f"不支持的损失函数类型: {self.loss_type}")
    
    def _ranknet_loss(self, scores1: torch.Tensor, scores2: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        RankNet损失函数
        
        Args:
            scores1: 第一个文本的得分
            scores2: 第二个文本的得分
            labels: 标签
            
        Returns:
            RankNet损失
        """
        # 计算得分差
        score_diff = scores1 - scores2
        
        # # 将标签转换为-1或1
        # target = 2 * labels - 1
        
        # # 计算交叉熵损失
        # loss = torch.log(1 + torch.exp(-target * score_diff))

        # loss = F.softplus(-score_diff) # 形状: (batch_size, 1)
        # 计算a比b好的概率
        # prob_a_better = torch.sigmoid(score_diff)
        
        # # 二分类交叉熵损失
        # loss = F.binary_cross_entropy(prob_a_better, labels.float())
        
        # return loss.mean()
        loss = F.binary_cross_entropy_with_logits(
            score_diff,
            labels.float()
        )

        return loss
    
    def _margin_ranking_loss(self, scores1: torch.Tensor, scores2: torch.Tensor, labels: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
        """
        Margin Ranking损失函数
        
        Args:
            scores1: 第一个文本的得分
            scores2: 第二个文本的得分
            labels: 标签
            margin: 边界值
            
        Returns:
            Margin Ranking损失
        """
        # 将标签转换为-1或1
        target = 2 * labels - 1
        
        # 计算损失
        loss = torch.relu(-target * (scores1 - scores2) + margin)
        
        return loss.mean()
    
    def _bpr_loss(self, scores1: torch.Tensor, scores2: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Bayesian Personalized Ranking (BPR) 损失函数
        
        Args:
            scores1: 第一个文本的得分
            scores2: 第二个文本的得分
            labels: 标签
            
        Returns:
            BPR损失
        """
        # 只处理正样本对（labels=1）
        mask = labels == 1
        if not mask.any():
            return torch.tensor(0.0, device=scores1.device, requires_grad=True)
        
        # 计算得分差
        score_diff = scores1[mask] - scores2[mask]
        
        # 计算BPR损失
        loss = -torch.log(torch.sigmoid(score_diff))
        
        return loss.mean()

if __name__ == "__main__":
    # 简单测试
    loss_fn = MaskedListNetLoss(tau=1.0)
    predicted = torch.tensor([[2.0, 1.0, 0.5, -1e9],
                              [0.5, 1.5, -1e9, -1e9]])
    labels = torch.tensor([[3.0, 2.0, 1.0, -1e9],
                           [1.0, 2.0, -1e9, -1e9]])
    mask = torch.tensor([[1, 1, 1, 0],
                         [1, 1, 0, 0]], dtype=torch.bool)
    
    loss = loss_fn(predicted, labels, mask)
    print(f"Masked ListNet Loss: {loss.item()}")

    # 测试损失函数
    loss_fn = MaskedListNetLoss()
    
    # 模拟数据
    predicted = torch.tensor([[1.0, 2.0, 3.0], [2.0, 1.0, 3.0]])
    true_labels = torch.tensor([[0.1, 0.2, 0.3], [0.3, 0.1, 0.2]])  # CTR值
    mask = torch.tensor([[1, 1, 1], [1, 1, 0]])
    
    loss = loss_fn(predicted, true_labels, mask)
    print(f"Test loss: {loss}")