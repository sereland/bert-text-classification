import torch
from torch import nn
import torch.nn.functional as F
class MaskedListNetLoss(nn.Module):
    def forward(self, predicted_scores, true_labels, mask):
        """
        predicted_scores: (batch_size, max_list_size)
        true_labels: (batch_size, max_list_size) 
        mask: (batch_size, max_list_size)
        """
        batch_size = predicted_scores.size(0)
        
        # 对padding部分用极小的值替换，使得softmax时权重为0
        masked_scores = predicted_scores.masked_fill(~mask, -1e9)
        masked_labels = true_labels.masked_fill(~mask, -1e9)
        
        # 计算softmax概率分布（只对实际候选有效）
        P_pred = F.softmax(masked_scores, dim=-1)
        P_true = F.softmax(masked_labels, dim=-1)
        
        # 只计算实际候选的损失
        loss = - (P_true * torch.log(P_pred + 1e-9)) * mask
        loss = loss.sum(dim=1) / mask.sum(dim=1)  # 按实际候选数平均
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
        
        # 将标签转换为-1或1
        target = 2 * labels - 1
        
        # 计算交叉熵损失
        loss = torch.log(1 + torch.exp(-target * score_diff))
        
        return loss.mean()
    
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