import torch
import torch.nn.functional as F
import torch.nn as nn

def masked_mse_loss(
    input: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """
    Compute the masked MSE loss between input and target.
    """
    mask = mask.float()
    loss = F.mse_loss(input * mask, target * mask, reduction="sum")
    return loss / mask.sum()


def masked_cross_entry_loss(input: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    logits = input.view(-1, input.size(-1))  # (batch_size * seq_len, bin_size)
    target = target.view(-1)  # (batch_size * seq_len)
    masked_positions = mask.view(-1)  # (batch_size * seq_len)
    loss = F.cross_entropy(
        logits[masked_positions],
        target[masked_positions].long(),
        reduction='mean'
    )
    return loss


def criterion_neg_log_bernoulli(
    input: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """
    Compute the negative log-likelihood of Bernoulli distribution
    """
    mask = mask.float()
    bernoulli = torch.distributions.Bernoulli(probs=input)
    masked_log_probs = bernoulli.log_prob((target > 0).float()) * mask
    return -masked_log_probs.sum() / mask.sum()


def masked_relative_error(
    input: torch.Tensor, target: torch.Tensor, mask: torch.LongTensor
) -> torch.Tensor:
    """
    Compute the masked relative error between input and target.
    """
    assert mask.any()
    loss = torch.abs(input[mask] - target[mask]) / (target[mask] + 1e-6)
    return loss.mean()



class MultiTaskLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_sigma1 = nn.Parameter(torch.tensor(0.0))  # for MLM
        self.log_sigma2 = nn.Parameter(torch.tensor(0.0))  # for Topo
        self.log_sigma3 = nn.Parameter(torch.tensor(0.0))  # for MVC

    def forward(self, mlm_loss, topo_loss=None, mvc_loss=None):
        loss = 0.0
        if mlm_loss is not None:
            loss += torch.exp(-self.log_sigma1) * mlm_loss + self.log_sigma1
        if topo_loss is not None:
            loss += torch.exp(-self.log_sigma2) * topo_loss + self.log_sigma2
        if mvc_loss is not None:
            loss += torch.exp(-self.log_sigma3) * mvc_loss + self.log_sigma3
        return loss
