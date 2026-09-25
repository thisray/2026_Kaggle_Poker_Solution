"""Minimal loss primitives for the full campaign. No data or pretrained model."""
import torch
from torch.nn import functional as F


def nnpu_loss(pos_logits,unlabeled_logits,prior):
    """nnPU surrogate. Prior/positive selection assumptions need sensitivity tests."""
    if not 0<prior<1 or not len(pos_logits) or not len(unlabeled_logits):raise ValueError('PU inputs')
    positive=prior*F.softplus(-pos_logits).mean()
    negative=F.softplus(unlabeled_logits).mean()-prior*F.softplus(pos_logits).mean()
    return positive+torch.clamp(negative,min=0.)


def soft_pseudo_loss(logits,probability,confidence):
    """Teacher targets are probabilities, not forced five-positive lists."""
    if logits.shape!=probability.shape or logits.shape!=confidence.shape:raise ValueError('shape')
    if torch.any((probability<0)|(probability>1)) or torch.any(confidence<0):raise ValueError('target')
    return (F.binary_cross_entropy_with_logits(logits,probability,reduction='none')*confidence).sum()/confidence.sum().clamp_min(1.)


def scalar_stochastic_bag_logits(instance_logits,mask,temperature=1.):
    """Length-normalized log-mean-exp; independent baseline for attention MIL."""
    if temperature<=0 or not mask.any(-1).all():raise ValueError('Empty bags/temperature')
    z=(instance_logits/temperature).masked_fill(~mask,-torch.inf)
    return temperature*(torch.logsumexp(z,-1)-mask.sum(-1).float().log())
