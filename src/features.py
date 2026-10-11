"""Generation-Signal View: per-token entropy and confidence from raw logits."""
import numpy as np
import torch


def generation_signals(step_logits, token_ids):
    """
    step_logits: list of T tensors, each [V] (raw logits, no temperature/top-k applied)
    token_ids:   LongTensor [T] of the tokens actually generated
    Returns per-token entropy and confidence as numpy arrays.
    """
    logits = torch.stack(step_logits).float()          # [T, V]
    logp = torch.log_softmax(logits, dim=-1)
    p = logp.exp()
    entropy = -(p * logp).sum(-1)                      # H_t
    conf = p.gather(1, token_ids.view(-1, 1)).squeeze(1)  # c_t
    return entropy.cpu().numpy(), conf.cpu().numpy()


def summarize(entropy, conf):
    """Sequence-level features: [mean entropy, mean confidence, confidence variance]."""
    return [float(entropy.mean()), float(conf.mean()), float(conf.var())]  # population variance (1/T)