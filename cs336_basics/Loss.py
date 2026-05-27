import torch
import torch.nn as nn


def cross_entropy(inputs,  targets):
	max_logits = torch.max(inputs, dim=-1, keepdim=True).values

	stablized_logits = inputs - max_logits

	log_sum_exp = torch.log(torch.sum(torch.exp(stablized_logits), dim=-1, keepdim=True))

	log_probs = stablized_logits - log_sum_exp

	target_log_probs = torch.gather(log_probs, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

	loss = - target_log_probs

	return loss.mean()