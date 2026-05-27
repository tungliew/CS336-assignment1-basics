from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math

class AdamW(torch.optim.Optimizer):
	def __init__(self, params, lr=1.0, betas=(0.9,0.999), eps=1e-8, weight_decay=0.0):
		if lr < 0:
			raise ValueError(f"Invalid learning rate: {lr}")

		beta1, beta2 = betas
		
		defaults = {"lr":lr, "betas":betas, "eps":eps, "weight_decay":weight_decay}
		super().__init__(params, defaults)


	def step(self, closure: Optional[Callable] = None):
		loss = None if closure is None else closure()
		for group in self.param_groups:
			lr = group["lr"] # Get the learning rate.
			beta1, beta2 = group["betas"]
			eps = group["eps"]
			weight_decay = group["weight_decay"]
			
			# for every parameters in "params"
			for p in group["params"]:
				if p.grad is None:
					continue

				grad = p.grad.data # Get the gradient of loss with respect to p.
				
				state = self.state[p] # Get state associated with p.

				if len(state)==0:
					state["t"] = 0
					state["m"] = torch.zeros_like(p.data)
					state["v"] = torch.zeros_like(p.data)

				m = state["m"]
				v = state["v"]

				state['t'] += 1
				t = state["t"] # Get iteration number from the state, or 0.
				
				lr_t = (lr * math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t))

				# apply weight decay
				# 𝜃 ← 𝜃 − 𝛼𝜆𝜃 ▷ Apply weight decay
				p.data -= lr * weight_decay * p.data # Update weight tensor in-place.
				

				m.mul_(beta1).add_(grad * (1-beta1))
				v.mul_(beta2).add_(grad * grad * (1-beta2))

				p.data -= lr_t * m / (torch.sqrt(v) + eps)


		return loss