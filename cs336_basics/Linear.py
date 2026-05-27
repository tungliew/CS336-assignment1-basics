import torch
import torch.nn as nn
import math


class Linear(nn.Module):
	def __init__(self, in_features: int, out_features: int, device = None, dtype = None):
		super().__init__()
		
		weights = torch.empty(out_features, in_features, dtype=dtype, device=device)
		
		std = math.sqrt(2.0 / (in_features + out_features))
		
		torch.nn.init.trunc_normal_(weights,
			mean = 0.0,
			std = std,
			a = -3 * std,
			b = 3 * std
		)

		self.weights = nn.Parameter(weights)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		output = x @ self.weights.T

		return output
