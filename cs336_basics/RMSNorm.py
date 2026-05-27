import torch
import torch.nn as nn

class RMSNorm(nn.Module):
	def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
		super().__init__()

		self.d_model = d_model
		self.eps = eps
		
		g = torch.ones(d_model, device=device, dtype=dtype)

		self.weight = nn.Parameter(g)


	def forward(self, x: torch.Tensor) -> torch.Tensor:
		# x shape (batch_size, sequence_length, d_model)
		in_dtype = x.dtype 
		x = x.to(torch.float32)

		rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)

		output = x / rms * self.weight

		return output.to(in_dtype)