import torch
import torch.nn as nn

class RoPE(nn.Module):
	def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
		super().__init__()
		self.d_k = d_k
		self.max_seq_len = max_seq_len
		self.theta = theta




	def create_Rik(self, theta, pos, k, d_model):
		angle = torch.tensor(pos / theta ** ((2*k)/d_model))
		sin = torch.sin(angle)
		cos = torch.cos(angle)
		Rik = torch.tensor([[cos, -sin],[sin, cos]])

		return Rik 

	
	def forward(self, x: torch.Tensor, token_positions: torch.Tensor):
		# (..., seq_len, d_k)
		seq_len, d_k = x.shape[-2], x.shape[-1]


		if token_positions is None:
			token_positions = torch.arange(seq_len)
		
		matrix_table = torch.ones((self.max_seq_len, self.d_k, self.d_k))

		for i in range(self.max_seq_len):
			blocks = [self.create_Rik(self.theta, i, k, self.d_k) for k in range(self.d_k // 2)]
			matrix_table[i, :, :] = torch.block_diag(*blocks)

		rotation_matrix = matrix_table[token_positions]

		output = rotation_matrix @ x.unsqueeze(-1)
		output = output.squeeze(-1)

		return output