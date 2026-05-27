import torch
import torch.nn as nn
import math

class Embedding(nn.Module):
	def __init__(self, num_embeddings:int, embedding_dim:int, device=None, dtype=None):
		super().__init__()
		
		table = torch.empty(num_embeddings, embedding_dim, device = device, dtype = dtype)

		std = 1

		torch.nn.init.trunc_normal_(
			table,
			mean = 0,
			std = 1.0,
			a = -3,
			b = 3)

		self.embed_table = nn.Parameter(table)

	def forward(self, token_ids: torch.Tensor):
		output = self.embed_table[token_ids]

		return output