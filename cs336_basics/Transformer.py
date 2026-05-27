import torch
import torch.nn as nn

from cs336_basics.Attention import MultiheadSelfAttentionRope
from cs336_basics.FeedForward import FeedForward
from cs336_basics.RMSNorm import RMSNorm
from cs336_basics.Embedding import Embedding
from cs336_basics.Linear import Linear

class TransformerBlock(nn.Module):
	def __init__(self, d_model:int, num_heads: int, d_ff:int, theta:float, max_seq_len:int):
		super().__init__()
		self.d_model = d_model
		self.num_heads = num_heads
		self.d_ff = d_ff
		self.theta = theta
		self.max_seq_len = max_seq_len

		self.attention = MultiheadSelfAttentionRope(self.d_model, self.num_heads, self.theta, self.max_seq_len)
		self.feedforward = FeedForward(self.d_model, self.d_ff)
		self.rms_norm_1 = RMSNorm(self.d_model)
		self.rms_norm_2 = RMSNorm(self.d_model)


	def forward(self, x):
		x = x + self.attention(self.rms_norm_1(x))
		x = x + self.feedforward(self.rms_norm_2(x))

		return x

	


class TransformerLM(nn.Module):
	def __init__(self, 
		vocab_size: int,
		context_length: int,
		d_model: int,
		num_layers: int,
		num_heads: int,
		d_ff: int,
		rope_theta: float,
	):
		super().__init__()
		self.vocab_size = vocab_size
		self.max_seq_len = context_length
		self.d_model = d_model
		self.num_layers = num_layers
		self.num_heads = num_heads
		self.d_ff = d_ff
		self.theta = rope_theta

		self.embed = Embedding(self.vocab_size, self.d_model)
		self.blocks = nn.Sequential(*[TransformerBlock(d_model = self.d_model, num_heads = self.num_heads, d_ff = self.d_ff, theta = self.theta, max_seq_len = self.max_seq_len)for _ in range(self.num_layers)])
		self.norm = RMSNorm(self.d_model)
		self.output_proj = Linear(self.d_model, self.vocab_size)

	def forward(self, x):
		x = self.embed(x)
		for block in self.blocks:
			x = block(x)

		output = self.output_proj(self.norm(x))

		return output
