# Transformer CSA
import torch
import torch.nn as nn

from cs336_basics.DeepSeek_V4.CompressedSparseAttention import CSA, LightingIndexer
from cs336_basics.FeedForward import FeedForward
from cs336_basics.RMSNorm import RMSNorm
from cs336_basics.Embedding import Embedding
from cs336_basics.Linear import Linear

class TransformerBlock(nn.Module):
	def __init__(self,
		d_ff: int,
		seq_len : int,
		d_model : int, 
		head_dim : int, 
		m : int, 
		d_c: int, 
		c_I : int,
		n_h_I: int,
		k : int,
		n_h: int, 
		c: int,
		g: int,
		d_g: int 
	):
		super().__init__()
		self.d_ff = d_ff
		self.seq_len = seq_len
		self.d_model = d_model
		self.head_dim = head_dim 
		self.m = m
		self.d_c = d_c 
		self.c_I = c_I
		self.n_h_I = n_h_I
		self.k = k
		self.n_h = n_h
		self.c = c
		self.g = g
		self.d_g = d_g

		self.csa_a = CSA(seq_len = self.seq_len, d_model=self.d_model, head_dim=self.head_dim, m=self.m)
		self.csa_k = CSA(seq_len = self.seq_len, d_model=self.d_model, head_dim=self.c_I, m=self.m)

		self.lighting_indexer = LightingIndexer(
			d_model = self.d_model,
			d_c = self.d_c,
			c_I = self.c_I, 
			n_h_I = self.n_h_I, 
			k = self.k, 
			m = self.m, 
			n_h = self.n_h, 
			c = self.c, 
			g = self.g, 
			d_g = self.d_g
		)


		# self.attention = MultiheadSelfAttentionRope(self.d_model, self.num_heads, self.theta, self.max_seq_len)
		self.feedforward = FeedForward(self.d_model, self.d_ff)
		self.rms_norm_1 = RMSNorm(self.d_model)
		self.rms_norm_2 = RMSNorm(self.d_model)


	def forward(self, x):
		h = x
		x = self.rms_norm_1(x)
		
		c_comp = self.csa_a(x)
		k_I_comp = self.csa_k(x)

		x = h + self.lighting_indexer(x, k_I_comp, c_comp)
		h = x
		x = h + self.feedforward(self.rms_norm_2(x))

		return x

	


class TransformerLM(nn.Module):
	def __init__(self, 
		vocab_size: int,
		context_length: int,
		d_model: int,
		num_layers: int,
		d_ff: int,
		head_dim: int,
		m : int, 
		d_c: int, 
		c_I : int,
		n_h_I: int,
		k : int,
		n_h: int, 
		c: int,
		g: int,
		d_g: int 
	):
		super().__init__()
		self.vocab_size = vocab_size
		self.seq_len = context_length
		self.d_model = d_model
		self.num_layers = num_layers
		self.d_ff = d_ff
		self.head_dim = head_dim
		self.m = m 
		self.d_c = d_c 
		self.c_I = c_I
		self.n_h_I = n_h_I
		self.k = k
		self.n_h = n_h
		self.c = c
		self.g = g
		self.d_g = d_g


		self.embed = Embedding(self.vocab_size, self.d_model)
		self.blocks = nn.Sequential(*[TransformerBlock(
			d_ff = self.d_ff,
			seq_len = self.seq_len,
			d_model = self.d_model, 
			head_dim = self.head_dim,
			m = self.m, 
			d_c = self.d_c, 
			c_I = self.c_I,
			n_h_I = n_h_I,
			k = self.k,
			n_h = self.n_h, 
			c = self.c,
			g = self.g,
			d_g = self.d_g
		) for _ in range(self.num_layers)])
		
		self.norm = RMSNorm(self.d_model)
		self.output_proj = Linear(self.d_model, self.vocab_size)

	def forward(self, x):
		x = self.embed(x)
		for block in self.blocks:
			x = block(x)

		output = self.output_proj(self.norm(x))

		return output
