import torch
import torch.nn as nn

from cs336_basics.Linear import Linear
from cs336_basics.RoPE import RoPE



def softmax(x, dim: int):
	if dim is None:
		max_value = torch.max(x)

		x_max = x - max_value

		output = torch.exp(x_max) / sum(torch.exp(x_max))

		return output
	
	else:
		max_value = torch.max(x, dim=dim, keepdim=True).values
		x = x - max_value
		
		exp_x = torch.exp(x)
		output = exp_x / torch.sum(exp_x, dim=dim, keepdim=True)

		return output


def silu(x):
	output = x * torch.sigmoid(x)
	return output


def scaled_dot_product_attention(Q, K, V, mask):
	d_k = torch.tensor(K.shape[-1])

	scores = Q @ K.transpose(-2,-1) / torch.sqrt(d_k)

	if mask is not None:
		scores = scores.masked_fill(mask==0, float("-inf"))

	scores = torch.softmax(scores, dim=-1)

	output = scores @ V 

	return output


class MultiheadSelfAttention(nn.Module):
	def __init__(self, d_model:int, num_heads: int):
		super().__init__()
		self.d_model = d_model
		self.num_heads = num_heads
		assert self.d_model % self.num_heads == 0

		self.head_dim = self.d_model // self.num_heads

		self.Q = Linear(d_model, d_model)
		self.K = Linear(d_model, d_model)
		self.V = Linear(d_model, d_model)
		self.output_proj = Linear(d_model, d_model)


	def forward(self, x):
		
		batch_size, seq_len, d_model = x.shape

		# (batch_size, seq_len, d_model)
		Q = self.Q(x)
		K = self.K(x)
		V = self.V(x)

		Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)
		K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)
		V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)

		mask = torch.tril(torch.ones(seq_len, seq_len))
		
		attention_output = scaled_dot_product_attention(Q, K, V, mask)

		attention_output = attention_output.transpose(1,2)
		attention_output = attention_output.contiguous().view(batch_size, seq_len, self.d_model)

		output = self.output_proj(attention_output)

		return output


# applying rope
class MultiheadSelfAttentionRope(nn.Module):
	def __init__(self, d_model:int, num_heads: int, theta:float, max_seq_len:int):
		super().__init__()
		self.d_model = d_model
		self.num_heads = num_heads
		assert self.d_model % self.num_heads == 0

		self.head_dim = self.d_model // self.num_heads
		self.theta = theta
		self.max_seq_len = max_seq_len

		self.Q = Linear(d_model, d_model)
		self.K = Linear(d_model, d_model)
		self.V = Linear(d_model, d_model)
		self.output_proj = Linear(d_model, d_model)

		self.rope = RoPE(self.theta, self.head_dim, self.max_seq_len)


	def forward(self, x, token_positions=None):
		
		batch_size, seq_len, d_model = x.shape

		# (batch_size, seq_len, d_model)
		Q = self.Q(x)
		K = self.K(x)
		V = self.V(x)

		Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)
		K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)
		V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)

		Q = self.rope(Q, token_positions)
		K = self.rope(K, token_positions)

		mask = torch.tril(torch.ones(seq_len, seq_len))
		
		attention_output = scaled_dot_product_attention(Q, K, V, mask)

		attention_output = attention_output.transpose(1,2)
		attention_output = attention_output.contiguous().view(batch_size, seq_len, self.d_model)

		output = self.output_proj(attention_output)

		return output
