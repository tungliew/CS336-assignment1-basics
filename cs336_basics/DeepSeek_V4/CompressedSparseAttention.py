import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from cs336_basics.Linear import Linear
from cs336_basics.Attention import softmax

# =====================================================
# expand the input shape from (N, D) to (B, N, D)
# =====================================================

# compression process
class CSA(nn.Module):
	def __init__(self, seq_len, d_model, head_dim, m):
		super().__init__()
		self.n = seq_len
		self.d = d_model
		self.c = head_dim
		self.m = m

		assert seq_len % m == 0

		self.w_akv = Linear(self.d, self.c)
		self.w_bkv = Linear(self.d, self.c)
		self.w_za = Linear(self.d, self.c)
		self.w_zb = Linear(self.d, self.c)


		B_a = torch.zeros(self.m, self.c)
		B_b = torch.zeros(self.m, self.c)

		# learnable positional biases
		self.b_a = nn.Parameter(B_a)
		self.b_b = nn.Parameter(B_b)


	def forward(self, H):
		# H shape (B, N , D)
		B, n, D = H.shape
		##### assert n == self.n

		# shape (b,n,c)
		c_a, c_b = self.w_akv(H), self.w_bkv(H) # 2 kv series
		z_a, z_b  = self.w_za(H), self.w_zb(H)  # 2 compression weights

		##### num_blocks = self.n // self.m
		num_blocks = math.ceil(n / self.m)

		compressed = []

		for block_idx in range(num_blocks):
			'''
			start = block_idx * self.m 
			end = (block_idx + 1) * self.m

			# current block
			# shape (b, m, c)
			za_block = z_a[:, start:end, :] + self.b_a 
			ca_block = c_a[:, start:end, :]
			'''
			start = block_idx * self.m
			end = min(start + self.m, n)

			block_len = end - start

			za_block = (z_a[:, start:end, :] + self.b_a[:block_len])

			ca_block = c_a[:, start:end, :]

			if block_idx==0:
				zb_block = torch.full((B, self.m, self.c),
					float("-inf"),
					device = H.device,
					dtype = H.dtype
				)

				cb_block = torch.zeros(B, 
					self.m,
					self.c,
					device = H.device,
					dtype = H.dtype
				)

			else:
				prev_start = (block_idx-1) * self.m 
				prev_end = block_idx * self.m 

				zb_block = z_b[:, prev_start:prev_end, :] + self.b_b 
				cb_block = c_b[:, prev_start:prev_end, :]


			# shape (B, 2m, c)
			scores = torch.cat([za_block, zb_block], dim=1)

			# softmax over rows
			weights = softmax(scores, dim=1)

			'''
			s_a = weights[:, :self.m, :]
			s_b = weights[:, self.m:, :]
			'''
			s_a = weights[:, :block_len, :]
			s_b = weights[:, block_len:, :] 

			c_comp_i = (s_a * ca_block).sum(dim=1) + (s_b * cb_block).sum(dim=1)

			compressed.append(c_comp_i)

		c_comp = torch.stack(compressed, dim=1)

		return c_comp



class LightingIndexer(nn.Module):
	def __init__(self, d_model, d_c, c_I, n_h_I, k, m, n_h, c, g, d_g):
		super().__init__()
		self.d_model = d_model
		self.d_c = d_c # the query compression dimension
		self.c_I = c_I # the indexer head_dim
		self.n_h_I = n_h_I # the number of indexer query heads
		self.k = k  # top-k selection
		self.m = m
		self.n_h = n_h # the number of query heads
		self.c = c # head_dim

		# attention output projection
		assert d_g < c * n_h / g
		self.g = g  # split n_h into different groups
		self.d_g = d_g
		self.n_h_per_group = n_h // g



		self.w_dq = Linear(d_model, d_c) # down-projection matrix
		self.w_iuq = Linear(d_c, c_I * n_h_I) # up-projection matrix

		self.w_w = Linear(d_model, n_h_I)

		self.w_uq = Linear(d_c, c * n_h)

		w_group = torch.randn((self.g, self.n_h_per_group * self.c, self.d_g))
		self.w_group = torch.nn.Parameter(w_group)

		self.w_final = Linear(self.d_g * self.g, self.d_model)



	def forward(self, H, K_I_comp, c_comp):
		# H shape (n, d)
		# K_I_comp shape (n/m, c_I)

		B, n, D = H.shape
		n_blocks = K_I_comp.shape[1]

		# compressed latent vector for every query token
		c_q = self.w_dq(H) # (b, n, d_c)
		q_i = self.w_iuq(c_q) #(b, n, c_I * n_h_I)


		q_i = q_i.view(B, n, self.n_h_I, self.c_I) # (b, n, n_h_I, c_I)


		w_i = self.w_w(H) # (b, n, n_h_I)

		# q_i shape (b, n, n_h_I, c_I)
		# K_I_comp shape (b, n/m, c_I)
		# (b, n, n_h_I, n/m)
		dot_products = torch.einsum("bnhc, bsc -> bnhs", q_i, K_I_comp)


		dot_products = F.relu(dot_products)

		# (b, n, n_h_I, n/m)
		# (b, n, n_h_I)
		# (b, n, n_h_I, n/m)
		# (b, n, n/m) = (b, n, n_blocks)
		# for every token t, we have one score for every compressed block s
		index_scores = (dot_products * w_i.unsqueeze(-1)).sum(dim=2)


		# =====================================================
		# s < floor(t / m)
		# masking
		# =====================================================
		token_block_ids = (torch.arange(n, device=H.device) // self.m)
		block_ids = torch.arange(n_blocks,device=H.device)
		valid_mask = (block_ids.unsqueeze(0) <= token_block_ids.unsqueeze(1))
		index_scores = index_scores.masked_fill(~valid_mask.unsqueeze(0), float("-inf"))

		# =====================================================
		# top-k selection
		# =====================================================
		k = min(self.k, n_blocks)

		# (b, n, k)
		topk_scores, topk_indices = torch.topk(
			index_scores,
			k = k,
			dim = -1
		)

		# c_comp shape (b, n/m, c)
		# top_k indices (b, n, k)
		# c_t_SprsComp (b, n, k, c)
		batch_indices = torch.arange(B, device=H.device)[:, None, None]
		token_indices = torch.arange(n, device=H.device)[None, :, None]
		c_t_SprsComp = c_comp[batch_indices, topk_indices, :]  # (B, N, k, c)


		# attention queries
		# c_q shape (b, n, d_c)
		# w_uq shape (b, d_c, c * n_h)
		# q_t shape (b, n, c * n_h)
		q_t = self.w_uq(c_q)
		q_t = q_t. view(B, n, self.n_h, self.c)


		# =====================================================
		# Multi-Query Attention (MQA) on q_t and c_t_SprsComp
		# =====================================================
		# q_t shape (b n, n_h, c)
		# c_t_SprsComp (b n, k, c)
		# in torch.einsum h is n_h
		scores = torch.einsum("bnhc, bnkc -> bnhk",q_t, c_t_SprsComp) / (self.c ** 0.5)
		scores = softmax(scores, dim=-1)
		# o_t shape (b, n, n_h, c)
		o_t = torch.einsum("bnhk, bnkc -> bnhc", scores, c_t_SprsComp)


		# =====================================================
		# grouped output projection
		# =====================================================
		group_outputs = []

		for i in range(self.g):
			# (b, n, n_h_per_group, c)
			o_g = o_t[:, :, i*self.n_h_per_group:(i+1)*self.n_h_per_group, :]
			o_g = o_g.view(B, n, self.n_h_per_group * self.c)

			# (b, n, n_h_per_group * c)
			# (b, n_h_per_group * c, d_g)
			# (b, n, d_g)
			o_g_proj = o_g @ self.w_group[i]
			group_outputs.append(o_g_proj)

		# (b, n, d_g * g)
		concat = torch.cat(group_outputs, dim=-1)

		# (b, n, d_model)
		output = self.w_final(concat)

		return output





# =====================================================
# Simple Test
# =====================================================
if __name__=="__main__":

	batch_size = 2
	seq_len = 16
	d_model = 32
	head_dim = 8
	m = 4

	d_c = 16
	c_I = 4
	n_h_I = 2
	k = 3

	n_h = 4
	c = 8
	g = 2
	d_g = 12

	H = torch.randn(batch_size, seq_len, d_model)   # (n, d_model)
	print("H shape:", H.shape)


	csa_c = CSA(seq_len, d_model, head_dim, m)
	c_comp = csa_c(H)
	print("CSA compressed output shape:", c_comp.shape)

	csa_k = CSA(seq_len, d_model, c_I, m)
	k_I_comp = csa_k(H)
	print("K_I_comp shape:", k_I_comp.shape)

	indexer = LightingIndexer(d_model, d_c, c_I, n_h_I, k, m, n_h, c, g, d_g)

	output = indexer(H, k_I_comp, c_comp)
	print("output shape:", output.shape)


# =====================================================
# Result
# =====================================================
# H shape: torch.Size([2, 16, 32])
# CSA compressed output shape: torch.Size([2, 4, 8])
# K_I_comp shape: torch.Size([2, 4, 4])
# LightingIndexer output shape: torch.Size([2, 16, 32])