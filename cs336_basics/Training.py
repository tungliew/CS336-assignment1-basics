import os
import torch
import torch.nn as nn
import argparse
import wandb
import numpy as np

from cs336_basics.DataLoader import get_batch
from cs336_basics.Loss import cross_entropy
from cs336_basics.CheckPointing import save_checkpoint, load_checkpoint

from cs336_basics.Transformer import TransformerLM
from cs336_basics.OptimizerAdamW import AdamW

from cs336_basics.GradientClipping import gradient_clipping
from cs336_basics.LearningRateSchedule import lr_cosine_schedule



# =========================================================
# Evaluation
# =========================================================
@torch.no_grad()
def evaluate(
    model,
    dataset,
    device,
    batch_size,
    context_length,
    eval_steps,
):
    model.eval()

    losses = []

    for _ in range(eval_steps):
        x, y = get_batch(dataset, batch_size, context_length, device,)

        logits = model(x)

        loss = cross_entropy(logits, y)

        losses.append(loss.item())

    mean_loss = sum(losses) / len(losses)

    model.train()

    return mean_loss


# =========================================================
# Training
# =========================================================
def train(
	model,
	optimizer,
	dataset_train,
	dataset_val,
	batch_size,
	context_length,
	max_iters,
	device,
	max_l2_norm, # glipping
	T_w, # learning rate schedule
	T_c, # learning rate schedule
	alpha_max, # learning rate schedule
	alpha_min, # learning rate schedule
	eval_interval=5,
	save_interval=5,
	eval_steps=10,
	checkpoint_path=None,  # saving

):
	model.to(device)
	model.train()

	best_val_loss = float("inf")
	val_loss = None


	# ----------------------------------------------------
	# WandB
	# -----------------------------------------------------
	wandb.init(
        project="cs336-transformer",
        config={
            "batch_size": batch_size,
            "context_length": context_length,
            "max_iters": max_iters,
            "max_l2_norm": max_l2_norm,
            "T_w": T_w,
            "T_c": T_c,
            "alpha_max": alpha_max,
            "alpha_min": alpha_min,
        },
    )

	# iteration nums
	for iteration in range(max_iters):
		optimizer.zero_grad()
		
		x, y = get_batch(dataset_train, batch_size, context_length, device)

		logits = model(x)

		loss = cross_entropy(logits, y)

		loss.backward()

		gradient_clipping(model.parameters(), max_l2_norm)


		learning_rate = lr_cosine_schedule(iteration, alpha_max, alpha_min, T_w, T_c)
		#optimizer.lr = learning_rate  # update learning rate
		for param_group in optimizer.param_groups:
			param_group["lr"] = learning_rate

		optimizer.step()
		print(f"finished iteration {iteration}")


		# logging
		if iteration % eval_interval == 0:
			val_loss = evaluate(
				model = model,
				dataset = dataset_val,
				device = device,
				batch_size = batch_size,
				context_length = context_length,
				eval_steps = eval_steps,
			)

			print(
				f"iter {iteration} | "
                f"train loss {loss.item():.4f} | "
                f"val loss {val_loss:.4f}"
                )

			wandb.log(
            {
                "iteration": iteration,
                "train_loss": loss.item(),
                "val_loss": val_loss,
                "learning_rate": learning_rate,
            }
        )

		# ---------------------------------------------
		# Save best model
		# ---------------------------------------------
		if (val_loss is not None) and val_loss < best_val_loss:
			best_val_loss = val_loss
			best_model_path = "/Users/tungliew/Downloads/assignment1-basics-main/models/best_model.pt"

			save_checkpoint(model, optimizer, iteration, best_model_path)

			print(
				f"New best model saved "
				f"with val loss {val_loss:.4f}"
			)

		# -----------------------------------------
		# checkpoint
		# -----------------------------------------
		if (checkpoint_path is not None and iteration % save_interval == 0):
			save_checkpoint(
				model,
				optimizer,
				iteration,
				checkpoint_path,
			)
			print(f"saved checkpoint to {checkpoint_path}")

	wandb.finish()


if __name__=="__main__":

	parser = argparse.ArgumentParser(description="Parse model hyperparameters")

	# Integer arguments
	parser.add_argument("--vocab_size", type=int, required=True, help="Vocabulary size")
	parser.add_argument("--context_length", type=int, required=True, help="Context length")
	parser.add_argument("--d_model", type=int, required=True, help="Dimension of model embeddings")
	parser.add_argument("--num_layers", type=int, required=True, help="Number of transformer layers")
	parser.add_argument("--num_heads", type=int, required=True, help="Number of attention heads")
	parser.add_argument("--d_ff", type=int, required=True, help="Feed-forward layer dimension")

	parser.add_argument("--batch_size", type=int, required=True, help="Batch size")

	parser.add_argument("--T_w", type=int, required=True, help="T_w for learning rate schedule")
	parser.add_argument("--T_c", type=int, required=True, help="T_c for learning rate schedule")
	parser.add_argument("--max_iters", type=int, required=True,)


	# Float argument
	parser.add_argument("--rope_theta", type=float, required=True, help="Rotary positional embedding theta")
	parser.add_argument("--weight_decay", type=float, required=False, help="Weight decay", default=0.0)

	parser.add_argument("--max_l2_norm", type=float, required=True, help="Max l2 norm for gradient clipping")

	parser.add_argument("--alpha_max", type=float, required=True, help="Alpha max for lr schedule")
	parser.add_argument("--alpha_min", type=float, required=True, help="Alpha min for lr schedule")


	# Str argument
	parser.add_argument("--data_path", type=str, required=True, help="data file")
	parser.add_argument("--val_path", type=str, required=True, help="validation file")
	parser.add_argument("--device", type=str, required=False, help="device", default="cpu",)
	parser.add_argument("--checkpoint_path", type=str, default="checkpoint.pt",)

	# get all arguments
	args = parser.parse_args() # args.vocab_size


	# =====================================================
	# Model
	# =====================================================
	model = TransformerLM(vocab_size = args.vocab_size,
		context_length = args.context_length,
		d_model = args.d_model,
		num_layers = args.num_layers,
		num_heads = args.num_heads,
		d_ff = args.d_ff,
		rope_theta = args.rope_theta,
	)


	# =====================================================
	# Optimizer
	# =====================================================
	optimizer = AdamW(model.parameters(), 
		lr = 1.0, 
		betas=(0.9,0.999), 
		eps=1e-8, 
		weight_decay=0.0
		)

	# =====================================================
	# Dataset
	# Replace with real datasets
	# =====================================================
	dataset_train = np.memmap(args.data_path, dtype=np.uint16, mode="r")
	dataset_val = np.memmap(args.val_path, dtype=np.uint16, mode="r")


	# =====================================================
	# Train
	# =====================================================
	train(
        model=model,
        optimizer=optimizer,
        dataset_train=dataset_train,
        dataset_val=dataset_val,
        batch_size=args.batch_size,
        context_length=args.context_length,
        max_iters=args.max_iters,
        device=args.device,
        max_l2_norm=args.max_l2_norm,
        T_w=args.T_w,
        T_c=args.T_c,
        alpha_max=args.alpha_max,
        alpha_min=args.alpha_min,
        checkpoint_path=args.checkpoint_path,
    )