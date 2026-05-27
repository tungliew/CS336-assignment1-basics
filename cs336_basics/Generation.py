import torch
import torch.nn as nn
import pickle

from cs336_basics.Transformer import TransformerLM
from cs336_basics.OptimizerAdamW import AdamW
from cs336_basics.BPETokenizer import Tokenizer
from cs336_basics.Attention import softmax
from cs336_basics.CheckPointing import load_checkpoint


def generation(
    model,
    tokenizer,
    prompt: str,
    max_length: int = 100,
    temperature: float = 1.0,
    p_value: float = 0.9,
    device: str = "cpu",
):

    model.eval() # model set to evaluation mode

    # -------------------------------------------------
    # Encode prompt
    # -------------------------------------------------

    input_ids = tokenizer.encode(prompt)

    # Convert to tensor
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device,)

    # Find EOS token if exists
    eos_id = None

    if "<|endoftext|>" in tokenizer.special_tokens:

        eos_id = tokenizer.byte_to_id[b"<|endoftext|>"]

    # -------------------------------------------------
    # Autoregressive generation loop
    # -------------------------------------------------

    with torch.no_grad():

        for _ in range(max_length):

            logits = model(input_tensor) # shape (B, T, vocab_size)

            
            # Get logits for next token
            # We take the last element of this matrix
            next_token_logits = logits[:, -1, :]

            
            # Temperature scaling
            next_token_logits = (next_token_logits / temperature)


            probs = softmax(next_token_logits,dim=-1,)

            
            # Top-p (nucleus) sampling
            sorted_probs, sorted_indices = torch.sort(probs,descending=True,)

            cumulative_probs = torch.cumsum(sorted_probs,dim=-1,)

            # Keep smallest set with cumulative prob >= p
            mask = cumulative_probs <= p_value

            # Ensure at least 1 token kept
            mask[..., 0] = True

            filtered_probs = sorted_probs * mask

            # Renormalize
            filtered_probs = (filtered_probs / filtered_probs.sum(dim=-1, keepdim=True))

            # Sample token
            sampled_index = torch.multinomial(filtered_probs,num_samples=1,)

            # Convert back to original vocab index
            next_token = torch.gather(sorted_indices,-1,sampled_index,)

            # -----------------------------------------
            # Append token
            # -----------------------------------------

            input_tensor = torch.cat([input_tensor, next_token],dim=1,)

            # -----------------------------------------
            # Stop if EOS generated
            # -----------------------------------------

            if (eos_id is not None and next_token.item() == eos_id):
            	break

    generated_ids = input_tensor[0].tolist()

    generated_text = tokenizer.decode(generated_ids)

    return generated_text



if __name__=="__main__":
	model = TransformerLM(
		vocab_size = 50257,
		context_length = 256,
		d_model = 512,
		num_layers = 4,
		num_heads = 16,
		d_ff = 1344,
		rope_theta = 10000.0
	)

	optimizer = AdamW(
		model.parameters(),
		lr = 3e-4
	)


	file_path = "/Users/tungliew/Downloads/assignment1-basics-main/cs336_basics/results/TinyStoriesV2_GPT4_tokenizer.pkl"

	vocab, merges = [], []
	special_tokens=["<|endoftext|>"]

	with open(file_path, "rb") as f:
		data = pickle.load(f)


		vocab = data["vocab"]

		merges = data["merges"]

		f.close()

	tokenizer = Tokenizer(vocab, merges, special_tokens)

	model_path = "/Users/tungliew/Downloads/assignment1-basics-main/models/best_model.pt"
	iteration = load_checkpoint(
		src = model_path,
		model = model,
		optimizer = optimizer
	)

	device = "cpu"
	model.to(device)


	prompt = "The dragon opened the door"
	generated = generation(
		model = model,
		tokenizer = tokenizer,
		prompt = prompt,
		max_length = 1000,
		temperature = 0.8,
		p_value = 0.9,
		device = device,
	)


	print(generated)


'''
The dragon opened the door. The bird said, "I love you for the bird. What fly."
"Maybe you have a small forest!" laughed and Lily. Tom said, "That's you sad, Lily. I don'
"I am sad and play with you," was so much.
One day, Tom can the heavy tree.
"No, you are not many boat's, and be careful. Tim's okay, I want to be careful on the good
Tom's mom went to the bunny and the dog and the car every day.
"I will go. You don't find his friends."
"OK, I'm proud of the ball's friend. She wanted to be toys. You can play together. He foun
The dog was scared. She decided to give her friends. The boy saw the tree and was so tired
<|endoftext|>
'''