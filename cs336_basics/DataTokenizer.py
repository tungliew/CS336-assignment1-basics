from transformers import AutoTokenizer
import numpy as np
from cs336_basics.BPETokenizer import Tokenizer
import pickle


file_path = "/Users/tungliew/Downloads/assignment1-basics-main/cs336_basics/results/TinyStoriesV2_GPT4_tokenizer.pkl"

vocab, merges = [], []
special_tokens=["<|endoftext|>"]

with open(file_path, "rb") as f:
    data = pickle.load(f)

    vocab = data["vocab"]
    merges = data["merges"]



    print("Vocab size:", len(vocab))
    for key, value in vocab.items():
        print(str(key) + " " + str(value))

    f.close()

# tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer = Tokenizer(vocab, merges, special_tokens)

input_path = "/Users/tungliew/Downloads/assignment1-basics-main/data/TinyStoriesV2-GPT4-valid.txt"
output_path = "/Users/tungliew/Downloads/assignment1-basics-main/data/TinyStoriesV2-GPT4-valid-tokens.bin"
chunk_size = 1000000 # character per chunk



# ---------------------------------------------------
# count total tokens
# train.txt 16020 tokens
# ---------------------------------------------------
total_tokens = 0
with open(input_path, "r", encoding="utf-8") as f:
    while True:
        text_chunk = f.read(chunk_size)

        if not text_chunk:
            break

        ids = tokenizer.encode(text_chunk)
        total_tokens += len(ids)
print(f"Total tokens: {total_tokens}")



# ---------------------------------------------------
# Create memmap file
# uint16 works for GPT2 vocab (50257 < 65535)
# ---------------------------------------------------
arr = np.memmap(
    output_path,
    dtype=np.uint16,
    mode="w+",
    shape=(total_tokens,),
)


# ---------------------------------------------------
# Second pass:
# tokenize + write incrementally
# ---------------------------------------------------
write_index = 0

with open(input_path, "r", encoding="utf-8") as f:

    while True:

        text_chunk = f.read(chunk_size)

        if not text_chunk:
            break

        ids = tokenizer.encode(text_chunk)

        ids_np = np.array(ids, dtype=np.uint16)

        arr[write_index : write_index + len(ids_np)] = ids_np

        write_index += len(ids_np)

        print(f"Written {write_index}/{total_tokens}")


# flush to disk
arr.flush()

print("Finished writing memmap dataset")


# ---------------------------------------------------
# Verify correctness
# ---------------------------------------------------
loaded = np.memmap(
    output_path,
    dtype=np.uint16,
    mode="r",
)

print("Dataset shape:", loaded.shape)
print("Min token:", loaded.min())
print("Max token:", loaded.max())

# assert loaded.max() < tokenizer.vocab_size

# train dataset
# Total tokens: 8332581


# test dataset
# Total tokens: 5474572