# track time and memory usage
from BPETokenizer import train_bpe, save_tokenizer

import os

import cProfile
import pstats
import tracemalloc
import time

import json
import pickle


# vocab_size = 10,000
# special_token = "<|endoftext|>"
# vocab, merge = train_bpe()
# save vocab, merge -> the longest token in the vocab
# time and memory the training takes

def profile_train_bpe(input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs
):
    # Start memory tracking
    tracemalloc.start()

    # Start timing
    start_time = time.perf_counter()

    # Start CPU profiler
    profiler = cProfile.Profile()
    profiler.enable()

    # Run training
    vocab, merges = train_bpe(input_path, vocab_size, special_tokens)

    # Stop profiler
    profiler.disable()

    # End timing
    end_time = time.perf_counter()

    # Get memory stats
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Print wall-clock time
    print(f"\nExecution Time: {end_time - start_time:.4f} seconds")

    # Print memory usage
    print(f"Current Memory Usage: {current / 1024 / 1024:.4f} MB")
    print(f"Peak Memory Usage: {peak / 1024 / 1024:.4f} MB")

    # Print profiling stats
    print("\nTop functions by cumulative time:\n")

    stats = pstats.Stats(profiler)
    stats.sort_stats("cumulative")
    stats.print_stats(20)   # top 20 functions

    return vocab, merges


def post_processing(output_file, vocab, merges):
	longest = max(vocab.values(), key=len)

	print(f"the longest vocab: {longest}")
	print(f"It's length: {len(longest)}")

	save_tokenizer(output_file, vocab, merges)



if __name__=="__main__":
	input_path = "/Users/tungliew/Downloads/assignment1-basics-main/data/TinyStoriesV2-GPT4-train.txt"
	vocab_size = 10000
	special_tokens = ["<|endoftext|>"]
	vocab, merges = profile_train_bpe(
		input_path,
		vocab_size,
		special_tokens)
	output_file = "/Users/tungliew/Downloads/assignment1-basics-main/cs336_basics/results/TinyStoriesV2_GPT4_tokenizer.pkl"
	post_processing(output_file, vocab, merges)





# ------------------------------------------------------------
# results
# train time ~ 35min
# ------------------------------------------------------------
'''
Execution Time: 2033.8185 seconds
Current Memory Usage: 2.1246 MB
Peak Memory Usage: 130.8824 MB

Top functions by cumulative time:

         1637439243 function calls (1637439145 primitive calls) in 2033.804 seconds

   Ordered by: cumulative time
   List reduced from 180 to 20 due to restriction <20>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1   99.837   99.837 2033.804 2033.804 /Users/tungliew/Downloads/assignment1-basics-main/cs336_basics/BPETokenizer.py:113(train_bpe)
     9743  938.037    0.096 1575.183    0.162 /Users/tungliew/Downloads/assignment1-basics-main/cs336_basics/BPETokenizer.py:81(merge_pair)
384459829  403.830    0.000  403.830    0.000 {method 'append' of 'list' objects}
     9743  302.440    0.031  315.405    0.032 /Users/tungliew/Downloads/assignment1-basics-main/cs336_basics/BPETokenizer.py:65(get_pair_counts)
1073534148  240.888    0.000  240.888    0.000 {built-in method builtins.len}
  8263835   15.004    0.000   15.004    0.000 {method 'group' of '_regex.Match' objects}
  8263836   13.609    0.000   13.609    0.000 {method 'encode' of 'str' objects}
 42025351   12.557    0.000   12.557    0.000 /Users/tungliew/Downloads/assignment1-basics-main/cs336_basics/BPETokenizer.py:172(<genexpr>)
119536531    5.346    0.000    5.346    0.000 /Users/tungliew/.local/share/uv/python/cpython-3.13.5-macos-aarch64-none/lib/python3.13/collections/__init__.py:613(__missing__)
19488/19487    1.082    0.000    1.082    0.000 {built-in method builtins.max}
    41829    0.080    0.000    0.951    0.000 /Users/tungliew/Downloads/assignment1-basics-main/.venv/lib/python3.13/site-packages/regex/_main.py:350(finditer)
    41830    0.364    0.000    0.817    0.000 /Users/tungliew/Downloads/assignment1-basics-main/.venv/lib/python3.13/site-packages/regex/_main.py:459(_compile)
    83819    0.108    0.000    0.279    0.000 /Users/tungliew/.local/share/uv/python/cpython-3.13.5-macos-aarch64-none/lib/python3.13/enum.py:1607(__and__)
   251483    0.061    0.000    0.134    0.000 /Users/tungliew/.local/share/uv/python/cpython-3.13.5-macos-aarch64-none/lib/python3.13/enum.py:1589(_get_value)
    41830    0.069    0.000    0.096    0.000 <frozen importlib._bootstrap>:1390(_handle_fromlist)
     9743    0.067    0.000    0.077    0.000 /Users/tungliew/.local/share/uv/python/cpython-3.13.5-macos-aarch64-none/lib/python3.13/collections/__init__.py:599(__init__)
   461070    0.076    0.000    0.076    0.000 {built-in method builtins.isinstance}
    41829    0.056    0.000    0.056    0.000 {method 'finditer' of '_regex.Pattern' objects}
    41830    0.055    0.000    0.055    0.000 /Users/tungliew/Downloads/assignment1-basics-main/.venv/lib/python3.13/site-packages/regex/_main.py:481(complain_unused_args)
        1    0.000    0.000    0.046    0.046 /Users/tungliew/Downloads/assignment1-basics-main/.venv/lib/python3.13/site-packages/regex/_main.py:324(split)


the longest vocab: b' enthusiastically'
It's length: 17
'''