import os
import regex as re
from collections import Counter, defaultdict
from typing import BinaryIO
from collections.abc import Iterable, Iterator
import pickle


def save_tokenizer(output_file, vocab, merges):
    with open(output_file, "wb") as f:
        pickle.dump(
            {
                "vocab": vocab,
                "merges": merges
            },
            f
        )


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    """
    assert isinstance(split_special_token, bytes)

    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)

        while True:
            mini_chunk = file.read(mini_chunk_size)

            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            found_at = mini_chunk.find(split_special_token)

            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break

            initial_position += mini_chunk_size

    return sorted(set(chunk_boundaries))


def get_pair_counts(
    word_freqs: dict[tuple[bytes, ...], int],
) -> Counter:
    """
    Count adjacent token pairs.
    """
    pair_counts = Counter()

    for word, freq in word_freqs.items():
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            pair_counts[pair] += freq

    return pair_counts


def merge_pair(
    pair: tuple[bytes, bytes],
    word_freqs: dict[tuple[bytes, ...], int],
) -> dict[tuple[bytes, ...], int]:
    """
    Merge all occurrences of `pair` in every word.
    """
    merged_token = pair[0] + pair[1]

    new_word_freqs = {}

    for word, freq in word_freqs.items():
        new_word = []

        i = 0
        while i < len(word):
            if (
                i < len(word) - 1
                and word[i] == pair[0]
                and word[i + 1] == pair[1]
            ):
                new_word.append(merged_token)
                i += 2
            else:
                new_word.append(word[i])
                i += 1

        new_word_freqs[tuple(new_word)] = freq

    return new_word_freqs


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train a byte-level BPE tokenizer.
    """

    # ------------------------------------------------------------
    # Initialize vocabulary
    # ------------------------------------------------------------

    vocab: dict[int, bytes] = {}

    # Special tokens first
    token_id = 0

    for token in special_tokens:
        vocab[token_id] = token.encode("utf-8")
        token_id += 1

    # All 256 byte values
    for byte_value in range(256):
        vocab[token_id] = bytes([byte_value])
        token_id += 1

    # ------------------------------------------------------------
    # Read corpus
    # ------------------------------------------------------------

    with open(input_path, "r", encoding="utf-8") as f:
        corpus = f.read()

    # ------------------------------------------------------------
    # Remove / split special tokens
    # ------------------------------------------------------------

    if special_tokens:
        split_pattern = "|".join(re.escape(tok) for tok in special_tokens)
        segments = re.split(split_pattern, corpus)
    else:
        segments = [corpus]

    # ------------------------------------------------------------
    # Pre-tokenization
    # ------------------------------------------------------------

    word_freqs: defaultdict[tuple[bytes, ...], int] = defaultdict(int)

    for segment in segments:

        for match in re.finditer(PAT, segment):
            pretoken = match.group(0)

            # Convert pretoken into tuple of single-byte bytes objects
            token_bytes = pretoken.encode("utf-8")

            byte_tuple = tuple(
                bytes([b]) for b in token_bytes
            )

            word_freqs[byte_tuple] += 1

    # ------------------------------------------------------------
    # Compute number of merges
    # ------------------------------------------------------------

    initial_vocab_size = len(vocab)

    num_merges = vocab_size - initial_vocab_size

    if num_merges < 0:
        raise ValueError(
            "vocab_size smaller than initial vocabulary size"
        )

    merges: list[tuple[bytes, bytes]] = []

    # ------------------------------------------------------------
    # BPE merge loop
    # ------------------------------------------------------------

    for _ in range(num_merges):

        pair_counts = get_pair_counts(word_freqs)

        if not pair_counts:
            break

        max_freq = max(pair_counts.values())

        # Tie-breaking:
        # choose lexicographically greatest pair
        best_pairs = [
            pair
            for pair, freq in pair_counts.items()
            if freq == max_freq
        ]

        best_pair = max(best_pairs)

        merges.append(best_pair)

        merged_token = best_pair[0] + best_pair[1]

        vocab[len(vocab)] = merged_token

        word_freqs = merge_pair(best_pair, word_freqs)

    return vocab, merges


class Tokenizer:

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):

        self.vocab = vocab
        self.merges = merges

        self.special_tokens = special_tokens or []

        # token bytes -> token id
        self.byte_to_id = {v: k for k, v in vocab.items()}

        # merge priority:
        # earlier merge = higher priority
        self.merge_ranks = {
            pair: i
            for i, pair in enumerate(merges)
        }

        # Add special tokens if missing
        for token in self.special_tokens:

            token_bytes = token.encode("utf-8")

            if token_bytes not in self.byte_to_id:

                new_id = len(self.vocab)

                self.vocab[new_id] = token_bytes
                self.byte_to_id[token_bytes] = new_id

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ):

        with open(vocab_filepath, "r", encoding="utf-8") as f:
            vocab_data = json.load(f)

        with open(merges_filepath, "r", encoding="utf-8") as f:
            merges_data = json.load(f)

        vocab = {
            int(k): v.encode("latin1")
            for k, v in vocab_data.items()
        }

        merges = [
            (
                a.encode("latin1"),
                b.encode("latin1"),
            )
            for a, b in merges_data
        ]

        return cls(vocab, merges, special_tokens)

    def _apply_bpe(
        self,
        token_bytes: bytes,
    ) -> list[bytes]:
        """
        Apply BPE merges to ONE pretoken.
        """

        # Start with single-byte tokens
        tokens = [
            bytes([b])
            for b in token_bytes
        ]

        while True:

            best_pair = None
            best_rank = float("inf")

            # Find highest-priority merge
            for i in range(len(tokens) - 1):

                pair = (tokens[i], tokens[i + 1])

                if pair in self.merge_ranks:

                    rank = self.merge_ranks[pair]

                    if rank < best_rank:
                        best_rank = rank
                        best_pair = pair

            # No more merges
            if best_pair is None:
                break

            # Merge all occurrences
            merged = []

            i = 0

            while i < len(tokens):

                if (i < len(tokens) - 1
                    and tokens[i] == best_pair[0]
                    and tokens[i + 1] == best_pair[1]
                ):

                    merged.append(best_pair[0] + best_pair[1])

                    i += 2

                else:
                    merged.append(tokens[i])
                    i += 1

            tokens = merged

        return tokens

    def encode(
        self,
        text: str,
    ) -> list[int]:

        token_ids = []

        # Handle special tokens first
        if self.special_tokens:

            sorted_special_tokens = sorted(
                self.special_tokens,
                key=len,
                reverse=True,
            )

            split_pattern = (
                "("
                + "|".join(
                    re.escape(tok)
                    for tok in sorted_special_tokens
                )
                + ")"
            )

            segments = re.split(split_pattern, text)

        else:
            segments = [text]

        for segment in segments:

            # Skip empty strings
            if segment == "":
                continue

            # If special token -> direct lookup
            if segment in self.special_tokens:

                token_bytes = segment.encode("utf-8")

                token_ids.append(self.byte_to_id[token_bytes])

                continue

            # Normal pre-tokenization
            for match in re.finditer(PAT, segment):

                pretoken = match.group(0)

                token_bytes = pretoken.encode("utf-8")

                bpe_tokens = self._apply_bpe(token_bytes)

                for tok in bpe_tokens:

                    token_ids.append(self.byte_to_id[tok])

        return token_ids

    def encode_iterable(
        self,
        iterable: Iterable[str],
    ) -> Iterator[int]:
        """
        Lazily tokenize chunks from an iterable.
        """

        for chunk in iterable:

            ids = self.encode(chunk)

            for token_id in ids:
                yield token_id

    def decode(
        self,
        ids: list[int],
    ) -> str:

        byte_sequence = b"".join(self.vocab[token_id] for token_id in ids)

        return byte_sequence.decode("utf-8", errors="replace",)



if __name__=="__main__":
    vocab, merges = train_bpe(
        input_path="/Users/tungliew/Downloads/assignment1-basics-main/data/train.txt",
        vocab_size=5000,
        special_tokens=["<|endoftext|>"]
    )

    output_path = "/Users/tungliew/Downloads/assignment1-basics-main/cs336_basics/results/bpe_tokenizer_test.pkl"
    save_tokenizer(output_path, vocab, merges)

    print("Vocab size:", len(vocab))
    for m in merges:
        print(m)