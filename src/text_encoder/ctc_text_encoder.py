import re
from string import ascii_lowercase
import torch
import heapq
from collections import defaultdict
# TODO add BPE
# Note: think about metrics and encoder
# The design can be remarkably improved
# to calculate stuff more efficiently and prettier


class CTCTextEncoder:
    EMPTY_TOK = ""

    def __init__(self, alphabet=None, beam_size = 10, **kwargs):
        """
        Args:
            alphabet (list): alphabet for language. If None, it will be
                set to ascii
        """

        if alphabet is None:
            alphabet = list(ascii_lowercase + " ")

        self.alphabet = alphabet
        self.vocab = [self.EMPTY_TOK] + list(self.alphabet)

        self.ind2char = dict(enumerate(self.vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}

        self.beam_size = beam_size

    def __len__(self):
        return len(self.vocab)

    def __getitem__(self, item: int):
        assert type(item) is int
        return self.ind2char[item]

    def encode(self, text) -> torch.Tensor:
        text = self.normalize_text(text)
        try:
            return torch.Tensor([self.char2ind[char] for char in text]).unsqueeze(0)
        except KeyError:
            unknown_chars = set([char for char in text if char not in self.char2ind])
            raise Exception(
                f"Can't encode text '{text}'. Unknown chars: '{' '.join(unknown_chars)}'"
            )

    def decode(self, inds) -> str:
        """
        Raw decoding without CTC.
        Used to validate the CTC decoding implementation.

        Args:
            inds (list): list of tokens.
        Returns:
            raw_text (str): raw text with empty tokens and repetitions.
        """
        return "".join([self.ind2char[int(ind)] for ind in inds]).strip()

    def ctc_decode(self, inds) -> str:
        """
        Decodes a sequence of indices using CTC rules.
        Removes consecutive duplicates and ignores empty tokens.

        Args:
            inds (list or Tensor): list of token indices predicted by the model.
        Returns:
            decoded_text (str): the final decoded text.
        """
        if isinstance(inds, torch.Tensor):  # Convert Tensor to list if needed
            inds = inds.tolist()

        prev_token = None
        decoded_chars = []

        for token in inds:
            if token == 0:  # Ignore empty token (self.EMPTY_TOK)
                continue
            if token != prev_token:  # Avoid consecutive duplicates
                decoded_chars.append(self.ind2char[token])
            prev_token = token

        return "".join(decoded_chars).strip()

    def ctc_beam_search(self, probs: torch.Tensor) -> str:
        """
        Perform beam search decoding.

        Args:
            probs (torch.Tensor): Tensor of shape (T, C), where T is the sequence length
                                  and C is the number of classes (vocabulary size).

        Returns:
            str: Decoded text.
        """
        T, C = probs.shape
        beam = [(0.0, tuple())]  # Initialize beam with (log_prob, sequence)

        for t in range(T):
            next_beam = defaultdict(lambda: -float('inf'))

            for log_prob, seq in beam:
                for c in range(C):
                    new_seq = seq + (c,)  # Add new token to sequence
                    new_log_prob = log_prob + probs[t, c].item()

                    if new_log_prob > next_beam[new_seq]:
                        next_beam[new_seq] = new_log_prob

            # Keep top `beam_size` sequences
            beam = [(prob, seq) for seq, prob in
                    heapq.nlargest(self.beam_size, next_beam.items(), key=lambda x: x[1])]

        # Choose the sequence with the highest probability
        best_seq = max(beam, key=lambda x: x[0])[1]

        # Decode the sequence
        return self.ctc_decode(best_seq)

    @staticmethod
    def normalize_text(text: str):
        text = text.lower()
        text = re.sub(r"[^a-z ]", "", text)
        return text
