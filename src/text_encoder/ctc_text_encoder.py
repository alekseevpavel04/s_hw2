import re
from string import ascii_lowercase
import torch
import numpy as np
from pyctcdecode import build_ctcdecoder

class CTCTextEncoder:
    EMPTY_TOK = ""

    def __init__(self, alphabet=None, beam_size=100, lm = None, vocab = None, **kwargs):
        if alphabet is None:
            alphabet = list(ascii_lowercase + " ")

        self.alphabet = alphabet
        self.vocab = [self.EMPTY_TOK] + list(self.alphabet)

        self.ind2char = dict(enumerate(self.vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}

        self.beam_size = beam_size
        self.lm = lm

        with open(vocab) as f:
            unigrams = [x.strip() for x in f.readlines()]

        if lm:
            self.decoder = build_ctcdecoder(
                [w.upper() for w in self.vocab],
                kenlm_model_path = lm,
                unigrams = unigrams
            )
        else:
            self.decoder = build_ctcdecoder(
                labels=self.vocab,
            )

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
        return "".join([self.ind2char[int(ind)] for ind in inds]).strip()

    def ctc_decode(self, inds) -> str:
        if isinstance(inds, torch.Tensor):
            inds = inds.tolist()

        prev_token = None
        decoded_chars = []

        for token in inds:
            if token == 0:
                continue
            if token != prev_token:
                decoded_chars.append(self.ind2char[token])
            prev_token = token

        return "".join(decoded_chars).strip()

    def ctc_beam_search(self, probs) -> str:
        if isinstance(probs, torch.Tensor):
            probs = probs.cpu().numpy()

        # Применяем softmax для преобразования логитов в вероятности
        def softmax(x):
            exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
            return exp_x / np.sum(exp_x, axis=1, keepdims=True)

        probs = softmax(probs)

        return self.normalize_text(self.decoder.decode(probs, beam_width=self.beam_size))

    @staticmethod
    def normalize_text(text: str):
        text = text.lower()
        text = re.sub(r"[^a-z ]", "", text)
        return text