import re
from string import ascii_lowercase
import torch
import numpy as np
from pyctcdecode import build_ctcdecoder
from collections import defaultdict


class CTCTextEncoder:
    EMPTY_TOK = ""

    def __init__(self, alphabet=None, beam_size=10, lm = None, **kwargs):
        if alphabet is None:
            alphabet = list(ascii_lowercase + " ")

        self.alphabet = alphabet
        self.vocab = [self.EMPTY_TOK] + list(self.alphabet)

        self.ind2char = dict(enumerate(self.vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}

        self.beam_size = beam_size
        if lm:
            self.decoder = build_ctcdecoder(
                labels=self.alphabet,
                kenlm_model_path = lm
            )
        else:
            self.decoder = build_ctcdecoder(
                labels=self.alphabet,
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
        """
        Выполняет beam search декодирование используя pyctcdecode.

        Args:
            probs: Массив вероятностей формы (T, C), где T - длина последовательности,
                  C - размер словаря. Может быть torch.Tensor или np.ndarray.

        Returns:
            str: Декодированный текст.
        """
        # Проверяем тип входных данных и конвертируем при необходимости
        if isinstance(probs, torch.Tensor):
            probs = probs.detach().cpu().numpy()

        # Используем beam search из pyctcdecode
        beam_results = self.decoder.decode_beams(
            probs,
            beam_width=self.beam_size
        )

        # Берем лучший результат
        best_text = beam_results[0][0]

        return best_text

    @staticmethod
    def normalize_text(text: str):
        text = text.lower()
        text = re.sub(r"[^a-z ]", "", text)
        return text