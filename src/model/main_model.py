import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple


class UnifiedDeepSpeech2(nn.Module):
    supported_rnns = {'lstm': nn.LSTM, 'gru': nn.GRU, 'rnn': nn.RNN}

    def __init__(
            self,
            input_dim: int,
            n_tokens: int,
            rnn_type='gru',
            num_rnn_layers: int = 5,
            rnn_hidden_dim: int = 512,
            dropout_p: float = 0.1,
            bidirectional: bool = True,
    ):
        super().__init__()

        self.in_channels, self.out_channels = 1, 32
        self._build_cnn_layers()
        self._setup_rnn_dimensions(rnn_hidden_dim)
        self._construct_rnn_stack(rnn_type, num_rnn_layers, dropout_p, bidirectional)
        self._build_classifier(n_tokens, rnn_hidden_dim)

    def _build_cnn_layers(self):
        conv_configs = [
            ((41, 11), (2, 2), (20, 5)),
            ((21, 11), (2, 1), (10, 5))
        ]
        layers = []
        in_ch = self.in_channels

        for kernel, stride, padding in conv_configs:
            layers.extend([
                nn.Conv2d(in_ch, self.out_channels, kernel, stride, padding),
                nn.BatchNorm2d(self.out_channels),
                nn.Hardtanh(0, 20, inplace=True)
            ])
            in_ch = self.out_channels

        self.conv_block = MaskedConvolution(nn.Sequential(*layers))

    def _setup_rnn_dimensions(self, rnn_hidden_dim):
        self.rnn_hidden_dim = rnn_hidden_dim
        rnn_input_size = 128  # n_mels

        for kernel, stride, _ in [(41, 2, None), (21, 2, None)]:
            rnn_input_size = self._compute_conv_output_size(rnn_input_size, kernel, stride)

        self.rnn_input_size = rnn_input_size * self.out_channels

    def _construct_rnn_stack(self, rnn_type, num_layers, dropout_p, bidirectional):
        self.rnn_stack = nn.ModuleList([
            EnhancedRNNLayer(
                self.rnn_input_size if i == 0 else self.rnn_hidden_dim,
                self.rnn_hidden_dim,
                rnn_type,
                bidirectional,
                dropout_p
            ) for i in range(num_layers)
        ])

    def _build_classifier(self, n_tokens, rnn_hidden_dim):
        self.classifier = nn.Sequential(
            nn.LayerNorm(rnn_hidden_dim),
            nn.Linear(rnn_hidden_dim, n_tokens, bias=False)
        )

    @staticmethod
    def _compute_conv_output_size(size, kernel, stride):
        return int(math.floor(size + 2 * (kernel // 2) - kernel) / stride + 1)

    def forward(self, spectrogram, spectrogram_length, **batch):
        x, lengths = self.conv_block(spectrogram.unsqueeze(1), spectrogram_length)

        # Reshape for RNN
        b, c, h, w = x.size()
        x = x.permute(0, 3, 1, 2).contiguous().view(b, w, c * h)

        # Process through RNN stack
        for rnn in self.rnn_stack:
            x = rnn(x, lengths)

        return {
            "log_probs": self.classifier(x).log_softmax(dim=-1),
            "log_probs_length": lengths
        }

    def transform_input_lengths(self, lengths):
        return lengths

    def __str__(self):
        params = {
            'total': sum(p.numel() for p in self.parameters()),
            'trainable': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }
        return (f"{super().__str__()}\n"
                f"All parameters: {params['total']}\n"
                f"Trainable parameters: {params['trainable']}")


class MaskedConvolution(nn.Module):
    def __init__(self, sequential):
        super().__init__()
        self.conv_seq = sequential

    def _compute_new_length(self, module, length):
        if isinstance(module, nn.Conv2d):
            return (length + 2 * module.padding[1] -
                    module.dilation[1] * (module.kernel_size[1] - 1) - 1
                    ).float().div(module.stride[1]).int() + 1
        return length

    def forward(self, x, seq_lengths):
        for module in self.conv_seq:
            x = module(x)
            mask = torch.zeros(x.size(), dtype=torch.bool, device=x.device)
            seq_lengths = self._compute_new_length(module, seq_lengths)

            for idx, length in enumerate(seq_lengths):
                if (mask[idx].size(2) - length) > 0:
                    mask[idx, :, :, length:] = True

            x.masked_fill_(mask, 0)

        return x, seq_lengths


class EnhancedRNNLayer(nn.Module):
    def __init__(self, input_size, hidden_size, rnn_type, bidirectional, dropout_p):
        super().__init__()

        self.bidirectional = bidirectional
        self.hidden_size = hidden_size

        self.norm = nn.BatchNorm1d(input_size)
        self.activation = nn.Hardtanh(0, 20, inplace=True)

        rnn_class = UnifiedDeepSpeech2.supported_rnns[rnn_type]
        self.rnn = rnn_class(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            bias=True,
            batch_first=True,
            dropout=dropout_p,
            bidirectional=bidirectional
        )

    def forward(self, x, lengths):
        x = self.activation(self.norm(x.transpose(1, 2))).transpose(1, 2)

        packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        output, _ = self.rnn(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(
            output,
            total_length=x.size(1),
            batch_first=True
        )

        if self.bidirectional:
            output = output.view(output.size(0), output.size(1), 2, -1)
            output = output.sum(2)

        return output