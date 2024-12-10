import torch
import torch.nn as nn


class MaskedConvBlock(nn.Module):
    """
    Модуль свёрточных слоёв с маскированием для работы с последовательностями переменной длины
    """

    def __init__(self, layers):
        """
        Args:
            layers (nn.Sequential): Последовательность свёрточных слоёв
        """
        super().__init__()
        self.layers = layers

    def forward(self, x, seq_lengths):
        """
        Применяет свёрточные слои с маскированием для обработки переменной длины входных данных.

        Args:
            x (Tensor): Входной тензор с размерностью [B, 1, F, T].
            seq_lengths (Tensor): Длины последовательностей [B].

        Returns:
            Tensor: Результат после применения слоёв [B, C, F', T'].
            Tensor: Обновлённые длины последовательностей [B].
        """
        for layer in self.layers:
            x = layer(x)
            seq_lengths = self.adjust_lengths(layer, seq_lengths)

            device = x.device  # Используем устройство, на котором находится x
            seq_lengths = seq_lengths.to(device)  # Переносим seq_lengths на то же устройство, что и x

            mask = torch.arange(x.size(-1), device=device) >= seq_lengths.unsqueeze(-1)  # [B, T]

            x.masked_fill_(mask.unsqueeze(1).unsqueeze(2), 0)

        return x, seq_lengths

    def adjust_lengths(self, layer, seq_lengths):
        """
        Корректирует длины последовательностей в зависимости от слоя.

        Args:
            layer (nn.Module): Текущий слой (например, Conv2d или MaxPool2d).
            seq_lengths (Tensor): Текущие длины последовательностей.

        Returns:
            Tensor: Обновлённые длины последовательностей после применения слоя.
        """
        if isinstance(layer, nn.Conv2d):
            numerator = seq_lengths + 2 * layer.padding[1] - layer.dilation[1] * (layer.kernel_size[1] - 1) - 1
            seq_lengths = (numerator.float() / float(layer.stride[1])).int() + 1
        elif isinstance(layer, nn.MaxPool2d):
            seq_lengths = seq_lengths // 2

        return seq_lengths.int()


class NormalizedRNNBlock(nn.Module):
    """
    RNN блок с нормализацией по батчам и активацией.
    """

    def __init__(self, input_dim, hidden_dim, is_bidirectional, rnn_type):
        """
        Args:
            input_dim (int): Размерность входа для RNN.
            hidden_dim (int): Размерность скрытых состояний RNN.
            is_bidirectional (bool): Используется ли двусторонний RNN.
            rnn_type (str): Тип RNN ('lstm', 'gru', или 'rnn').
        """
        super().__init__()
        rnn_types = {
            "lstm": nn.LSTM,
            "gru": nn.GRU,
            "rnn": nn.RNN,
        }

        self.batch_norm = nn.BatchNorm1d(input_dim)
        self.activation = nn.ReLU()

        self.rnn = rnn_types[rnn_type](
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            bias=True,
            batch_first=True,
            dropout=0.1,
            bidirectional=is_bidirectional,
        )

    def forward(self, x, seq_lengths):
        """
        Прямой проход через слой RNN.

        Args:
            x (Tensor): Входной тензор с размерностью [B, T, H].
            seq_lengths (Tensor): Длины последовательностей [B].

        Returns:
            Tensor: Результат после RNN слоя.
        """
        x = self.activation(self.batch_norm(x.transpose(1, 2)))  # [B, H, T]

        x = x.transpose(1, 2)  # [B, T, H]
        max_len = x.size(1)

        x = nn.utils.rnn.pack_padded_sequence(x, seq_lengths.cpu(), batch_first=True, enforce_sorted=False)
        x, _ = self.rnn(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(x, total_length=max_len, batch_first=True)  # [B, T, H]

        return x


class SpeechRecognitionModel(nn.Module):
    """
    Модель для распознавания речи, основанная на свёрточных и RNN слоях.
    """

    def __init__(self, n_tokens , spec_dim=128, num_rnn_layers=5, rnn_hidden_size=512, is_bidirectional=True,
                 rnn_type='gru'):
        """
        Args:
            n_tokens (int): Количество токенов в словаре.
            spec_dim (int): Размерность спектрограммы (количество частотных бинов).
            num_rnn_layers (int): Количество слоёв RNN.
            rnn_hidden_size (int): Размер скрытого состояния для RNN слоёв.
            is_bidirectional (bool): Является ли RNN двусторонним.
            rnn_type (str): Тип RNN ('gru', 'lstm', или 'rnn').
        """
        super().__init__()

        in_channels = 1
        out_channels = 32  # Количество выходных каналов после свёртки

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Hardtanh(0, 20, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Hardtanh(0, 20, inplace=True),
        )

        self.masked_conv = MaskedConvBlock(self.conv_block)
        conv_output_size = out_channels * (spec_dim // 4)

        self.rnn_blocks = nn.ModuleList()
        rnn_output_size = rnn_hidden_size * 2 if is_bidirectional else rnn_hidden_size

        for _ in range(num_rnn_layers):
            self.rnn_blocks.append(
                NormalizedRNNBlock(
                    input_dim=conv_output_size,
                    hidden_dim=rnn_hidden_size,
                    is_bidirectional=is_bidirectional,
                    rnn_type=rnn_type,
                )
            )

        self.fc = nn.Sequential(
            nn.LayerNorm(rnn_output_size),
            nn.Linear(rnn_output_size, n_tokens, bias=False),
        )

    def forward(self, spectrogram , spectrogram_length, **kwargs):
        """
        Прямой проход через модель.

        Args:
            spectrogram  (Tensor): Входная спектрограмма с размерностью [B, F, T].
            spectrogram_length (Tensor): Длины спектрограмм [B].

        Returns:
            dict: Логарифмические вероятности и длины выходных последовательностей.
        """
        x = spectrogram.unsqueeze(1)  # [B, 1, F, T]
        x, output_lengths = self.masked_conv(x, spectrogram_length)  # [B, 32, F/4, T/2], [B]

        B, C, F, T = x.size()
        x = x.view(B, C * F, T)  # [B, C*F/4, T/2]
        x = x.transpose(1, 2)  # [B, T/2, C*F/4]

        for rnn_block in self.rnn_blocks:
            x = rnn_block(x, output_lengths)

        x = self.fc(x)  # [B, T/2, vocab_size]
        x = x.log_softmax(dim=-1)

        return {"log_probs": x, "log_probs_length": output_lengths}

    def __str__(self):
        """
        Выводит информацию о количестве параметров модели.
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        model_info = super().__str__()
        model_info += f"\nTotal parameters: {total_params}"
        model_info += f"\nTrainable parameters: {trainable_params}"

        return model_info
