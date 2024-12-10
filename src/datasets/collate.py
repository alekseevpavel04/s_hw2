import torch


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """
    # Extract fields
    audio = [item["audio"] for item in dataset_items]
    spectrogram = [item["spectrogram"] for item in dataset_items]
    text = [item["text"] for item in dataset_items]
    text_encoded = [item["text_encoded"] for item in dataset_items]
    audio_paths = [item["audio_path"] for item in dataset_items]

    # Pad audio
    audio_lengths = torch.tensor([a.size(-1) for a in audio], dtype=torch.int64)
    audio_padded = torch.nn.utils.rnn.pad_sequence(audio, batch_first=True)

    # Pad spectrograms
    spectrogram_lengths = torch.tensor([s.size(-1) for s in spectrogram], dtype=torch.int64)
    spectrogram_padded = torch.nn.utils.rnn.pad_sequence(spectrogram, batch_first=True)

    # Pad text encoded
    text_encoded_lengths = torch.tensor([t.size(0) for t in text_encoded], dtype=torch.int64)
    text_encoded_padded = torch.nn.utils.rnn.pad_sequence(text_encoded, batch_first=True)

    # Combine into a batch
    result_batch = {
        "audio": audio_padded,
        "audio_lengths": audio_lengths,
        "spectrogram": spectrogram_padded,
        "spectrogram_lengths": spectrogram_lengths,
        "text": text,
        "text_encoded": text_encoded_padded,
        "text_encoded_lengths": text_encoded_lengths,
        "audio_paths": audio_paths,
    }

    return result_batch
