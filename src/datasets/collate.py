import torch
import torch.nn.functional as F


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
    result_batch = {}
    spectrogram_length = []
    spectrogram_raw_length = []
    text_encoded_length = []
    text = []
    audio = []
    audio_raw = []
    audio_path = []

    for item in dataset_items:
        spectrogram_length.append(item["spectrogram"].size(-1))
        text_encoded_length.append(item["text_encoded"].size(-1))
        if "text" in item:
            text.append(item["text"])
        audio.append(item["audio"])
        audio_raw.append(item["audio_raw"])
        audio_path.append(item["audio_path"])

    spectrogram_length = torch.tensor(spectrogram_length)
    text_encoded_length = torch.tensor(text_encoded_length)

    padded_spectrogram = []
    padded_spectrogram_raw = []
    padded_text_encoded = []
    max_spec = torch.max(spectrogram_length)
    max_tokens = torch.max(text_encoded_length)

    for item in dataset_items:
        spectrogram = item["spectrogram"]
        spectrogram_raw = item["spectrogram_raw"]
        text_encoded = item["text_encoded"]
        padded_spectrogram.append(F.pad(spectrogram, (0, max_spec - spectrogram.size(-1)), value=0))
        padded_spectrogram_raw.append(F.pad(spectrogram_raw, (0, max_spec - spectrogram_raw.size(-1)), value=0))
        padded_text_encoded.append(F.pad(text_encoded, (0, max_tokens - text_encoded.size(-1)), value=0))

    result_batch["spectrogram_length"] = spectrogram_length
    result_batch["text_encoded_length"] = text_encoded_length
    result_batch["spectrogram"] = torch.cat(padded_spectrogram, dim=0)
    result_batch["spectrogram_raw"] = torch.cat(padded_spectrogram_raw, dim=0)
    result_batch["text_encoded"] = torch.cat(padded_text_encoded, dim=0)
    result_batch["text"] = text
    result_batch["audio"] = audio
    result_batch["audio_raw"] = audio_raw
    result_batch["audio_path"] = audio_path

    return result_batch