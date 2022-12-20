import logging
from typing import Any, List, Dict
import torch
from collections import defaultdict

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[Dict]) -> Dict[str, Any]:
    """
    Collate and pad fields in dataset items
    """
    result_batch: Dict[str, Any] = defaultdict(list)
    for item in dataset_items:
        result_batch["text"].append(item["text"])
        result_batch["spectrogram_length"].append(item["spectrogram"].shape[-1])
        result_batch["audio_length"].append(item["audio"].shape[-1])
        result_batch["audio_path"].append(item["audio_path"])

    batch_spec = torch.zeros(len(dataset_items), dataset_items[0]["spectrogram"].shape[1],
                             max(result_batch["spectrogram_length"]))
    batch_audio = torch.zeros(len(dataset_items), max(result_batch["audio_length"]))

    for i, item in enumerate(dataset_items):
        batch_spec[i, :, :result_batch["spectrogram_length"][i]] = item["spectrogram"]
        batch_audio[i, :result_batch["audio_length"][i]] = item["audio"]

    result_batch["spectrogram_length"] = torch.tensor(result_batch["spectrogram_length"], dtype=torch.int)
    result_batch["spectrogram"] = batch_spec
    result_batch["audio"] = batch_audio

    return result_batch
