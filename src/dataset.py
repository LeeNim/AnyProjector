"""
dataset.py - Dataset và DataLoader cho AnyProjector.

Phase 2: AlignmentDataset - Dữ liệu căn chỉnh vector + VAD.
Phase 3: AgenticDataset - Dữ liệu Intent/Tool Calling (sẽ thêm sau).

Định dạng metadata_alignment.jsonl:
    {"audio_file": "wavs/audio_001.wav", "text": "Xin chào...", "is_end_of_speech": 1}
    {"audio_file": "wavs/audio_002_cut.wav", "text": "Hôm nay thời tiết ở", "is_end_of_speech": 0}
"""

import json
import logging
from pathlib import Path

import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


class AlignmentDataset(Dataset):
    """Dataset cho Phase 2: Alignment Training.

    Đọc metadata JSONL + load audio file, trả về:
    - audio waveform (đã resample về 16kHz)
    - text transcript
    - is_end_of_speech label (0 hoặc 1)

    Args:
        data_dir: Đường dẫn thư mục phase2_alignment/.
        sample_rate: Tần số lấy mẫu mục tiêu (default 16000 Hz).
        max_audio_seconds: Giới hạn độ dài audio (seconds). Cắt nếu quá dài.
    """

    METADATA_FILE = "metadata_alignment.jsonl"

    def __init__(
        self,
        data_dir: str | Path,
        sample_rate: int = 16000,
        max_audio_seconds: float = 30.0,
    ):
        self.data_dir = Path(data_dir)
        self.sample_rate = sample_rate
        self.max_audio_samples = int(max_audio_seconds * sample_rate)

        # Đọc metadata
        metadata_path = self.data_dir / self.METADATA_FILE
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata file not found: {metadata_path}\n"
                f"Expected format: JSONL with fields: audio_file, text, is_end_of_speech"
            )

        self.samples = []
        with open(metadata_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    # Validate required fields
                    for key in ("audio_file", "text", "is_end_of_speech"):
                        if key not in entry:
                            raise KeyError(f"Missing field: '{key}'")
                    self.samples.append(entry)
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"Skipping line {line_num} in metadata: {e}")

        logger.info(
            f"AlignmentDataset loaded: {len(self.samples)} samples from {metadata_path}"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        """Trả về 1 sample.

        Returns:
            dict với keys:
            - "waveform": torch.Tensor (1, num_samples) — audio mono 16kHz
            - "text": str — transcript
            - "is_end_of_speech": float — 0.0 hoặc 1.0
        """
        entry = self.samples[idx]
        audio_path = self.data_dir / entry["audio_file"]

        # Load audio
        waveform, orig_sr = torchaudio.load(str(audio_path))

        # Chuyển về mono nếu stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample nếu cần
        if orig_sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_sr, self.sample_rate)
            waveform = resampler(waveform)

        # Cắt nếu quá dài
        if waveform.shape[1] > self.max_audio_samples:
            waveform = waveform[:, : self.max_audio_samples]

        return {
            "waveform": waveform.squeeze(0),  # (num_samples,)
            "text": entry["text"],
            "is_end_of_speech": float(entry["is_end_of_speech"]),
        }


def collate_alignment(batch: list[dict]) -> dict:
    """Custom collate function — pad audio waveforms to same length.

    Args:
        batch: List of dicts from AlignmentDataset.__getitem__.

    Returns:
        dict với:
        - "waveforms": (batch_size, max_num_samples) — padded
        - "waveform_lengths": (batch_size,) — original lengths (trước padding)
        - "texts": list[str]
        - "is_end_of_speech": (batch_size,) — labels
    """
    waveforms = [sample["waveform"] for sample in batch]
    texts = [sample["text"] for sample in batch]
    vad_labels = torch.tensor(
        [sample["is_end_of_speech"] for sample in batch], dtype=torch.float32
    )

    # Lưu độ dài gốc trước khi pad
    lengths = torch.tensor([w.shape[0] for w in waveforms], dtype=torch.long)

    # Pad waveforms to max length in batch
    waveforms_padded = torch.nn.utils.rnn.pad_sequence(
        waveforms, batch_first=True, padding_value=0.0
    )

    return {
        "waveforms": waveforms_padded,        # (batch, max_samples)
        "waveform_lengths": lengths,            # (batch,)
        "texts": texts,                         # list[str]
        "is_end_of_speech": vad_labels,         # (batch,)
    }


def create_alignment_dataloader(
    data_dir: str | Path,
    batch_size: int = 4,
    num_workers: int = 2,
    shuffle: bool = True,
    sample_rate: int = 16000,
    max_audio_seconds: float = 30.0,
) -> DataLoader:
    """Tạo DataLoader cho Phase 2 Alignment Training.

    Args:
        data_dir: Đường dẫn thư mục phase2_alignment/.
        batch_size: Số sample mỗi batch.
        num_workers: Số worker cho data loading.
        shuffle: Xáo trộn dữ liệu.
        sample_rate: Tần số lấy mẫu.
        max_audio_seconds: Giới hạn độ dài audio.

    Returns:
        DataLoader sẵn sàng cho training loop.
    """
    dataset = AlignmentDataset(
        data_dir=data_dir,
        sample_rate=sample_rate,
        max_audio_seconds=max_audio_seconds,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_alignment,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    logger.info(
        f"DataLoader created: {len(dataset)} samples, "
        f"batch_size={batch_size}, {len(dataloader)} batches/epoch"
    )

    return dataloader
