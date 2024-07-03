import torch
from typing import Sequence
import numpy as np
import io
import tarfile
from pytorchvideo.data.encoded_video import EncodedVideo
from omegaconf import DictConfig
from tqdm import tqdm

from .base_video_dataset import BaseVideoDataset


class MinecraftVideoDataset(BaseVideoDataset):
    """
    Minecraft dataset
    """

    def __init__(self, cfg: DictConfig, split: str = "training"):
        if split == "test":
            split = "validation"
        super().__init__(cfg, split)

    def download_dataset(self) -> Sequence[int]:
        from internetarchive import download

        part_suffixes = [
            "aa",
            "ab",
            "ac",
            "ad",
            "ae",
            "af",
            "ag",
            "ah",
            "ai",
            "aj",
            "ak",
        ]
        for part_suffix in part_suffixes:
            identifier = f"minecraft_marsh_dataset_{part_suffix}"
            file_name = f"minecraft.tar.part{part_suffix}"
            download(identifier, file_name, destdir=self.save_dir, verbose=True)

        combined_bytes = io.BytesIO()
        for part_suffix in part_suffixes:
            identifier = f"minecraft_marsh_dataset_{part_suffix}"
            file_name = f"minecraft.tar.part{part_suffix}"
            part_file = self.save_dir / identifier / file_name
            with open(part_file, "rb") as part:
                combined_bytes.write(part.read())
        combined_bytes.seek(0)
        with tarfile.open(fileobj=combined_bytes, mode="r") as combined_archive:
            combined_archive.extractall(self.save_dir)
        (self.save_dir / "minecraft/test").rename(self.save_dir / "validation")
        (self.save_dir / "minecraft/train").rename(self.save_dir / "training")
        (self.save_dir / "minecraft").rmdir()
        for part_suffix in part_suffixes:
            identifier = f"minecraft_marsh_dataset_{part_suffix}"
            file_name = f"minecraft.tar.part{part_suffix}"
            part_file = self.save_dir / identifier / file_name
            part_file.rmdir()

    def get_data_paths(self, split):
        data_dir = self.save_dir / split
        paths = sorted(list(data_dir.glob("**/*.npz")), key=lambda x: x.name)
        return paths

    def get_data_lengths(self, split):
        lengths = [300] * len(self.get_data_paths(split))
        return lengths

    def __getitem__(self, idx):
        idx = self.idx_remap[idx]
        file_idx, frame_idx = self.split_idx(idx)
        action_path = self.data_paths[file_idx]
        video_path = action_path.with_suffix(".mp4")
        video = EncodedVideo.from_path(video_path, decode_audio=False)
        video = video.get_clip(start_sec=0.0, end_sec=video.duration)["video"]
        video = video.permute(1, 2, 3, 0).numpy()
        actions = np.load(action_path)["actions"][1:]

        video = video[frame_idx : frame_idx + self.n_frames]  # (t, h, w, 3)
        actions = actions[frame_idx : frame_idx + self.n_frames]  # (t, )
        actions = np.eye(4)[actions]  # (t, 3)

        pad_len = self.n_frames - len(video)

        nonterminal = np.ones(self.n_frames)
        if len(video) < self.n_frames:
            video = np.pad(video, ((0, pad_len), (0, 0), (0, 0), (0, 0)))
            actions = np.pad(actions, ((0, pad_len),))
            nonterminal[-pad_len:] = 0

        video = torch.from_numpy(video / 255.0).float().permute(0, 3, 1, 2).contiguous()
        video = self.transform(video)

        return (
            video[:: self.frame_skip],
            actions[:: self.frame_skip],
            nonterminal[:: self.frame_skip],
        )


if __name__ == "__main__":
    import torch
    from unittest.mock import MagicMock
    import tqdm

    cfg = MagicMock()
    cfg.resolution = 64
    cfg.external_cond_dim = 0
    cfg.n_frames = 64
    cfg.save_dir = "data/minecraft"
    cfg.validation_multiplier = 1

    dataset = MinecraftVideoDataset(cfg, "training")
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=4, shuffle=True, num_workers=16
    )

    for batch in tqdm.tqdm(dataloader):
        pass
