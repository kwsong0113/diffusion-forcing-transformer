from datasets.video import (
    MinecraftVideoDataset,
    DmlabVideoDataset,
)
from algorithms.diffusion_forcing import DiffusionForcingVideo
from .exp_base import BaseLightningExperiment


class VideoPredictionExperiment(BaseLightningExperiment):
    """
    A video prediction experiment
    """

    compatible_algorithms = dict(
        df_video=DiffusionForcingVideo,
    )

    compatible_datasets = dict(
        # video datasets
        video_minecraft=MinecraftVideoDataset,
        video_dmlab=DmlabVideoDataset,
    )
