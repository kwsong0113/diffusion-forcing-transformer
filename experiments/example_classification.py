import torch
import torch.nn as nn
from typing import Optional

import lightning.pytorch as pl
from .exp_base import BaseLightningExperiment

from algorithms.examples.classifier import Classifier
from datasets import CIFAR10Dataset


class ClassificationExperiment(BaseLightningExperiment):
    """
    A classification experiment.

    This experiment class seems empty because all tasks, [train, validation, test]
    are already implemented in BaseLightningExperiment. If you have a new task,
    you can add the method here and put it in the yaml file in `configurations/experiment`.
    For example, `experiments/example_helloworld.py` has a task called `main`.

    A common practice that might be useful:
    in `compatible_algorithms` or `compatible_datasets` dict below, you specify a key and a value.
    The key correspond to the `algorithm=xxx` or `dataset=xxx` in your command line argument, as well
    as the yaml file name. The value is a class that such yaml config is passed into. Therefore
    you can create multiple keys with same value to pass different set of configs to the same class,
    creating many different algorithms and datasets with the same class but different configurations.
    """

    # each key has to be a yaml file under '[project_root]/configurations/algorithm' without .yaml suffix
    compatible_algorithms = dict(
        example_classifier=Classifier,
    )

    # each key has to be a yaml file under '[project_root]/configurations/dataset' without .yaml suffix
    compatible_datasets = dict(
        example_cifar10=CIFAR10Dataset,
    )
