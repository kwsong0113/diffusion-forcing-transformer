# 3D Unet / Temporal Attention implementation Diffusion Forcing

#### [[Diffusion Forcing Website]](https://boyuan.space/diffusion-forcing) [[Original Implementation]](https://github.com/buoyancy99/diffusion-forcing)

This is a 3D Unet implementation of paper [Diffusion Forcing: Next-token Prediction Meets Full-Sequence Diffusion](https://boyuan.space/diffusion-forcing).

This repo is coded by [Kiwhan Song](https://www.linkedin.com/in/kiwhan-song/), an amazing MIT undergrad working with Boyuan Chen and Vincent Sitzmann based on [Boyuan](https://boyuan.space/)'s research template repo.

The content is not used in the original [Diffusion Forcing](https://boyuan.space/diffusion-forcing) paper but a reimplementation with better architecture for video generation. Original Diffusion Forcing code is RNN based to optimize for sequential decision making, while this repo uses Lucidrain's 3DUnet/Attention optimized for video.

This repo was originally part of our follow up project but we decided to release it early due to popularity of Diffusion Forcing among Generative AI community. Right now auto-regressive sampling with this repo is expected to be slow, since we haven't implemented causal attention caching. We've already verified diffusion forcing works in latent diffusion and can be extended to many more tokens without sacrificing compositionality with some special techniques, although those code will not be released immediately!

# Project Instructions

## Setup

Run `conda create python=3.10 -n df_unet` to create environment.
Run `conda activate df_unet` to activate this environment.
Run `pip install -r requirements.txt` to install all dependencies.

[Sign up](https://wandb.ai/site) a wandb account for cloud logging and checkpointing. In command line, run `wandb login` to login.

Then modify the wandb entity (account) in `configurations/config.yaml`.

## Quick start with pretrained checkpoints

Since dataset is huge, we provide a mini subset and pre-trained checkpoints for you to quickly test out our model! To do so, download mini dataset and checkpoints from [here](https://drive.google.com/file/d/16kAS1zu_ClMZFsLJZZn1qMInaNuW9hdP/view?usp=drive_link) to project root and extract with `tar -xzvf minecraft_3dunet.tar.gz`. Files shall appear as `data/minecraft` and `outputs/minecraft.ckpt`. Make sure you also git pull upstream to use latest version of code if you forked before ckpt release!

Then run the following commands and go to the wandb panel to see the results.

Minecraft:
Autoregressively generate with 1x the length it's trained on:
`python -m main +name=sample_minecraft_pretrained algorithm.weight_decay=0.002 algorithm.diffusion.network_size=64 algorithm.diffusion.attn_dim_head=64 algorithm.diffusion.attn_resolutions=[16,32,64,128] algorithm.diffusion.beta_schedule=sigmoid algorithm.diffusion.clip_noise=6.0 algorithm.diffusion.cum_snr_decay=0.96 algorithm.diffusion.stabilization_level=15 experiment.training.lr=8e-5 load=outputs/minecraft.ckpt experiment.tasks=[validation]`

To let the model roll out **longer than it's trained on**, simply append `dataset.validation_multiplier=8` to the above commands, and it will rollout `8x` longer than maximum sequence length it's trained on.

Our visualization is side by side, with prediction on the left and ground truth on the right. However, ground truth is expected to not align with prediction since the sequence is highly stochastic. Ground truth is provided to provide an idea about quality only.

The above checkpoint is trained for 100K steps with small number of frames. We've already verified diffusion forcing works in latent diffusion setting and can be extended to many more tokens without sacrificing compositionality.

## Training

Video prediction requires downloading giant datasets. First, if you downloaded the mini subset following `Quick start with pretrained checkpoints` section, delete the mini subset folders `data/minecraft` and `data/dmlab`. Them just run the following commands: we've coded in python that it will download the dataset for you it doesn't already exist. Due to the slowness of the [source](https://github.com/wilson1yan/teco), this may take a couple days. If you prefer to do it yourself via bash script, please refer to the bash scripts in original [TECO dataset](https://github.com/wilson1yan/teco) and use `dmlab.sh` and `minecraft.sh` in their Dataset section of README, any maybe split bash script into parallel scripts.

Then just run the corresponding commands:

### DMLab

`python -m main +name=your_experiment_name algorithm=df_video dataset=video_dmlab`

### Minecraft

`python -m main +name=your_experiment_name algorithm=df_video dataset=video_minecraft algorithm.weight_decay=0.002 algorithm.diffusion.network_size=64 algorithm.diffusion.attn_dim_head=64 algorithm.diffusion.attn_resolutions=[16,32,64,128] algorithm.diffusion.beta_schedule=sigmoid algorithm.diffusion.clip_noise=6.0 algorithm.diffusion.cum_snr_decay=0.96 algorithm.diffusion.stabilization_level=15 experiment.training.lr=8e-5 `

### No causal masking

Simply append `algorithm.diffusion.use_causal_mask=False` to your command.

## Sampling

Please take a look at "Load a checkpoint to eval" paragraph to understand how to use load checkpoint with `load=`. Then, run the exact training command with `experiment.tasks=[validation] load={wandb_run_id}` to load a checkpoint and experiment with sampling.

By default, we run autoregressive sampling with stablization. To sample next 2 tokens jointly, you can append the following to the above command: `algorithm.scheduling_matrix=full_sequence algorithm.chunk_size=2`.

# Infra instructions

This repo is forked from [Boyuan Chen](https://boyuan.space/)'s research template repo.

All experiments can be launched via `python -m main +name=xxxx {options}` where you can fine more details later in this article.

The code base will automatically use cuda or your Macbook M1 GPU when available.

For slurm clusters e.g. mit supercloud, you can run `python -m main cluster=mit_supercloud {options}` on login node.
It will automatically generate slurm scripts and run them for you on a compute node. Even if compute nodes are offline,
the script will still automatically sync wandb logging to cloud with <1min latency. It's also easy to add your own slurm
by following the `Add slurm clusters` section.

## Modify for your own project

First, create a new repository with this template. Make sure the new repository has the name you want to use for wandb
logging.

Add your method and baselines in `algorithms` following the `algorithms/README.md` as well as the example code in
`algorithms/diffusion_forcing/df_video.py`. For pytorch experiments, write your algorithm as a [pytorch lightning](https://github.com/Lightning-AI/lightning)
`pl.LightningModule` which has extensive
[documentation](https://lightning.ai/docs/pytorch/stable/). For a quick start, read "Define a LightningModule" in this [link](https://lightning.ai/docs/pytorch/stable/starter/introduction.html). Finally, add a yaml config file to `configurations/algorithm` imitating that of `configurations/algorithm/df_video.yaml`, for each algorithm you added.

Add your dataset in `datasets` following the `datasets/README.md` as well as the example code in
`datasets/video`. Finally, add a yaml config file to `configurations/dataset` imitating that of
`configurations/dataset/video_dmlab.yaml`, for each dataset you added.

Add your experiment in `experiments` following the `experiments/README.md` or following the example code in
`experiments/exp_video.py`. Then register your experiment in `experiments/__init__.py`.
Finally, add a yaml config file to `configurations/experiment` imitating that of
`configurations/experiment/exp_video.yaml`, for each experiment you added.

Modify `configurations/config.yaml` to set `algorithm` to the yaml file you want to use in `configurations/algorithm`;
set `experiment` to the yaml file you want to use in `configurations/experiment`; set `dataset` to the yaml file you
want to use in `configurations/dataset`, or to `null` if no dataset is needed; Notice the fields should not contain the
`.yaml` suffix.

You are all set!

`cd` into your project root. Now you can launch your new experiment with `python main.py +name=<name_your_experiment>`. You can run baselines or
different datasets by add arguments like `algorithm=xxx` or `dataset=xxx`. You can also override any `yaml` configurations by following the next section.

One special note, if your want to define a new task for your experiment, (e.g. other than `training` and `test`) you can define it as a method in your experiment class and use `experiment.tasks=[task_name]` to run it. Let's say you have a `generate_dataset` task before the task `training` and you implemented it in experiment class, you can then run `python -m main +name xxxx experiment.tasks=[generate_dataset,training]` to execute it before training.

## Pass in arguments

We use [hydra](https://hydra.cc) instead of `argparse` to configure arguments at every code level. You can both write a static config in `configuration` folder or, at runtime,
[override part of yur static config](https://hydra.cc/docs/tutorials/basic/your_first_app/simple_cli/) with command line arguments.

For example, arguments `algorithm=example_classifier experiment.lr=1e-3` will override the `lr` variable in `configurations/experiment/example_classifier.yaml`. The argument `wandb.mode` will override the `mode` under `wandb` namesspace in the file `configurations/config.yaml`.

All static config and runtime override will be logged to cloud automatically.

## Resume a checkpoint & logging

For machine learning experiments, all checkpoints and logs are logged to cloud automatically so you can resume them on another server. Simply append `resume={wandb_run_id}` to your command line arguments to resume it. The run_id can be founded in a url of a wandb run in wandb dashboard. By default, latest checkpoint in a run is stored indefinitely and earlier checkpoints in the run will be deleted after 5 days to save your storage.

On the other hand, sometimes you may want to start a new run with different run id but still load a prior ckpt. This can be done by setting the `load={wandb_run_id / ckpt path}` flag.

## Load a checkpoint to eval

The argument `experiment.tasks=[task_name1,task_name2]` (note the `[]` brackets here needed) allows to select a sequence of tasks to execute, such as `training`, `validation` and `test`. Therefore, for testing a machine learning ckpt, you may run `python -m main load={your_wandb_run_id} experiment.tasks=[test]`.

More generally, the task names are the corresponding method names of your experiment class. For `BaseLightningExperiment`, we already defined three methods `training`, `validation` and `test` for you, but you can also define your own tasks by creating methods to your experiment class under intended task names.

## Debug

We provide a useful debug flag which you can enable by `python main.py debug=True`. This will enable numerical error tracking as well as setting `cfg.debug` to `True` for your experiments, algorithms and datasets class. However, this debug flag will make ML code very slow as it automatically tracks all parameter / gradients!

## Add slurm clusters

It's very easy to add your own slurm clusters via adding a yaml file in `configurations/cluster`. You can take a look
at `configurations/cluster/mit_supercloud.yaml` for example.
