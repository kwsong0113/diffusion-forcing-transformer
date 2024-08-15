# 3D Unet / Temporal Attention implementation Diffusion Forcing

#### [[Diffusion Forcing Website]](https://boyuan.space/diffusion-forcing) [[Original Implementation]](https://github.com/buoyancy99/diffusion-forcing)

This is a 3D-Unet implementation of paper [Diffusion Forcing: Next-token Prediction Meets Full-Sequence Diffusion](https://boyuan.space/diffusion-forcing).

This repo is coded by [Kiwhan Song](https://www.linkedin.com/in/kiwhan-song/), an amazing MIT undergrad working with Boyuan Chen and Vincent Sitzmann based on [Boyuan](https://boyuan.space/)'s research template repo.

The content is not used in the original [Diffusion Forcing](https://boyuan.space/diffusion-forcing) paper but a reimplementation with better architecture for video generation. Original Diffusion Forcing code is RNN based to optimize for sequential decision making, while this repo uses Lucidrain's 3DUnet/Attention optimized for video.

This repo was originally part of our follow up project but we decided to release it early due to popularity of Diffusion Forcing among Generative AI community. Right now auto-regressive sampling with this repo is expected to be slow, since we haven't implemented causal attention caching. We've already verified diffusion forcing works in latent diffusion and can be extended to many more tokens without sacrificing compositionality with some special techniques, although those code will not be released immediately!

# Project Instructions
** Update Aug 2024 ** This repo has been merged into the main [[Diffusion Forcing Implementation]](https://github.com/buoyancy99/diffusion-forcing) with version number v1.5, please directly use that instead and follow the instruction there.
