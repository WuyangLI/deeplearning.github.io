# Build Avatar Diffusion Model from Scratch
<p align=center>
  <img src="figures/cartoon_set_diffusion_random_sample_grid.gif" alt="cartoon avatar diffusion random samples" width="500"/>
</p>

Diffusion models have garnered significant attention from both the general public and the machine learning community since their introduction. Huggingface provides the diffusers library, enabling developers to swiftly construct diffusion models. Rather than fine-tuning existing models and harnessing the diffuser library, we opted to create a compact diffusion model from scratch. This involved crafting the Unet along with its fundamental components, the diffusion process, and the conditioning module. Due to limited computational resources, we trained the model using a small and straightforward dataset called ["Cartoon Set"](https://google.github.io/cartoonset/download.html). 

The aim of this blog is to delve into the fundamentals of diffusion models via a practical experiment and to share insights gained from training and fine-tuning these models.

[The notebook of this hands-on lab](https://github.com/WuyangLI/ML_sessions/blob/main/diffusion_models/cartoonset_diffusion/cartoonset_diffusion_conditional_v5.ipynb) does conditional image generation on Cartoon Set by a diffusion model.
The code is modified from [cloneofsimo/minDiffusion](https://github.com/cloneofsimo/minDiffusion), [TeaPearce/Conditional_Diffusion_MNIST](https://github.com/TeaPearce/Conditional_Diffusion_MNIST), and [LucasDedieu/Denoising-Diffusion_Model_Flower102](https://github.com/LucasDedieu/Denoising-Diffusion-Model-Flower102/blob/main/diffusion_model.ipynb).

Before getting started with the practical lab, if you're looking to deepen your theoretical understanding of diffusion models, please take a look at the referenced papers.
1. [DDPM](https://arxiv.org/abs/2006.11239)
2. [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598)
3. [Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding](https://arxiv.org/abs/2205.11487)

# Dataset and Preprocessing
Cartoon Set is a collection of 2D cartoon avatar images, as illustrated below.
ADD IMAGES
This data is licensed by Google LLC under a [Creative Commons Attribution 4.0 International License](http://creativecommons.org/licenses/by/4.0/).
We apply the following preprocessing to images:
1. Central crop 360 pixels
2. Resize to 64 by 64 images
3. Convert int8 to float, in the range of [-1, 1]

The first important takeaway is that adjusting input tensors to the appropriate range significantly speeds up the model's convergence.

As [Djib2011](https://datascience.stackexchange.com/users/34269/djib2011) explained in his answer to the question ["should-input-images-be-normalized-to-1-to-1-or-0-to-1"](https://datascience.stackexchange.com/questions/54296/should-input-images-be-normalized-to-1-to-1-or-0-to-1)

> Theoretically it's better to scale your input to [−1,1] than [0,1] and I'd argue that it's even better to standardize your input (i.e. μ=0, σ=1).
>
>Deep neural networks, especially in their early days, had much trouble with backpropagation, as they suffered from vanishing/exploding gradients. A popular way to combat this is by initializing the weights of the network in smarter ways. Two initialization techniques have proven to be the most popular: Glorot initialization (sometimes referred to as Xavier initialization) and He initialization, both of which are variations of the same idea.
> 
>In order to derive their intialization they make a few assumptions, one of which is that the input features have a zero mean. You can read this post for simplified explanation, but I'd suggest reading the two papers mentioned above.


