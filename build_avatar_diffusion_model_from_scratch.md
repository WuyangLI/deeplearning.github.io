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

Initially, I attempted to train a basic diffusion model without any conditions. 
<p align=center>
  <img src="figures/conditional_diffusion-diffusion.drawio.png" alt="basic diffusion model" width="400"/>
</p>
In my first trial, I normalized the image tensors to fall within the [0, 1] range. 
However, after the first epoch of training, the model didn't show much learning and generated random noise as output. 
<p align=center>
  <img src="figures/figure1_vanilla_diffusion_0_ep0.png" alt="figure 1" width="800"/>
</p>
It took around 4 epochs for the model to begin generating images that weren't just white noise.
<p align=center>
  <img src="figures/figure2_vanilla_diffusion_0_ep4.png" alt="figure 2" width="800"/>
</p>
Subsequently, when I adjusted the image tensors to the [-1, 1] range, the model started learning the features of avatars more rapidly. The model is able to generate the following samples after training for the first epoch.
<p align=center>
  <img src="figures/figure3_vanilla_diffusion_1_ep0.png" alt="figure 3" width="800"/>
</p>
As seen in figure 4, it took 6 epochs for the model to produce some images resembling faces. 
<p align=center>
  <img src="figures/figure4_vanilla_diffusion_1_ep6.png" alt="figure 4" width="800"/>
</p>

By "epoch 39," the model managed to capture attributes such as skin tone, glasses, eye color, eye shape, and various hairline shapes.

However, the model struggled to learn different hairstyles and to paint them in good colors. 
For diffusion models, it's essential that the pixel values fall within the range of [−1,1] because Gaussian noise has zero mean and unit variance. Consequently, I utilized a lambda function to transform the range from [0,1] to [−1,1]. This explains why the model performs better when the normalization range is adjusted to [-1,1].








