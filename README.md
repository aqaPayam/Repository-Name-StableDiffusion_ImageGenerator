
# Stable Diffusion - Text-to-Image Generator

## Overview

This repository contains an implementation of **Stable Diffusion**, a powerful text-to-image latent diffusion model. Stable Diffusion is designed to generate images based on textual prompts by iteratively denoising random latent representations. The model reduces computational complexity by performing the diffusion process in a lower-dimensional latent space rather than the full pixel space, making it more efficient for high-resolution image generation.

### What is Stable Diffusion?

Stable Diffusion is a model developed by researchers and engineers from [CompVis](https://github.com/CompVis), [Stability AI](https://stability.ai/), and [LAION](https://laion.ai/). It uses a frozen CLIP ViT-L/14 text encoder to condition the image generation on textual inputs. The model was trained on a subset of the [LAION-5B](https://laion.ai/blog/laion-5b/) dataset and is capable of generating 512x512 resolution images.

Diffusion models are generative models that are trained to reverse a noise process. They start with random noise and progressively denoise it step-by-step to produce a sample of interest, such as an image.

### Key Components

1. **Text-to-Image Translation**: The model takes a text prompt as input and generates a corresponding image. The text is processed using CLIP's text encoder, which transforms the text into a latent embedding.
  
2. **Latent Diffusion Process**: The image is generated in a latent space where the diffusion process is applied. This reduces the memory and computation requirements compared to applying the process in pixel space.

3. **Denoising U-Net**: The U-Net architecture is used to iteratively remove noise from the latent image representations, conditioning the process on the text embeddings.

4. **Schedulers**: The denoising process is guided by a scheduler algorithm, which computes the predicted denoised image representation at each step. Various schedulers can be used, including:
    - **PNDM scheduler**
    - **K-LMS scheduler**
    - **Heun Discrete scheduler**
    - **DPM Solver Multistep scheduler** (Recommended for faster generation with fewer steps)

### Workflow

The generation process can be broken down into the following steps:
1. **Input Text**: Provide a textual prompt that describes the desired image.
2. **Latent Seed**: A random latent seed is used to initialize the latent space representation.
3. **Denoising Process**: The U-Net model iteratively denoises the latent representation step by step, improving the image with each iteration.
4. **Decoding**: Once the denoising process is complete, the latent image is decoded to produce the final image.

### Installation and Dependencies

To run this project, you need to have the following installed:

- Python 3.x
- PyTorch
- Hugging Face's `diffusers` library
- NumPy
- Matplotlib
- Transformers
- TQDM (optional for progress tracking)

You can install the required packages using:

```bash
pip install torch diffusers numpy matplotlib transformers tqdm
```

### How to Run the Notebook

1. Clone the repository:
   ```bash
   git clone https://github.com/AqaPayam/StableDiffusion_ImageGenerator.git
   ```

2. Navigate to the project directory:
   ```bash
   cd StableDiffusion_ImageGenerator
   ```

3. Run the Jupyter notebook:
   ```bash
   jupyter notebook StableDiffusion_Implementation.ipynb
   ```

4. Follow the instructions in the notebook to input text prompts and generate images.

### Example Usage

- Input Prompt: `"A futuristic city skyline at sunset."`
- Generated Image: The model will produce a high-quality image corresponding to the prompt.

### Model Details

- **Text Encoder**: CLIP ViT-L/14
- **U-Net**: 860M parameters
- **Text Embeddings**: Size 77x768
- **Latent Representation**: Size 64x64
- **Schedulers**: Different schedulers can be used for denoising, with the DPM Solver Multistep scheduler recommended for faster generation.

### Results

This model is capable of generating high-quality, high-resolution images based on textual descriptions. It performs iterative denoising in latent space, making it more efficient than traditional pixel-based diffusion models.

### Contribution

Feel free to contribute by improving the model, adding more features, or experimenting with new text prompts. Fork the repository and submit a pull request with your improvements!

### License

This project is licensed under the MIT License.
