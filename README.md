
# VAE Face Generator

A Generative AI project that implements a Variational Autoencoder (VAE) in PyTorch to learn the distribution of human facial images from the CelebA dataset and generate new synthetic faces from randomly sampled latent vectors.

The repository covers the complete workflow from VAE training and latent-space generation to model packaging, GitHub-based model hosting, and interactive Gradio deployment.

## Overview

```text
CelebA Face Images
        |
        v
Image Preprocessing
        |
        v
   VAE Encoder
        |
        v
 Mean + Log Variance
        |
        v
Reparameterization
        |
        v
   Latent Vector
        |
        v
   VAE Decoder
        |
        v
Generated Face
        |
        v
   Gradio Interface
````

## Objectives

The project aims to:

* Implement a Variational Autoencoder using PyTorch.
* Learn a compact latent representation of face images.
* Reconstruct images from their latent representations.
* Generate new synthetic human faces.
* Save and reuse trained model weights.
* Demonstrate model hosting through GitHub Releases.
* Build an interactive Gradio application.
* Explore deployment through Hugging Face Spaces.

## Dataset

The face-generation notebook uses a 2,000-image training subset from a CelebA face dataset available through Hugging Face Datasets.

The primary dataset source used in the notebook is:

```text
nielsr/CelebA-faces
```

Images are resized to:

```text
64 × 64
```

and converted to tensors before training.

The dataset itself is not included in this repository.

## Variational Autoencoder

A VAE consists of an encoder, latent probabilistic representation, and decoder.

### Encoder

The encoder maps an RGB face image into:

* Mean (`μ`)
* Log variance (`log σ²`)

### Reparameterization

A latent vector is sampled using:

```text
z = μ + σ × ε
```

where:

```text
ε ~ N(0, I)
```

### Decoder

The decoder maps the sampled latent vector back to an image.

It can also be used independently with randomly sampled latent vectors to generate new faces.

## Model Architecture

The supplied model uses a latent dimension of:

```text
128
```

The encoder contains four convolutional stages:

```text
3 → 32
32 → 64
64 → 128
128 → 256
```

The decoder uses transposed convolution layers:

```text
256 → 128
128 → 64
64 → 32
32 → 3
```

A sigmoid activation is used at the final layer.

## Loss Function

The training objective combines reconstruction loss and KL divergence:

```text
VAE Loss = Reconstruction Loss + β × KL Divergence
```

The face-generation notebook uses mean squared error for reconstruction and a KL divergence term to regularize the latent distribution.

## Training Configuration

| Parameter        |               Value |
| ---------------- | ------------------: |
| Dataset size     |        2,000 images |
| Image size       |             64 × 64 |
| Latent dimension |                 128 |
| Batch size       |                  32 |
| Optimizer        |                Adam |
| Learning rate    |              `1e-3` |
| Weight decay     |              `1e-5` |
| Epochs           |                 100 |
| Device           | CUDA when available |

## Training Results

The supplied training output shows the average loss decreasing during training.

Examples include:

```text
Epoch 1  → 673.8144
Epoch 2  → 410.3872
Epoch 3  → 367.1760
Epoch 10 → 266.9181
Epoch 20 → 230.2172
```

## Face Generation

After training, the decoder can generate a new image without an input face.

A random latent vector is sampled and passed through the decoder:

```text
Random Latent Vector
        |
        v
      Decoder
        |
        v
 Synthetic Face
```

A sample generated face is included in:

```text
assets/sample_generated_face.webp
```

## GitHub Model Hosting

The trained model is packaged as:

```text
vae_decoder.zip
```

containing:

```text
vae_celeba.pth
```

The deployment workflow downloads the archive from a GitHub Release, extracts the weights, reconstructs the VAE architecture, and loads the trained state dictionary.

## Gradio Application

The repository contains:

```text
app/app.py
```

The interface provides a simple:

```text
Generate Face
```

button.

Each click samples a new random latent vector and generates a synthetic face using the trained decoder.

## Hugging Face Deployment

The repository also contains:

```text
notebooks/03_huggingface_deployment.ipynb
```

which demonstrates the workflow for preparing the application for deployment to a Hugging Face Space.

Authentication credentials are intentionally not included in the repository.

## Repository Structure

```text
vae-face-generator/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── notebooks/
│   ├── 01_train_vae_faces.ipynb
│   ├── 02_github_deployment.ipynb
│   └── 03_huggingface_deployment.ipynb
│
├── app/
│   ├── app.py
│   └── vae_model.py
│
├── model/
│   ├── vae_celeba.pth
│   ├── vae_decoder.zip
│   └── README.md
│
├── assets/
│   └── sample_generated_face.webp
│
└── docs/
    ├── face_generation_report.pdf
    ├── github_deployment_report.pdf
    └── huggingface_deployment_or_lab_report.pdf
```

## Installation

```bash
git clone https://github.com/MansiiSapariya/vae-face-generator.git
cd vae-face-generator
pip install -r requirements.txt
```

A GPU-enabled environment is recommended for training.

## Running the Application

```bash
python app/app.py
```

The application first looks for:

```text
model/vae_celeba.pth
```

If the local model is unavailable, it can fall back to the configured GitHub Release archive.

## Training

Open:

```text
notebooks/01_train_vae_faces.ipynb
```

The notebook covers:

1. Dataset loading
2. Image preprocessing
3. DataLoader creation
4. VAE architecture
5. Model training
6. Loss tracking
7. Image reconstruction
8. Latent-space sampling
9. New face generation
10. Model saving

## Technologies Used

* Python
* PyTorch
* TorchVision
* NumPy
* Matplotlib
* Hugging Face Datasets
* Gradio
* Requests
* GitHub Releases
* Hugging Face Spaces
* Google Colab

## Limitations

* The training experiment uses 2,000 face images.
* Generated resolution is limited to 64 × 64.
* VAE outputs can contain blur and visual artifacts.
* Training requires significant computational resources.
* The model does not provide explicit control over facial attributes.
* Generated faces are synthetic and should not be interpreted as real identities.

## Future Scope

* Train on a larger portion of CelebA.
* Increase image resolution.
* Experiment with deeper architectures.
* Explore β-VAE.
* Add controllable facial attributes.
* Add quantitative image-generation evaluation.
* Add latent-space interpolation.
* Add generation controls to the Gradio interface.
* Deploy as a permanent Hugging Face Space.

## Key Learnings

This project provides practical experience with:

* Variational Autoencoders
* Generative modeling
* Probabilistic latent spaces
* Reparameterization
* KL divergence
* Reconstruction loss
* PyTorch
* Convolutional neural networks
* Transposed convolutions
* Model serialization
* GitHub model hosting
* Gradio deployment
* Hugging Face deployment

## Conclusion

This project demonstrates an end-to-end implementation of a Variational Autoencoder for synthetic face generation.

The model learns a compact probabilistic representation of facial images and uses this representation to reconstruct existing faces and generate new synthetic faces.

The project also demonstrates practical model versioning and deployment by packaging trained weights for GitHub Releases and preparing the application for Hugging Face Spaces.

## Author

**Mansi Sapariya**

MSc Data Science
Christ (Deemed to be University), Bangalore

```

This one is particularly good for your GitHub portfolio because it shows **both the ML work and the deployment side**, rather than looking like just another coursework notebook.
```
