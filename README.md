# Generative Deep Learning — Assignments

Assignment submissions for the Generative Deep Learning course, Master's degree in Computer Science.


## Assignment 1 — Gaussian Mixture Model & EM Algorithm
Implement a Gaussian Mixture Model (GMM) from scratch using only NumPy, trained with the Expectation-Maximization (EM) algorithm (diagonal covariance assumed).

**E-step:** compute soft assignments (responsibilities) per sample.  
**M-step:** update mixture weights $\pi$, means $\mu$, and variances $\sigma$.  
**Model selection:** use the Bayesian Information Criterion (BIC) to pick the optimal number of components $k \in \{1, \ldots, 6\}$.

```bash
cd first-assignment
jupyter notebook "GDL Midterm 1 - ID 10.ipynb"
```
## Assignment 2 — CNN Forward Pass

Implement the full forward pass of a CNN from scratch using only NumPy — no ML framework allowed. Pre-trained weights are provided and the implementation is validated on MNIST.

Primitives to implement:

| Function | Description |
|---|---|
| `conv2d` | 2D convolution, arbitrary stride/padding, NCHW format, no `np.convolve` |
| `max_pool` | Max-pooling |
| `avg_pool` | Average-pooling |
| `relu` | ReLU activation |
| `cnn_forward` | Full forward pass wiring all layers |

```bash
cd second-assignment
jupyter notebook "GDL Midterm 2 - ID 17.ipynb"
```

## Assignment 3 — Adversarial Autoencoder

Implement an Adversarial Autoencoder (Makhzani et al., 2015) in PyTorch. The latent space is regularized adversarially instead of via KL divergence: a discriminator learns to separate encoder outputs from prior samples $z \sim \mathcal{N}(0, I)$, while the encoder is trained to fool it.

**Components:** `Encoder`, `Decoder`, `Discriminator`, `AdversarialAutoencoder` wrapper.

**Training loop (two phases per mini-batch):**
- *Phase 1 — Reconstruction:* minimize BCE between input and reconstruction (encoder + decoder).
- *Phase 2 — Adversarial regularization:*
  - Discriminator step: train $D_\psi$ to distinguish prior (real) from encoder output (fake).
  - Generator step: train the encoder alone to fool $D_\psi$.

Three separate optimizers are required: `opt_ae`, `opt_disc`, `opt_gen`.

```bash
cd third-assignment
jupyter notebook "GDL Midterm 3 - ID 17.ipynb"
```


## Assignment 4 — Group Poster

Group poster (`group6_poster.pdf`) based on the paper **"On Oversquashing in Graph Neural Networks Through the Lens of Dynamical Systems"** (Gravina et al., 2025).

The paper introduces **SWAN**, a Graph Neural Network designed to tackle the *oversquashing* problem in Message-Passing Neural Networks (MPNNs) — i.e., the exponential decay of information between distant nodes. SWAN leverages properties from dynamical systems theory (global and local non-dissipativity) to maintain a constant information flow rate regardless of network depth, improving long-range node interactions.
