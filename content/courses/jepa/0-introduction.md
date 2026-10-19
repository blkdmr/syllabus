---
title: "Introduction"
weight: 1
categories: ["Theory"]
tags: ["Deep Learning"]
draft: false
---


# Joint-Embedding Predictive Architectures

## Introduction

**Joint-Embedding Predictive Architectures (JEPA)** are a family of self-supervised learning architectures designed to
learn meaningful representations of data by predicting missing information in a **latent representation space**,
rather than reconstructing the original input.

The central idea is that a model should learn to represent the underlying structure of the data, instead of memorizing every observable detail.

Consider an image containing a person holding a ball. If part of the image is hidden, a model may be able to predict that the missing region contains the person's hand or part of the ball. However, it may not be able to determine the exact color, texture, or pixel values of that region.

**Predict what is structurally and semantically predictable, rather than reconstructing every detail of the input.**

This approach is particularly relevant to representation learning because useful representations should capture features such as object identity, spatial relationships, and high-level structure without necessarily preserving irrelevant low-level information.

## 2. Self-Supervised Learning and Representation Learning

In **self-supervised learning**, a model learns from unlabeled data by constructing prediction tasks directly from the input.

The objective is not necessarily to solve a particular downstream task, such as classification. Instead, the model learns representations that can later be reused for classification, detection, segmentation, or other tasks.

### 2.1 Invariance-Based Learning

Invariance-based methods learn representations that remain similar across different transformations of the same input.

For example, two differently cropped versions of an image should produce similar embeddings:

$$
f_\theta(x_1) \approx f_\theta(x_2)
$$

where $x_1$ and $x_2$ are different views of the same image.

This strategy encourages the model to learn features that are stable under the selected transformations.

However, the transformations introduce **inductive biases**. If the model is trained to ignore certain spatial changes, it may also discard information needed for tasks involving object positions or geometric relationships.

### 2.2 Generative Self-Supervised Learning

Generative methods learn by reconstructing missing or corrupted parts of an input.

For instance, a Masked Autoencoder (MAE) receives an image with some patches removed and attempts to reconstruct their pixel values.

A typical reconstruction objective is

$$
\mathcal{L}_{\text{reconstruction}} = \|y-\hat{y}\|_2^2
$$

where $y$ is the original target and $\hat y$ is its reconstruction.

The limitation is that the model must predict low-level details even when they are difficult to infer or not useful for semantic understanding.

### 2.3 The JEPA Alternative

JEPA combines elements of both approaches.

Like joint-embedding methods, it learns representations in a latent space. Like predictive or generative methods, it learns by predicting information from context.

However, it predicts **representations of missing information**, not the original observations.

This allows the training objective to emphasize meaningful structure instead of exact reconstruction.

## 3. General JEPA Architecture

A JEPA consists conceptually of three components:

1. **Context encoder:** extracts a representation from the available input.
2. **Target encoder:** computes the representation that the model should predict.
3. **Predictor:** predicts the target representation using the context representation and, when needed, additional conditioning information.

### 3.1 Context Encoder

The context encoder receives an observed input $x$ and maps it into a latent representation:

$$
s_x = f_\theta(x)
$$

The representation $s_x$ should contain information that is useful for predicting other compatible observations.

### 3.2 Target Encoder

The target encoder processes the target input $y$:

$$
s_y = f_{\bar{\theta}}(y)
$$

Its output serves as the learning target.

Importantly, the target is not the raw input $y$, but its learned representation $s_y$.

### 3.3 Predictor

The predictor receives the context representation and optionally a conditioning variable $z$:

$$
\hat{s}_y = g_\phi(s_x,z)
$$

The variable $z$ can specify which information should be predicted.

For example, in image-based JEPA, it may encode the spatial positions of hidden image patches.

The predictor learns to produce an embedding close to the target encoder's representation:

$$
\hat{s}_y \approx s_y
$$

### 3.4 Prediction Objective

A general JEPA objective can be expressed as

$$
\mathcal{L}_{\text{JEPA}} = D\left(g_\phi(f_\theta(x),z),\operatorname{sg}\left(f_{\bar{\theta}}(y)\right)\right)
$$

where $D$ is a distance or similarity-based loss function, $\operatorname{sg}$ denotes the **stop-gradient operator**, $\theta$ and $\phi$ are trainable parameters, and $\bar{\theta}$ denotes the target encoder parameters.

The stop-gradient operator prevents gradients from flowing through the target representation during the predictor's optimization step.

This creates an asymmetric learning process in which one network predicts representations supplied by another network.

### 3.5 Architecture Diagram

```text
Context x                               Target y
    |                                       |
    v                                       v
Context encoder f_theta             Target encoder f_theta_bar
    |                                       |
    v                                       v
Context representation s_x          Target representation s_y
    |                                       |
    v                                       | (stop-gradient)
Predictor g_phi (conditioned on z)         |
    |                                       |
    v                                       v
Predicted representation s_hat_y ----> Prediction loss D(s_hat_y, s_y)
```

## 4. Learning in Representation Space

The main distinguishing feature of JEPA is the choice of prediction space.

A pixel-level prediction method attempts to learn

$$
\hat{y} = g_\phi(f_\theta(x))
$$

and compares $\hat y$ with the original target $y$.

JEPA instead learns

$$
\hat{s}_y = g_\phi(f_\theta(x),z)
$$

and compares this prediction with the target embedding:

$$
s_y = f_{\bar{\theta}}(y)
$$

This distinction matters because representations can discard information that is irrelevant to the predictive task.

For example, if an image contains a partially occluded dog, its identity, approximate shape, and location may be predictable from the visible regions. The exact texture of its fur may not be.

A good representation can preserve the predictable semantic and geometric properties without encoding every unpredictable detail.

However, **JEPA does not automatically guarantee semantic abstraction**. The quality of the learned representations depends on the prediction task, encoder architecture, training objective, masking strategy, and mechanisms used to avoid representation collapse.

## 5. Avoiding Representation Collapse

A central difficulty in non-contrastive representation learning is **representation collapse**.

Suppose the context encoder and target encoder produce the same constant vector for every input:

$$
f_\theta(x) = f_{\bar{\theta}}(y) = c
$$

If the predictor also outputs $c$, the prediction error becomes zero:

$$
D(c,c)=0
$$

The model would minimize the objective without learning anything useful about the data.

Therefore, JEPA implementations need mechanisms that prevent or discourage this trivial solution.

One approach, used in I-JEPA, is to maintain a separate target encoder whose parameters are updated through an **exponential moving average (EMA)** of the context encoder parameters:

$$
\bar{\theta}_{t} = \tau_t\bar{\theta}_{t-1} + (1-\tau_t)\theta_t
$$

where $\tau_t$ is the momentum coefficient, typically close to $1$.

The target encoder evolves more slowly than the context encoder, providing relatively stable prediction targets.

The combination of a slowly updated target encoder, stop-gradient operation, predictive architecture, and appropriately designed masking supports stable representation learning. EMA alone is not a universal mathematical guarantee against collapse.

## 8. JEPA Compared with Other Learning Paradigms

| Property | Contrastive learning | Masked autoencoders | JEPA |
|---|---|---|---|
| Main objective | Align compatible embeddings and separate negatives | Reconstruct missing input | Predict target embeddings |
| Prediction space | Representation space | Usually pixel space | Representation space |
| Negative samples | Typically required | Not required | Not required in I-JEPA |
| Main learning signal | Similarity and discrimination | Reconstruction error | Predictive consistency |
| Low-level detail reconstruction | Not required | Usually required | Not required |
| Typical representation bias | Invariance to selected transformations | Preservation of reconstructive information | Preservation of predictable information |

JEPA is not simply a combination of contrastive learning and masked reconstruction.

Its defining characteristic is that prediction occurs between **learned representations of compatible observations**, without requiring the model to reconstruct the original input or explicitly distinguish negative examples.

## 9. JEPA and Predictive World Models

JEPA is also relevant to the broader idea of **world models**.

A world model aims to learn representations of an environment and predict how relevant aspects of that environment change or relate to one another.

Instead of predicting every future pixel, a world model can predict the future state of a latent representation:

$$
\hat{s}_{t+1} = g_\phi(s_t,a_t)
$$

where $s_t$ represents the current state and $a_t$ represents an action.

This is a natural extension of the JEPA principle: predict informative latent states rather than complete observations.

However, **I-JEPA itself is not a temporal or action-conditioned world model**. It learns relationships between spatial regions of static images.

Extending the same idea to video, temporal prediction, and action-conditioned dynamics makes it possible to explore representations that capture not only visual structure but also how environments evolve over time.

## 10. Limitations of JEPA

**Representation collapse:** The learning process requires appropriate architectural and optimization mechanisms to avoid constant embeddings.

**Dependence on target representations:** The predictor learns to reproduce the target encoder's representations, so the usefulness of the learned features depends on the quality of those targets.

**Prediction-task design:** If targets are too easy to predict, the model may learn superficial correlations. If they are too difficult or unpredictable, the training signal may become less informative.

**No explicit pixel reconstruction:** JEPA is designed for representation learning rather than direct image generation. Producing images from learned representations requires an additional generative model or decoder.

**No guarantee of causal understanding:** Learning predictive representations does not automatically imply that the model has learned causal relationships, physical dynamics, or an actionable understanding of the environment.