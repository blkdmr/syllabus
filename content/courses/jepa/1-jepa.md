---
title: "JEPA"
weight: 1
categories: ["Theory"]
tags: ["Deep Learning"]
draft: false
---

# What is JEPA?

**JEPA (Joint-Embedding Predictive Architecture)** is a self-supervised learning architecture designed to learn meaningful representations of observations by **predicting abstract features rather than reconstructing raw data**.

Traditional generative models often learn to reconstruct or generate observations, such as images, text, or video frames. JEPA, instead, learns relationships between representations in a **latent space**.

> A latent representation is an internal numerical description of an observation. It may capture information about objects, spatial relationships, motion, or semantic properties without explicitly reproducing every detail of the original input.

JEPA is particularly relevant to:

- Self-supervised representation learning.
- Image and video understanding.
- World Models and physical reasoning.
- Robotics and autonomous agents.

---

## Core Intuition

Consider a video showing a person dropping a ball. After observing the first few frames, a human can predict that the ball will move downward due to gravity and may bounce when it reaches the floor. Making these predictions does not require knowing the exact color of every pixel, the texture of the floor, or the precise lighting conditions.

What matters is understanding the underlying structure of the scene: the objects involved, their positions, their motion, and the physical constraints governing their interactions. *JEPA attempts to learn representations that preserve this type of useful information.*

##### Prediction in observation space

A conventional predictive generative model may attempt to estimate the next observation directly:

$$ x_t \longrightarrow \hat{x}_{t+1} $$

where:

- $x_t$ is the observation at time $t$.
- $\hat{x}_{t+1}$ is the predicted future observation.

*For example, for video prediction, this could involve generating every pixel of the next frame.
Such a model may need to account for details that are difficult to predict, including reflections, lighting changes, and complex textures.*

##### Prediction in latent space

JEPA instead predicts a representation of the target observation:

$$
z_t \longrightarrow \hat{z}_{t+1}
$$

where:

- $z_t$ is the learned representation of the current observation.
- $\hat{z}_{t+1}$ is the predicted representation of a future observation.

The model is not required to reconstruct the complete future frame. Instead, it learns to predict features that its target encoder extracts from that frame.

*Temporal prediction is only one possible application of JEPA. In image-based JEPA, the target may be a hidden region of the same image rather than a future observation.*

Furthermore, predicting in latent space does not automatically guarantee that the learned features correspond to physically meaningful concepts. This depends on the training objective, data, and architecture.

---

## JEPA Architecture

A typical JEPA architecture contains three main components:

- **Context Encoder:** extracts representations from the available information.
- **Predictor:** predicts representations of missing or future information.
- **Target Encoder:** generates the representations that the Predictor must approximate.

The architecture learns by comparing predicted representations with target representations.

#### Context Encoder

The **Context Encoder** transforms an observed input into a latent representation.

Let $ x \in \mathcal{X}$ represent the **available context**, such as visible image patches or observed video frames. The encoder is a *neural network* $ f_\theta:\mathcal{X}\rightarrow\mathcal{Z} $ parameterized by $\theta$. Its output is:

$$
\boxed{z_x=f_\theta(x)}
$$

where $z_x$ belongs to the learned representation space $\mathcal{Z}$.

Depending on the training task, this representation may capture information about object structure, spatial relationships, geometry, motion, or semantics. These properties are not explicitly programmed into the encoder. They are learned through the prediction objective.

*For example, an encoder processing an image containing a partially hidden dog may learn features describing the animal's shape and surrounding context. Such information can help predict representations of the hidden region.*

#### Predictor

The **Predictor** receives the *context representation* and estimates the **representation of a target observation**. Its general formulation is:

$$
\boxed{
\hat{z}_y=g_\phi(z_x,m)
}
$$

where:

- $g_\phi$ is the Predictor parameterized by $\phi$.
- $z_x$ is the context representation.
- $m$ specifies the prediction target, such as its spatial position or temporal location.
- $\hat{z}_y$ is the predicted target representation.

The variable $m$ is important because the same context can support several different predictions.

*For example, consider an image in which multiple regions are hidden. The predictor must know which region it should predict. Positional information allows it to distinguish the target locations. Rather than generating the missing pixels, it estimates the latent features associated with each target region.*

#### Target Encoder

The **Target Encoder** produces the *reference representation* that the predictor is trained to approximate.

In the general formulation:

$$
\boxed{
z_y=f_{\bar{\theta}}(y)
}
$$

where:

- $y$ is the target input.
- $f_{\bar{\theta}}$ is the target encoder.
- $\bar{\theta}$ represents its parameters.
- $z_y$ is the target representation.

The training objective encourages $\hat{z}_y \approx z_y$. The target encoder provides the learning signal
without requiring manually annotated labels.

*In I-JEPA, the target encoder processes the full image, and the representations corresponding to the selected target patches are extracted afterward. This is different from simply encoding an isolated crop of the missing region. Processing the complete image allows the target representations to incorporate information from the surrounding scene.*

#### Overall Architecture

The complete predictive process can be represented as:

$$
\boxed{
x
\xrightarrow{f_\theta}
z_x
\xrightarrow{g_\phi(\cdot,m)}
\hat{z}_y
}
$$

Meanwhile, the Target Encoder produces:

$$
\boxed{
y
\xrightarrow{f_{\bar{\theta}}}
z_y
}
$$

The predicted and target representations are compared through a loss function:

$$
\mathcal{L}=d(\hat{z}_y,z_y)
$$

where $d$ measures their discrepancy.

The Context Encoder learns to extract useful information, while the Predictor learns relationships between the observed context and the prediction target.

---

## 4. JEPA Training Objective

### 4.1 Representation Prediction Loss

The fundamental objective of JEPA is to minimize the difference between predicted and target representations.

A simple example is the squared Euclidean distance:

$$
\boxed{
\mathcal{L}_{\text{JEPA}}
=
\left\|
\hat{z}_y-z_y
\right\|_2^2
}
$$

Substituting the encoder and predictor functions:

$$
\mathcal{L}_{\text{JEPA}}
=
\left\|
g_\phi(f_\theta(x),m)
-
f_{\bar{\theta}}(y)
\right\|_2^2
$$

More generally, the training objective is:

$$
\boxed{
\min_{\theta,\phi}
\mathbb{E}_{(x,y,m)}
\left[
d\left(
g_\phi(f_\theta(x),m),
f_{\bar{\theta}}(y)
\right)
\right]
}
$$

The distance function $d$ depends on the implementation.

Possible choices include the squared Euclidean distance and L1-based losses:

$$
d_{\text{L1}}(\hat{z},z)
=
\|\hat{z}-z\|_1
$$

I-JEPA uses a feature-prediction objective based on differences between predicted and target embeddings.

The key property is that **the loss is computed in representation space, not directly in pixel space**.

### 4.2 Parameter Optimization

The Context Encoder and Predictor are optimized through gradient descent.

The Target Encoder is typically updated through a separate mechanism rather than receiving gradients from the prediction loss.

This separation helps stabilize the learning process.

In particular, I-JEPA uses an **Exponential Moving Average (EMA)** to update the Target Encoder parameters.

### 4.3 Exponential Moving Average

The Target Encoder is maintained as a slowly changing copy of the Context Encoder.

After updating the Context Encoder, the Target Encoder parameters are updated according to:

$$
\boxed{
\bar{\theta}_{k+1}
=
\tau\bar{\theta}_k
+
(1-\tau)\theta_{k+1}
}
$$

where:

- $\theta_{k+1}$ contains the updated Context Encoder parameters.
- $\bar{\theta}_k$ contains the previous Target Encoder parameters.
- $\tau$ is the EMA momentum coefficient, usually close to $1$.

For example, with $\tau=0.996$:

$$
\bar{\theta}_{k+1}
=
0.996\bar{\theta}_k
+
0.004\theta_{k+1}
$$

The Target Encoder therefore changes more slowly than the Context Encoder.

This provides relatively stable targets for the Predictor.

In I-JEPA, the EMA mechanism is combined with a stop-gradient operation: the target representations are treated as fixed values during each gradient update.

EMA contributes to stable learning, although it does not provide a mathematical guarantee against representation collapse.

---

## 5. Self-Supervised Learning

JEPA belongs to the family of **self-supervised learning methods**.

The main objective of self-supervised learning is to learn representations from data without requiring manually assigned labels.

### 5.1 Supervised Learning

In conventional supervised learning, the training dataset contains input-label pairs:

$$
\mathcal{D}=\{(x_i,y_i)\}_{i=1}^{N}
$$

For image classification, $x_i$ might be an image and $y_i$ its manually assigned category.

The model learns a mapping from observations to labels.

### 5.2 Self-Supervised Learning

In self-supervised learning, training targets are constructed automatically from the input data.

For example, an image can be divided into an observed context and several hidden target regions.

The model receives the context and is trained to predict the representations of the targets.

There is no need for a human to annotate the image with semantic labels.

For a dataset containing $N$ training examples, the objective can be written as:

$$
\boxed{
\mathcal{L}
=
\frac{1}{N}
\sum_{i=1}^{N}
d\left(
g_\phi(f_\theta(x_i),m_i),
f_{\bar{\theta}}(y_i)
\right)
}
$$

The training signal is obtained directly from the data through the Target Encoder.

This enables JEPA models to learn from large collections of unlabeled images or videos.

### 5.3 Masking Strategy

Masking is particularly important in image-based and video-based JEPA architectures.

During training, parts of the observation are excluded from the Context Encoder.

The Predictor must estimate the representations associated with these missing regions.

The masking strategy determines how much information remains available and which features must be predicted.

If the task is too easy, the model may solve it using local correlations without learning useful semantic structure.

If the task is too difficult, the context may not contain enough information to support an informative prediction.

In I-JEPA, the use of sufficiently large target blocks and a spatially informative context is a central design choice.

Large target regions encourage the model to learn higher-level relationships rather than relying exclusively on local image details.

---

## 6. Why Predict in Latent Space?

Real-world observations contain enormous amounts of information, but not all of that information is equally predictable or relevant.

Consider a video showing a moving car.

The car's approximate trajectory, position relative to the road, and interactions with nearby objects may be predictable from the preceding frames.

However, the exact configuration of reflections on the windshield or the movement of individual leaves may be difficult to estimate.

A model trained to reconstruct every pixel must account for many such details.

JEPA instead attempts to learn representations that retain information useful for the prediction task.

### 6.1 Representation Learning

The encoder maps an observation into a latent representation:

$$
x\xrightarrow{f_\theta}z
$$

The representation $z$ is learned so that the Predictor can estimate the target embeddings from the available context.

Ideally, information that cannot be inferred reliably from the context does not dominate the prediction objective.

For example, a useful representation of a moving ball may preserve information about its position and motion without explicitly representing the exact appearance of every pixel.

### 6.2 Predictable and Unpredictable Information

A major motivation for JEPA is the distinction between predictable structure and unpredictable details.

If multiple future observations are consistent with the same context, predicting one exact future observation can be unnecessarily difficult.

By learning suitable latent representations, the model may focus on properties shared by those possible outcomes.

However, this benefit depends on the learned representation.

A latent space does not automatically discard irrelevant information or preserve all information required for physical reasoning.

The encoder and training objective determine which properties are retained.

### 6.3 Representation Space Is Not Necessarily Lower-Dimensional

Although latent representations are often described as compressed descriptions of observations, they do not necessarily have fewer numerical dimensions than the original data.

Their defining property is that they provide a **learned feature representation** in which the prediction task is performed.

The advantage of JEPA therefore comes from the structure of the learned representations and the training objective, not simply from dimensionality reduction.

---

## 7. Representation Collapse

A fundamental challenge in joint-embedding architectures is **representation collapse**.

Representation collapse occurs when the encoder produces representations that contain little or no useful information about the input.

### 7.1 Constant Representation Collapse

Suppose the encoder maps every input to the same constant vector:

$$
f_\theta(x)=c
\qquad
\forall x
$$

If the Target Encoder also produces the same constant vector and the Predictor outputs:

$$
g_\phi(c,m)=c
$$

then the prediction loss becomes:

$$
\mathcal{L}_{\text{JEPA}}
=
\|c-c\|_2^2
=
0
$$

The model achieves perfect prediction loss without learning anything useful.

This is a trivial solution.

The objective of representation learning is therefore not merely to minimize prediction error, but to learn representations that preserve useful distinctions between observations.

### 7.2 Preventing Collapse

Different joint-embedding methods use different mechanisms to reduce the risk of collapse.

These include asymmetric architectures, stop-gradient operations, momentum-based target encoders, representation regularization, and carefully designed prediction tasks.

In I-JEPA, the main mechanisms include:

- A Target Encoder updated through EMA.
- Stop-gradient on target representations.
- A masking strategy that encourages meaningful contextual prediction.
- Architectural and optimization choices that support stable training.

These mechanisms help the model learn informative representations, although none should be interpreted as a universal guarantee against collapse.

---

## 8. JEPA and World Models

JEPA is closely associated with research on **World Models**.

A World Model is an internal predictive model that represents aspects of an environment and estimates how those aspects may evolve.

This capability is important for artificial agents that must interact with the physical world.

For example, a robot moving an object must reason about the object's location, the effects of its actions, and the consequences of possible interactions.

A predictive latent-space model can provide part of the machinery required for this reasoning.

### 8.1 Latent State Representation

Let:

$$
o_t
$$

represent the observation received by an agent at time $t$.

An encoder produces a latent state:

$$
\boxed{
z_t=f_\theta(o_t)
}
$$

The latent state is intended to capture information useful for predicting and controlling the environment.

Depending on the task, it may represent object positions, geometry, motion, or interactions.

However, a state representation computed from one observation may not contain all relevant information about the environment.

For partially observable tasks, the model may need information from multiple observations or an internal memory.

### 8.2 Action-Conditioned Prediction

A World Model intended for planning must generally account for the actions performed by an agent.

Let $a_t$ represent an action executed at time $t$.

An action-conditioned predictive model can be written as:

$$
\boxed{
\hat{z}_{t+1}
=
g_\phi(z_t,a_t)
}
$$

where:

- $z_t$ is the current latent state.
- $a_t$ is the selected action.
- $\hat{z}_{t+1}$ is the predicted next latent state.

For example, a robot may predict how the position of a cup will change if it pushes the cup in a particular direction.

The model can estimate the consequences of candidate actions before those actions are executed.

**Not every JEPA architecture is action-conditioned.**

Standard I-JEPA is primarily designed for image representation learning. Action-conditioned variants extend the predictive framework toward modeling environmental dynamics.

### 8.3 Multi-Step Prediction

A learned transition model can be applied repeatedly to estimate future states.

Starting from the current state:

$$
\hat{z}_{t+1}
=
g_\phi(z_t,a_t)
$$

The next prediction becomes:

$$
\hat{z}_{t+2}
=
g_\phi(\hat{z}_{t+1},a_{t+1})
$$

Continuing:

$$
\hat{z}_{t+3}
=
g_\phi(\hat{z}_{t+2},a_{t+2})
$$

More generally:

$$
\boxed{
\hat{z}_{t+k}
=
g_\phi^{(k)}
\left(
z_t,
a_t,\ldots,a_{t+k-1}
\right)
}
$$

where $g_\phi^{(k)}$ denotes the repeated application of the transition model.

This allows the model to simulate possible future trajectories in latent space.

A significant difficulty is that prediction errors may accumulate over time.

An inaccurate prediction at one step can affect every subsequent prediction, making long-horizon forecasting particularly challenging.

---

## 9. JEPA and Planning

A predictive World Model can be combined with a planning algorithm to select actions that move an agent toward a desired goal.

Consider a robot that must move an object to a specific position.

Instead of immediately executing an action, the robot can evaluate several possible action sequences.

### 9.1 Candidate Action Sequences

Let:

$$
A=(a_t,a_{t+1},\ldots,a_{t+H-1})
$$

represent a candidate sequence of actions.

Here, $H$ is the planning horizon.

For each candidate sequence, the World Model predicts a trajectory of future latent states.

### 9.2 Cost Function

A cost function evaluates how desirable a predicted trajectory is.

A simple formulation is:

$$
\boxed{
J(A)
=
\sum_{k=1}^{H}
C(\hat{z}_{t+k},a_{t+k-1})
}
$$

where $C$ assigns a cost to a predicted state and its associated action.

For a robot manipulation task, the cost may depend on the distance between the predicted object position and the desired target position.

Additional penalties may account for collisions, excessive motion, or unsafe actions.

### 9.3 Action Optimization

The planner searches for the action sequence that minimizes the total predicted cost:

$$
\boxed{
A^*=\arg\min_A J(A)
}
$$

The resulting sequence represents the best candidate according to the model and the chosen objective.

The robot may execute only the first action, observe the updated environment, and repeat the optimization process.

This approach is known as **Model Predictive Control (MPC)**.

MPC is useful because the planner continuously updates its predictions using new observations.

### 9.4 Relationship Between JEPA and Planning

JEPA alone does not constitute a complete planning system.

A planning agent requires several complementary capabilities:

- A representation of the current environment.
- A predictive model of environmental dynamics.
- A model of how actions affect future states.
- A cost or reward function.
- An optimization procedure for selecting actions.

JEPA-style representation learning and prediction can support the first components, while planning and control algorithms provide the decision-making machinery.

---

## 10. JEPA vs. Generative Models

The central distinction between JEPA and generative modeling concerns **what the model is trained to predict**.

### 10.1 Generative Models

Generative models learn to produce observations or estimate distributions over observations.

A probabilistic generative model may learn:

$$
p_\theta(x)
$$

An autoregressive model factorizes a joint distribution as:

$$
\boxed{
p_\theta(x_1,\ldots,x_T)
=
\prod_{t=1}^{T}
p_\theta(x_t\mid x_{<t})
}
$$

where $x_{<t}$ denotes the observations preceding position $t$.

In a language model, the sequence may consist of tokens.

In a video model, the system may be trained to generate future frames from previously observed frames.

Generative models must ultimately define a way to produce outputs in the original data domain.

Some perform generation directly, while others operate through learned latent variables and decoders.

### 10.2 JEPA

JEPA learns relationships between embeddings:

$$
\boxed{
\hat{z}_y
=
g_\phi(f_\theta(x),m)
}
$$

The predicted embedding is compared with the output of the Target Encoder:

$$
z_y=f_{\bar{\theta}}(y)
$$

The model is optimized so that:

$$
\hat{z}_y\approx z_y
$$

It does not need to reconstruct $y$.

Therefore, the key difference is not simply that JEPA uses latent variables, since many generative models also operate in latent spaces.

The distinction is that **JEPA is trained to predict target representations without requiring reconstruction of the original observations**.

### 10.3 Comparison

| Property | Generative Models | JEPA |
|---|---|---|
| Main objective | Model or generate observations | Predict target representations |
| Prediction target | Data or generative latent variables | Learned feature embeddings |
| Reconstruction | Often part of the generation process | Not required |
| Typical output | Text, images, audio, or video | Latent feature representations |
| Training signal | Generative or reconstruction objective | Representation prediction loss |
| Main applications | Content generation and probabilistic modeling | Representation learning and predictive modeling |
| Planning | Requires additional mechanisms | Requires additional mechanisms |

The approaches are not mutually exclusive.

Generative components and JEPA-style predictive representations can be combined within larger AI systems.

---

## 11. Main JEPA Variants

JEPA is a general architectural principle rather than a single neural network.

Different implementations adapt the framework to particular modalities and applications.

### 11.1 I-JEPA: Image-Based JEPA (2023)

**I-JEPA** applies joint-embedding prediction to images.

The architecture uses Vision Transformers and a masking strategy based on image patches.

During training, the Context Encoder processes the visible patches, while the Target Encoder processes the full image.

The Predictor receives the context embeddings and positional information specifying the target regions.

It predicts the representations of the target patches without reconstructing their pixels.

Conceptually:

$$
\boxed{
\hat{z}_{\text{target}}
=
g_\phi
\left(
f_\theta(x_{\text{context}}),
m
\right)
}
$$

A central feature of I-JEPA is the use of relatively large target blocks.

This encourages the model to predict higher-level features that depend on the overall structure of the image.

The learned representations can subsequently be used for downstream tasks such as image classification, object understanding, and depth estimation.

### 11.2 V-JEPA: Video-Based JEPA (2024)

**V-JEPA** extends joint-embedding prediction to video.

A video can be represented as a sequence of frames:

$$
X=(x_1,x_2,\ldots,x_T)
$$

Unlike image-based JEPA, video learning involves both spatial and temporal relationships.

V-JEPA uses masking across space and time.

The Context Encoder processes the visible portions of a video, and the Predictor estimates representations associated with masked space-time regions.

This training procedure encourages the learned representations to capture information about motion, object interactions, and temporal structure.

**Predicting masked video regions is not necessarily the same as forecasting future frames.**

A masked target may occur anywhere within the video clip rather than exclusively after the observed frames.

### 11.3 V-JEPA 2 (2025)

**V-JEPA 2** extends video-based self-supervised learning toward visual understanding, physical prediction, and planning.

The architecture learns visual representations from large-scale video data.

These representations can be used in tasks involving physical reasoning and the understanding of object interactions.

A related model, **V-JEPA 2-AC**, introduces action conditioning for robotic prediction.

Its general transition model can be expressed as:

$$
\boxed{
\hat{z}_{t+1}
=
g_\phi(z_t,a_t)
}
$$

This enables the model to estimate future latent states under different robot actions.

The approach has been investigated for robotic manipulation tasks, including reaching and grasping.

### 11.4 V-JEPA 2.1 (2026)

**V-JEPA 2.1** develops video representation learning further by focusing on spatially dense, temporally consistent representations.

Its main improvements include dense predictive learning, deep self-supervision, unified image-video training, and increased model and data scale.

A significant modification is the use of a **dense predictive loss**, in which both visible and masked tokens contribute to the training objective.

This encourages the representations to preserve information about spatial structure while maintaining strong semantic understanding.

Deep self-supervision additionally applies predictive learning objectives to intermediate encoder representations.

These modifications aim to improve performance on tasks requiring detailed visual understanding, including object interactions, action anticipation, depth estimation, and robotic manipulation.

---

## 12. JEPA vs. Large Language Models

JEPA is sometimes discussed in relation to Large Language Models because both approaches can learn useful representations through predictive training.

However, their standard training objectives are fundamentally different.

### 12.1 Autoregressive Language Models

A conventional autoregressive Large Language Model learns to predict the next token given the preceding tokens:

$$
\boxed{
P(x_{t+1}\mid x_1,\ldots,x_t)
}
$$

By training on large amounts of text, the model learns statistical relationships involving language, knowledge, and reasoning patterns.

The training objective is typically based on maximizing the probability assigned to the correct next token.

### 12.2 JEPA

JEPA predicts latent representations rather than next-token probabilities:

$$
\boxed{
\hat{z}_y
=
g_\phi(f_\theta(x),m)
}
$$

Its objective is to approximate the target representation:

$$
\hat{z}_y\approx f_{\bar{\theta}}(y)
$$

In visual applications, the target may describe a hidden image region, a masked video region, or a future environmental state.

### 12.3 Conceptual Difference

Autoregressive LLMs are generally trained to predict sequences of discrete tokens.

JEPA models are trained to predict features in a learned representation space.

This distinction can be particularly relevant when the prediction target contains information that is difficult or unnecessary to reproduce exactly.

For physical reasoning, a useful predictive system may need to estimate how objects move, interact, or respond to actions without generating a complete visual reconstruction.

However, neither approach automatically guarantees physical understanding, causal reasoning, or general intelligence.

### 12.4 Hybrid Architectures

Language models and JEPA-style World Models can potentially be combined.

A language model could support communication and high-level reasoning, while a JEPA-based visual system could provide representations and predictions of the physical environment.

A planning system could then use those predictions to select actions.

Such a combination may support embodied AI systems that integrate language understanding, perception, environmental prediction, and physical interaction.

---

## 13. Advantages and Limitations

### 13.1 Advantages

**Learning without manual labels.** JEPA can learn representations from large collections of unlabeled images and videos through self-supervised prediction.

**Prediction in representation space.** The model is not required to reconstruct every detail of an observation. This can reduce the burden of modeling fine-grained or unpredictable information.

**Learning useful visual features.** The predictive objective can encourage representations that capture semantic structure, spatial relationships, and temporal information.

**Potential computational efficiency.** Avoiding full observation reconstruction may reduce computational requirements, although the actual efficiency depends on the architecture, dataset, and training procedure.

**Applications to World Models.** Action-conditioned JEPA variants can provide predictive representations for robotics and planning systems.

### 13.2 Limitations

**Representation collapse.** Without appropriate training mechanisms, joint-embedding systems can converge to trivial or insufficiently informative representations.

**Limited interpretability.** The dimensions of a learned latent representation do not necessarily correspond to identifiable physical quantities.

**Incomplete physical understanding.** Successfully predicting embeddings does not prove that a model has learned causal relationships or general physical laws.

**Prediction uncertainty.** A deterministic Predictor may struggle when several different outcomes are possible. A single predicted embedding may fail to represent the complete uncertainty over future states.

**Long-horizon prediction.** Repeatedly applying a transition model can accumulate errors, reducing the reliability of predictions over extended time horizons.

**Dependence on the learned representation.** Important information may be discarded if it is not useful for the original self-supervised objective. Representations suitable for recognition are not necessarily sufficient for control.

**Planning requires additional components.** A standard JEPA architecture is not an autonomous agent. Planning requires appropriate dynamics models, action representations, objectives, and control mechanisms.

---

## 14. References

1. **Assran et al. (2023)** — *Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture.*  
   [CVPR Paper](https://openaccess.thecvf.com/content/CVPR2023/html/Assran_Self-Supervised_Learning_From_Images_With_a_Joint-Embedding_Predictive_Architecture_CVPR_2023_paper.html)

2. **Bardes et al. (2024)** — *Revisiting Feature Prediction for Learning Visual Representations from Video.*  
   [arXiv](https://arxiv.org/abs/2404.08471)

3. **Assran et al. (2025)** — *V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning.*  
   [arXiv](https://arxiv.org/abs/2506.09985)

4. **Mur-Labadia et al. (2026)** — *V-JEPA 2.1: Unlocking Dense Features in Video Self-Supervised Learning.*  
   [arXiv](https://arxiv.org/abs/2603.14482)

5. **LeCun (2022)** — *A Path Towards Autonomous Machine Intelligence.*  
   [OpenReview](https://openreview.net/forum?id=BZ5a1r-kVsf)