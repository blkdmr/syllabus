---
title: "TCN and Transformers"
weight: 30
categories: ["Theory"]
tags: ["Deep Learning"]
draft: false
---

# TCN and Transformers

## Sequence Modeling

A **sequence model** receives an ordered sequence of elements and produces either a single output or another sequence.

Examples of sequential inputs include:

- words or characters in text;
- time-series measurements;
- audio signals;
- video frames.

Typical sequence-modeling tasks include machine translation, time-series forecasting, video captioning, audio generation, and text summarization.

Although sequence modeling has historically been associated with **Recurrent Neural Networks (RNNs)**, convolutional architectures and Transformers can outperform recurrent models on many tasks.

---

# Temporal Convolutional Architectures

## From 2D to 1D Convolution

A standard **2D convolution** operates on two spatial dimensions and is therefore particularly suitable for images.

For sequential data, the analogous operation is a **1D convolution**, where the kernel slides along a single dimension representing time or sequence position.

A discrete causal-style 1D convolution can be written as

$$
(\mathbf{x} * \omega)(n) =
\sum_{i=0}^{k-1}
\mathbf{x}(n-i)\omega(i),
$$

where:

- $\mathbf{x}$ is the input sequence;
- $\omega$ is a kernel of length $k$;
- $n$ denotes the current sequence position.

For multivariate sequences or token embeddings, each sequence element is itself a vector. Therefore, a convolutional filter must span both:

1. a temporal window of $k$ elements;
2. the feature or embedding dimension.

For example, if each word is represented by a $4$-dimensional embedding and the temporal kernel covers $3$ words, the kernel has shape $3\times4$.

A 1D CNN can therefore model local temporal dependencies in time series and local syntactic patterns in text.

---

## Limitation of Standard Convolutions

The main difficulty of using ordinary CNNs for long sequences is their limited **receptive field**.

A layer with kernel size $k$ sees only a small local neighborhood. Capturing dependencies between very distant sequence elements consequently requires:

- many convolutional layers, or
- very large kernels.

This motivates **dilated convolution**.

---

## Dilated Convolution

A **dilated convolution** inserts gaps between the positions sampled by the kernel.

It is defined as

$$
(\mathbf{x} *_d \omega)(n) =
\sum_{i=0}^{k-1}
\mathbf{x}(n-di)\omega(i),
$$

where $d$ is the **dilation factor**.

For example:

- $d=1$: consecutive inputs are used;
- $d=2$: every second input is used;
- $d=4$: every fourth input is used.

Dilated convolutions increase the receptive field without proportionally increasing the number of parameters.

For a stack of $L$ convolutional layers with kernel size $k$ and dilation factors $d_1,\dots,d_L$, assuming stride $1$, the receptive field is

$$
R =
1+(k-1)\sum_{\ell=1}^{L}d_\ell.
$$

If dilation doubles at every layer,

$$
d_\ell=2^{\ell-1},
$$

then

$$
R =
1+(k-1)(2^L-1).
$$

Hence, the receptive field grows exponentially with network depth.

---

## Causal Convolution

A standard convolution may use both past and future inputs when computing the representation at position $t$.

For forecasting or autoregressive tasks, this can introduce **future information leakage**.

A **causal convolution** imposes

$$
y_t =
f(x_t,x_{t-1},x_{t-2},\dots),
$$

so the output at time $t$ depends only on observations at times $\leq t$.

Causality is essential whenever future observations must not influence current predictions.

---

# Temporal Convolutional Networks

A **Temporal Convolutional Network (TCN)** combines:

- 1D convolutions;
- causal convolutions;
- dilated convolutions;
- residual connections.

A typical TCN consists of residual blocks with progressively increasing dilation factors, for example

$$
d=1,2,4,8,\dots
$$

with a fixed kernel size such as $k=3$.

The increasing dilation gives higher layers access to progressively longer temporal histories.

## TCNs Compared with RNNs

TCNs have several practical advantages over recurrent architectures such as LSTMs and GRUs.

### Parallel computation

RNNs have sequential dependencies:

$$
h_t=f(h_{t-1},x_t),
$$

so computation at time $t$ depends on time $t-1$.

Convolutional layers process many sequence positions simultaneously, making TCNs substantially more parallelizable.

### Long effective memory

Dilated convolutions allow a TCN to capture long-range dependencies using a moderate number of layers.

### Gradient propagation

Residual connections and the absence of recurrent state transitions generally make optimization more stable and reduce the severity of vanishing-gradient problems compared with long recurrent chains.

It is too strong to say that TCNs can *never* suffer from vanishing or exploding gradients; rather, their architecture tends to mitigate these problems.

### Depth determines temporal context

Increasing network depth increases the receptive field. Therefore, the temporal history accessible to a TCN can be controlled explicitly through its architecture.

---

## Finance Case Study

A TCN-based financial forecasting system can combine several information sources:

- **price data** $X$: historical values of the DJIA;
- **news data** $N$: historical news headlines;
- **knowledge data** $G$: structured information obtained from resources such as Freebase and Wikidata.

The objective is binary prediction of the next market movement:

$$
\text{movement}\in\{\text{up},\text{down}\}.
$$

Reported baseline results include:

| Model | Accuracy | F1 Score |
|---|---:|---:|
| ARIMA | $52.40\%\pm0.25\%$ | $0.3844\pm0.0052$ |
| LSTM | $56.17\%\pm0.26\%$ | $0.5348\pm0.0039$ |
| CNN | $55.41\%\pm0.25\%$ | $0.5280\pm0.0021$ |
| TCN | $60.96\%\pm0.26\%$ | $0.5317\pm0.0036$ |

Adding textual and structured knowledge representations further improves the reported results, with the knowledge-driven model reaching

$$
69.77\%\pm0.25\%
$$

accuracy and

$$
0.7073\pm0.0030
$$

F1 score.

---

# From TCNs to Transformers

TCNs impose a strong **inductive bias**:

> nearby sequence elements interact first, while progressively larger temporal contexts are obtained through deeper and dilated convolutions.

This assumption is useful when temporal structure is approximately local and hierarchical.

However, important dependencies are not always local.

For example, in source code, the beginning of a loop or function may determine the interpretation of a variable appearing dozens of lines later.

A desirable model would therefore be able to learn directly from data **which elements should interact**, independently of their distance.

This motivates **attention-based architectures**, particularly the Transformer.

---

# Transformer Architecture

The original Transformer consists of two main components:

- an **encoder**;
- a **decoder**.

In the original architecture introduced by Vaswani et al., both were built from stacks of $N=6$ layers.

Transformers are now widely used in:

- machine translation;
- text generation;
- question answering;
- dialogue systems;
- computer vision;
- multimodal learning.

The following discussion focuses primarily on the **encoder**.

---

# Transformer Encoder

Suppose the input sequence contains $T$ tokens, each represented by a vector of dimension $d_{\text{model}}$:

$$
X\in\mathbb{R}^{T\times d_{\text{model}}}.
$$

The encoder produces another sequence with the same dimensionality:

$$
Z\in\mathbb{R}^{T\times d_{\text{model}}}.
$$

There is normally no temporal pooling inside the encoder itself. Instead, every input token receives a new **contextualized representation**.

Each encoder block contains:

- Multi-Head Self-Attention;
- a position-wise Feed-Forward Network;
- residual connections;
- Layer Normalization.

In the original Transformer, Layer Normalization is applied after each residual addition, a design commonly called **Post-LN**.

---

# Feed-Forward Network

After attention, every token independently passes through the same feed-forward network.

A standard formulation is

$$
\operatorname{FFN}(x) =
W_2\,\sigma(W_1x+b_1)+b_2,
$$

where $\sigma$ was originally the ReLU activation.

Typically,

$$
W_1:
\mathbb{R}^{d_{\text{model}}}
\rightarrow
\mathbb{R}^{d_{\text{ff}}},
$$

followed by

$$
W_2:
\mathbb{R}^{d_{\text{ff}}}
\rightarrow
\mathbb{R}^{d_{\text{model}}}.
$$

The same feed-forward network is applied independently to all sequence positions.

---

# Residual Connections

A residual connection adds the input of a module to its output:

$$
y=x+f(x).
$$

Residual connections help preserve information and improve gradient propagation through deep architectures.

In the original Transformer encoder, a sublayer is approximately

$$
\operatorname{LayerNorm}
\left(
x+\operatorname{Sublayer}(x)
\right).
$$

---

# Layer Normalization

**Layer Normalization** normalizes the features of each individual token rather than computing statistics across different samples in a minibatch.

Given

$$
\mathbf{x} =
(x_1,\dots,x_H),
$$

the mean is

$$
\mu =
\frac{1}{H}
\sum_{i=1}^{H}x_i,
$$

and the variance is

$$
\sigma^2 =
\frac{1}{H}
\sum_{i=1}^{H}(x_i-\mu)^2.
$$

Layer Normalization is

$$
\operatorname{LN}(\mathbf{x}) =
\gamma\odot
\frac{\mathbf{x}-\mu}
{\sqrt{\sigma^2+\epsilon}}
+
\beta,
$$

where:

- $\gamma$ is a learnable scaling vector;
- $\beta$ is a learnable bias vector;
- $\epsilon>0$ provides numerical stability.

Normalization is typically performed independently for each token across its $d_{\text{model}}$ features.

## LayerNorm vs BatchNorm

**Batch Normalization** computes statistics using multiple samples in a batch, whereas **Layer Normalization** computes statistics inside each individual sample or token.

Consequently, LayerNorm:

- does not depend on batch size;
- behaves consistently during training and inference;
- naturally handles variable sequence lengths;
- is particularly suitable for Transformers and other sequence models.

---

# Core Idea of Attention

In many sequence tasks, different input elements have different relevance to a given output.

Attention allows the network to determine dynamically **which elements should receive more weight**.

For an input sequence

$$
X=
[\mathbf{x}_1,\dots,\mathbf{x}_T],
$$

an attention mechanism produces contextualized vectors

$$
\mathbf{z}_t =
\sum_{i=1}^{T}
\alpha_{t,i}\mathbf{x}_i,
$$

where

$$
\sum_{i=1}^{T}\alpha_{t,i}=1,
\qquad
\alpha_{t,i}\geq0.
$$

The attention weights are obtained from learned compatibility scores:

$$
\alpha_{t,i} =
\operatorname{softmax}_i(s_{t,i}).
$$

Therefore, the weights are **input dependent**, rather than fixed parameters associated permanently with particular positions.

---

# Attention as Content-Based Memory Access

Attention can be interpreted as a differentiable memory-retrieval mechanism.

Each memory item contains:

- a **key** describing what information is stored;
- a **value** representing the information itself.

A **query** represents the information currently being requested.

For query $\mathbf{q}$ and key-value pairs

$$
\{(\mathbf{k}_i,\mathbf{v}_i)\}_{i=1}^{T},
$$

attention first computes similarity scores

$$
s_i =
\operatorname{similarity}
(\mathbf{q},\mathbf{k}_i).
$$

The scores are normalized:

$$
\alpha_i =
\frac{e^{s_i}}
{\sum_j e^{s_j}}.
$$

The retrieved representation becomes

$$
\mathbf{z} =
\sum_i\alpha_i\mathbf{v}_i.
$$

Retrieval therefore depends primarily on **content similarity**, rather than simply on the physical distance between sequence positions.

---

# 12. Queries, Keys, and Values

Given an input matrix

$$
X\in\mathbb{R}^{T\times d_{\text{model}}},
$$

the Transformer learns three linear projections:

$$
Q=XW^Q,
$$

$$
K=XW^K,
$$

$$
V=XW^V.
$$

For one attention head,

$$
W^Q\in
\mathbb{R}^{d_{\text{model}}\times d_k},
$$

$$
W^K\in
\mathbb{R}^{d_{\text{model}}\times d_k},
$$

and, more generally,

$$
W^V\in
\mathbb{R}^{d_{\text{model}}\times d_v}.
$$

Thus,

$$
Q,K\in\mathbb{R}^{T\times d_k},
$$

while

$$
V\in\mathbb{R}^{T\times d_v}.
$$

When queries, keys, and values all originate from the same input sequence, the mechanism is called **self-attention**.

---

# 13. Scaled Dot-Product Attention

The similarity between queries and keys is computed using a dot product.

For the complete sequence,

$$
QK^\top
\in
\mathbb{R}^{T\times T}.
$$

Entry $(t,i)$ measures the compatibility between the query generated by token $t$ and the key associated with token $i$.

Scaled dot-product attention is

$$
\operatorname{Attention}(Q,K,V) =
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d_k}}
\right)V.
$$

The softmax is applied row-wise.

Therefore,

$$
A =
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d_k}}
\right)
$$

is an attention matrix

$$
A\in\mathbb{R}^{T\times T},
$$

and

$$
Z=AV.
$$

Each output token representation is therefore a weighted combination of the value vectors of all tokens.

---

## 13.1 Why Divide by $\sqrt{d_k}$?

Suppose query and key components are approximately independent, zero-mean random variables with unit variance.

Their dot product is

$$
q^\top k =
\sum_{j=1}^{d_k}q_jk_j.
$$

Its variance grows approximately proportionally to $d_k$:

$$
\operatorname{Var}(q^\top k)
\approx d_k.
$$

Large dot products can make the softmax distribution extremely peaked, producing very small gradients.

Scaling by

$$
\frac{1}{\sqrt{d_k}}
$$

keeps the variance of the logits approximately constant as the key dimension increases.

---

# 14. Multi-Head Attention

A single attention head learns one set of query, key, and value projections.

**Multi-Head Attention (MHA)** repeats this process in parallel using different learned projections.

For head $i$,

$$
\operatorname{head}_i =
\operatorname{Attention}
\left(
XW_i^Q,
XW_i^K,
XW_i^V
\right).
$$

The heads are concatenated:

$$
H =
\operatorname{Concat}
(
\operatorname{head}_1,
\dots,
\operatorname{head}_h
).
$$

A final projection produces

$$
\operatorname{MHA}(X) =
H W^O.
$$

In the original Transformer,

$$
d_k=d_v=\frac{d_{\text{model}}}{h}.
$$

Thus, concatenating all $h$ heads recovers dimension

$$
h d_v=d_{\text{model}}.
$$

Different heads can learn different types of interactions, such as syntactic, semantic, positional, or task-dependent relationships.

---

# 15. From Token Representations to Classification

The Transformer encoder returns one contextualized vector for every token:

$$
Z =
[\mathbf{z}_1,\dots,\mathbf{z}_T].
$$

Sequence-level classification requires converting this sequence into a single global representation.

## Average Pooling

One possibility is

$$
\mathbf{z}_{\text{avg}} =
\frac{1}{T}
\sum_{t=1}^{T}
\mathbf{z}_t.
$$

The pooled vector is then passed to a classifier such as an MLP.

## Classification Token

Another strategy prepends a special token, commonly written `[CLS]`, to the input.

After the Transformer layers, its contextualized embedding

$$
\mathbf{z}_{[\mathrm{CLS}]}
$$

is treated as a learned global sequence representation and passed to the classification head.

---

# 16. TCNs and Transformers

TCNs and Transformers both model sequential dependencies but rely on different architectural assumptions.

| Property | TCN | Transformer |
|---|---|---|
| Main operation | Convolution | Attention |
| Dependency structure | Local, progressively expanded | Direct pairwise interactions |
| Receptive field | Determined by kernel, dilation, and depth | Global within an attention layer |
| Inductive bias | Strong temporal locality | Weaker locality bias |
| Parallelism | High | High |
| Long-range dependencies | Captured through depth/dilation | Directly accessible through attention |
| Causality | Built through causal convolution | Requires attention masking when needed |

TCNs encode a structured temporal prior into the architecture, whereas Transformers learn more of the interaction structure directly from data.

---

# 17. Regression Loss Functions

## 17.1 Mean Squared Error

For targets $y_t$ and predictions $\hat y_t$,

$$
\mathcal{L}_{\mathrm{MSE}} =
\frac{1}{N}
\sum_{t=1}^{N}
(y_t-\hat y_t)^2.
$$

Because errors are squared, large errors receive disproportionately high penalties.

Consequently, MSE is:

- smooth and differentiable;
- convenient for gradient-based optimization;
- sensitive to outliers.

Minimizing expected squared error leads to estimation of the **conditional mean**.

---

## 17.2 Mean Absolute Error

The Mean Absolute Error is

$$
\mathcal{L}_{\mathrm{MAE}} =
\frac{1}{N}
\sum_{t=1}^{N}
|y_t-\hat y_t|.
$$

MAE grows linearly rather than quadratically with error magnitude, making it more robust to outliers.

It is not differentiable exactly at zero error, although subgradients can be used without major difficulty in modern optimization frameworks.

Minimizing expected absolute error leads to estimation of the **conditional median**.

---

# 18. Regression Through Binning

A continuous regression target can be discretized into $K$ intervals called **bins**.

For example, for age prediction with

$$
y\in[0,100],
$$

one might define intervals such as

$$
[0,10),[10,20),\dots,[90,100].
$$

The original regression problem then becomes a $K$-class classification problem.

A classifier produces probabilities

$$
p_1,\dots,p_K,
$$

where

$$
\sum_{k=1}^{K}p_k=1.
$$

A continuous estimate may be obtained using the center $c_k$ of the most probable bin, or through the probability-weighted average

$$
\hat y =
\sum_{k=1}^{K}p_kc_k.
$$

This formulation permits the use of classification losses such as cross-entropy and also provides information about predictive uncertainty through the distribution of probability mass across bins.

Its main limitation is discretization error: very coarse bins reduce the precision of the continuous prediction.

---

# 19. Limitations of MSE and MAE

Both standard MSE and MAE are **symmetric losses**.

For a fixed absolute error,

$$
|y-\hat y|,
$$

overprediction and underprediction receive the same penalty.

This assumption may be undesirable when errors have asymmetric costs.

Examples include:

- energy-demand forecasting;
- risk assessment;
- financial forecasting;
- inventory management.

In addition, a point estimate such as a mean or median does not directly describe the range of possible outcomes.

These limitations motivate **quantile regression**.

---

# 20. Quantiles

For a random variable $Y$, a $\tau$-quantile $q_\tau$ is a value satisfying, in the continuous case,

$$
P(Y\leq q_\tau)=\tau,
$$

with

$$
\tau\in(0,1).
$$

Examples:

$$
\tau=0.5
$$

corresponds to the median,

$$
\tau=0.9
$$

corresponds to the 90th percentile, and

$$
\tau=0.1
$$

corresponds to the 10th percentile.

Thus, the 90th percentile is a value such that approximately $90\%$ of observations are at or below it and $10\%$ are above it.

## Median vs Mode

The **median** divides the probability mass into two halves.

The **mode** is the most probable value, or the location where the probability density reaches its maximum in a continuous distribution.

They need not coincide.

For a symmetric unimodal Gaussian distribution,

$$
\text{mean} =
\text{median} =
\text{mode}.
$$

---

# 21. Quantile Loss

To estimate a specific conditional quantile rather than the mean, quantile regression uses the **quantile loss**, also called **pinball loss**.

For target $y$, prediction $\hat y$, and quantile level $\tau\in(0,1)$,

$$
\mathcal{L}_\tau(y,\hat y) =
\begin{cases}
\tau(y-\hat y), & y\geq \hat y,\[4pt]
(1-\tau)(\hat y-y), & y<\hat y.
\end{cases}
$$

Equivalently, defining the residual

$$
u=y-\hat y,
$$

the loss is

$$
\rho_\tau(u) =
u
\left(
\tau-\mathbf{1}_{\{u<0\}}
\right).
$$

The loss is asymmetric whenever

$$
\tau\neq0.5.
$$

---

## 21.1 Case $\tau=0.5$

For

$$
\tau=0.5,
$$

the loss becomes

$$
\mathcal{L}_{0.5}(y,\hat y) =
\frac{1}{2}|y-\hat y|.
$$

Thus, it is proportional to MAE.

The factor $1/2$ does not change the minimizer, so optimizing the $\tau=0.5$ quantile loss estimates the median just as minimizing MAE does.

---

## 21.2 Case $\tau=0.9$

Suppose

$$
\tau=0.9.
$$

If the model **underpredicts**,

$$
\hat y<y,
$$

the loss is

$$
0.9(y-\hat y).
$$

If the model **overpredicts**,

$$
\hat y>y,
$$

the loss is

$$
0.1(\hat y-y).
$$

Therefore, for equal absolute error magnitude, underprediction is penalized

$$
\frac{0.9}{0.1}=9
$$

times more strongly than overprediction.

The optimization therefore favors larger predictions until approximately $90\%$ of the conditional target distribution lies below the prediction.

Hence, the model learns the **90th conditional percentile**.

---

## 21.3 Case $\tau=0.1$

For

$$
\tau=0.1,
$$

underprediction receives weight $0.1$, while overprediction receives weight $0.9$.

The model therefore learns a relatively low prediction corresponding to the **10th conditional percentile**.

---

# 22. Prediction Intervals with Quantile Regression

Training separate outputs for multiple quantiles can provide an estimate of predictive uncertainty.

For example, predicting

$$
q_{0.1}(x)
$$

and

$$
q_{0.9}(x)
$$

gives an interval

$$
[q_{0.1}(x),q_{0.9}(x)].
$$

Under correct calibration, approximately $80\%$ of the conditional target distribution lies between these two quantiles because

$$
0.9-0.1=0.8.
$$

Unlike MSE regression, quantile regression does not require the target distribution to be Gaussian.

---

# References

Bai, S., Kolter, J. Z., and Koltun, V. *An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling*. ICML, 2018.

Deng, S., Zhang, N., Zhang, W., Chen, J., Pan, J. Z., and Chen, H. *Knowledge-Driven Stock Trend Prediction and Explanation via Temporal Convolutional Network*. WWW Companion, 2019.

Selvaraju, R. R. et al. *Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization*. ICCV, 2017.

Vaswani, A. et al. *Attention Is All You Need*. NeurIPS, 2017.