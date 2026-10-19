# I-JEPA: A Self-Contained Guide to the Method and Its PyTorch Implementation

*From the original paper to a practical ViT-Tiny implementation*

**Primary sources**

- [Paper — *Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture* (Assran et al., CVPR 2023)](https://arxiv.org/pdf/2301.08243)
- [Official PyTorch implementation — `facebookresearch/ijepa`](https://github.com/facebookresearch/ijepa)

This guide explains the theoretical foundations, architecture, training objective, and implementation of **Image-based Joint-Embedding Predictive Architecture (I-JEPA)**.

It assumes familiarity with the general JEPA framework, self-supervised learning, and basic neural network training. Some understanding of Vision Transformers (ViTs) is useful, but their internal operation is explained where necessary.

The emphasis is on **what each component computes, how information flows through the architecture, and how those computations translate into PyTorch**.

We use **ViT-Tiny** as a running implementation example. Although ViT-Tiny is defined in the official repository, the paper's reported experiments primarily use larger architectures. Our Tiny configuration is therefore an educational adaptation, not a reproduction of an officially reported ViT-Tiny experiment.

Distributed and parallel training details are intentionally excluded.

---

## Part I — The I-JEPA Method

## 1. The central idea of I-JEPA

I-JEPA is a non-generative, self-supervised learning method designed to learn semantic image representations through prediction in latent space.

Given an image, the model:

1. Observes a subset of its patches, called the **context**.
2. Encodes those patches into contextualized representations.
3. Predicts representations corresponding to several missing image regions, called **target blocks**.
4. Compares those predictions against representations computed by a slowly evolving target encoder.
5. Updates the context encoder and predictor to improve their predictions.

The essential difference from traditional masked image modeling is the prediction target.

A Masked Autoencoder (MAE), for example, attempts to reconstruct missing image patches in pixel space:

$$
\hat{x}_{\mathrm{missing}} \approx x_{\mathrm{missing}}.
$$

I-JEPA instead predicts their learned representations:

$$
\boxed{\hat{s}_{\mathrm{missing}} \approx s_{\mathrm{missing}}.}
$$

These representations are produced by a neural network rather than being defined directly from pixels.

### 1.1 Why predict representations instead of pixels?

Pixel-level reconstruction requires a model to reproduce visual details such as textures, colors, and local appearance. Many of these details are unpredictable from the visible context and are not necessarily useful for semantic understanding.

Consider an image of a dog whose head is hidden. The visible body might provide enough information to infer that the missing region contains a dog's head. However, its exact fur texture, illumination, and individual pixel values might be impossible to determine.

A pixel-reconstruction objective penalizes the model for failing to recover these details. A representation-prediction objective can instead encourage it to predict higher-level information associated with the missing region.

The underlying hypothesis is that **predicting appropriately constructed latent targets encourages the model to learn semantic structure rather than unnecessary low-level detail**.

This is not guaranteed merely by operating in latent space. The architecture, target encoder, masking strategy, and training dynamics all contribute to the representations that emerge.

### 1.2 What distinguishes I-JEPA from a generic JEPA?

I-JEPA makes several concrete architectural choices:

| Component | I-JEPA implementation |
|---|---|
| Input modality | Images |
| Basic representation units | Image patches |
| Context encoder | Vision Transformer |
| Target encoder | Momentum-updated copy of the context encoder |
| Predictor | A separate Vision Transformer |
| Context selection | One large spatial block, excluding target regions |
| Prediction targets | Multiple relatively large image blocks |
| Prediction space | Learned patch embeddings |
| Positional conditioning | Position embeddings and learned mask tokens |
| Training loss | Regression between predicted and target embeddings |
| Collapse mitigation | Stop-gradient target branch, EMA updates, and architectural asymmetry |

Two design decisions are especially important:

1. **The target encoder processes the complete image before target representations are selected.** Its outputs can therefore incorporate surrounding-image information.
2. **The masking strategy uses several relatively large targets and spatially distributed context.** This encourages predictions that depend on meaningful relationships among image regions.

Both are central to the method rather than incidental implementation details.

## 2. Mathematical formulation

Let an input image be

$$
I \in \mathbb{R}^{3\times H\times W}.
$$

Divide it into non-overlapping square patches of size $P\times P$. Assuming $H$ and $W$ are divisible by $P$, the number of patches is

$$
N=\frac{H}{P}\frac{W}{P}.
$$

For a $224\times224$ image with patch size $16$:

$$
N=\frac{224}{16}\frac{224}{16}=196,
$$

arranged in a $14\times14$ grid.

I-JEPA contains three networks.

### 2.1 Context encoder

The context encoder is $f_\theta$. It receives only the visible context patches.

Let $C\subseteq\{1,\ldots,N\}$ denote their indices. The context representations are

$$
S_C=f_\theta(I;C).
$$

Here, $f_\theta(I;C)$ means that the image is converted into patch embeddings, only embeddings indexed by $C$ are retained, and those visible tokens are processed by the Transformer.

If $K=|C|$ and the embedding dimension is $D$, then

$$
S_C\in\mathbb{R}^{K\times D}.
$$

The context encoder is updated using gradient descent.

### 2.2 Target encoder

The target encoder is $f_{\bar\theta}$. It has the same architecture as the context encoder, but different parameters. Unlike the context encoder, it receives the **complete image**:

$$
S=f_{\bar\theta}(I),\qquad S\in\mathbb{R}^{N\times D}.
$$

Every output token corresponds to a spatial patch location. Because a ViT uses self-attention, these outputs are not independent encodings of individual patches: each token can incorporate information from elsewhere in the image.

Suppose we sample $M$ target blocks:

$$
T_1,T_2,\ldots,T_M,
$$

where each $T_i$ is a set of patch indices. The corresponding target representations are

$$
Y_i=S[T_i].
$$

**The ordering is essential:**

```text
Full image -> Target encoder -> Select target embeddings
```

We do **not** isolate the target image regions first and encode each one independently.

### 2.3 Predictor

The predictor is $g_\phi$. It receives the context representations and positional information identifying the target locations.

For a target block $T_i$, its prediction is

$$
\hat{Y}_i=g_\phi(S_C,T_i).
$$

More precisely, indices $T_i$ determine the position-conditioned mask tokens provided to the predictor.

If target $i$ contains $L_i$ patches, then

$$
\hat{Y}_i\in\mathbb{R}^{L_i\times D}.
$$

The predictor produces **one embedding per target patch**. It does not directly output RGB values, a reconstructed image, or a single vector for an entire block.

### 2.4 Training objective

The original paper formulates the objective using squared Euclidean distance:

$$
\boxed{
\mathcal{L}_{\mathrm{paper}}
=\frac{1}{M}\sum_{i=1}^{M}\sum_{j\in T_i}
\left\|\hat{s}_j-s_j\right\|_2^2.
}
$$

Here, $s_j$ is the target-encoder representation for patch $j$, $\hat{s}_j$ is its predicted representation, and $M$ is the number of target blocks.

The context encoder and predictor minimize this loss. The target encoder does not receive gradients from the objective: its parameters are updated through an **exponential moving average (EMA)**:

$$
\boxed{
\bar\theta_{t+1}
=m_t\bar\theta_t+(1-m_t)\theta_{t+1}.
}
$$

The momentum $m_t$ is close to one, so the target encoder evolves more slowly than the context encoder.

**Paper/code distinction:** the official PyTorch training loop uses `torch.nn.functional.smooth_l1_loss`, not the exact squared L2 expression shown in the paper. We explain this in Section 9.

## 3. Multi-block masking: the most important design choice

I-JEPA uses structured masking rather than independently masking arbitrary image patches. It constructs a prediction task with:

- One large context block.
- Multiple relatively large target blocks.
- No overlap between the intended final context and targets.

The aim is to make predictions difficult enough to require meaningful visual understanding while preserving enough information for useful predictions.

### 3.1 Target block sampling

I-JEPA typically samples four target blocks per image:

$$
M=4.
$$

Each target is a rectangle in the patch grid.

| Parameter | Typical value |
|---|---|
| Number of target blocks | 4 |
| Target area scale | $[0.15,0.20]$ |
| Target aspect ratio | $[0.75,1.5]$ |
| Target-target overlap | Allowed |

Target area scale is a fraction of the total patch-grid area. For a $14\times14$ grid and scale $0.18$:

$$
196(0.18)\approx35
$$

patches. A roughly square target might therefore be a $6\times6$ region, depending on sampled aspect ratio and integer rounding.

These are **spatially contiguous rectangles**, not arbitrarily selected individual patches. Large blocks tend to require understanding visual structures rather than only local continuity. The different targets are allowed to overlap.

### 3.2 Context block sampling

The context is constructed differently:

| Parameter | Typical value |
|---|---|
| Number of context blocks | 1 |
| Context area scale | $[0.85,1.0]$ |
| Context aspect ratio | 1 |
| Context-target overlap | Disallowed in the intended sampling procedure |

Initially, the sampler selects a large, approximately square region $C_0$. It then removes target positions:

$$
\boxed{
C=C_0\setminus\bigcup_{i=1}^M T_i.
}
$$

Thus, under the intended non-overlapping sampling rule,

$$
C\cap T_i=\varnothing\qquad\forall i.
$$

The **final visible context need not be a rectangle**. It can consist of spatially separated areas around the hidden targets. The repository includes a fallback for repeated failed context samples that can relax exclusions; see Section 8.3.

### 3.3 Why large targets?

Tiny missing regions might be predictable from nearby texture, edges, or simple local statistics. Larger targets make such shortcuts less effective: the model may need to exploit object identity, shape, and relationships among distant regions.

A missing dog's head, for example, may be predictable in representation space from the animal's visible body without its exact pixel appearance being predictable.

This is an **inductive bias toward semantic prediction**, not a guarantee that every learned feature is semantic.

### 3.4 Why a large context?

A very small context can make the missing content excessively ambiguous. A large visible region gives evidence about objects, scene structure, and spatial relationships.

I-JEPA balances two requirements:

$$
\text{Large targets}\quad\Longrightarrow\quad\text{non-trivial predictions},
$$

$$
\text{Large distributed context}\quad\Longrightarrow\quad\text{informative predictions}.
$$

The model should not be able to copy the answer, but should have enough information to make the prediction task meaningful.

### 3.5 Evidence from the paper

The paper reports ablations demonstrating the importance of masking. For a ViT-B/16 trained for 300 epochs and evaluated on the ImageNet-1% low-shot benchmark:

| Target blocks | Reported top-1 accuracy |
|---|---:|
| 1 | 9.0% |
| 2 | 22.0% |
| 3 | 48.5% |
| 4 | 54.2% |

The paper also compares how target representations are constructed, using a ViT-H/16 and the ImageNet-1% benchmark:

| Target construction | Reported top-1 accuracy |
|---|---:|
| Mask **after** target encoding | 67.3% |
| Mask **before** target encoding | 56.1% |

These experiments emphasize that contextualized targets computed from the full image are more useful than features obtained by encoding isolated masked regions. The results are from larger architectures, **not** our ViT-Tiny adaptation.

Source: [original paper](https://arxiv.org/pdf/2301.08243).

## 4. How the three networks interact

### 4.1 Target branch

The target encoder sees the complete image:

```text
                 COMPLETE IMAGE
                       |
                       v
                Patch embedding
                       |
                       v
             Positional embeddings
                       |
                       v
                  Target ViT
                       |
                       v
              N patch representations
                       |
             +---------+---------+
             |         |         |
             v         v         v
          Target 1  Target 2     ...
```

No mask tokens are inserted into the target encoder. The network processes the full sequence, then the selected target-position embeddings are extracted. One full-image encoder pass provides representations for all target blocks.

### 4.2 Context branch

```text
                 COMPLETE IMAGE
                       |
                       v
                Patch embedding
                       |
                       v
             Positional embeddings
                       |
                       v
              Select visible tokens
                       |
                       v
                  Context ViT
                       |
                       v
                Context features
```

Missing target tokens are **removed before Transformer blocks**. They are not represented by zeros or mask tokens inside the context encoder.

With $K$ visible tokens rather than $N$ total patches, the self-attention matrix has size $K\times K$ instead of $N\times N$. The attention component's quadratic token-complexity decreases from $O(N^2)$ to $O(K^2)$.

Patch embeddings are still computed from the image before token selection; it is the more expensive Transformer computation that operates only over the selected context tokens.

### 4.3 Predictor branch

For each target block, the predictor receives the context encoder output and **one mask token per target patch**, with embeddings identifying the original patch positions.

```text
               Context encoder output
                         |
                         v
                Linear projection
                         |
                         v
                Context embeddings
                         |
                         |
  Target positions       |
         |               |
         v               |
  Position embeddings    |
         +               |
  Learned mask token     |
         |               |
         v               |
   Target query tokens   |
         |               |
         +-------+-------+
                 |
                 v
   Concatenate context + target tokens
                 |
                 v
            Predictor ViT
                 |
                 v
       Keep target token outputs
                 |
                 v
     Project to encoder dimension
                 |
                 v
       Predicted target features
                 |
                 v
     Compare with target encoder
```

The predictor **does not see the hidden target pixels**. It knows *where* to predict, but must infer *what* to predict from context.

### 4.4 Why a Transformer predictor?

For $K$ context tokens and $L$ target mask tokens, the predictor receives a sequence of length $K+L$. Its self-attention allows target tokens to attend to context tokens, and target tokens to interact with each other. After processing, it discards outputs associated with the context and retains the $L$ target predictions.

The predictor is conceptually evaluated independently for each target block, though the official code batches these evaluations together.

### 4.5 Why positional embeddings?

Without positions, the predictor cannot reliably distinguish whether it should predict a missing region at the top or bottom of the image.

For target position $j$, the input query token is

$$
q_j=m+p_j,
$$

where $m\in\mathbb{R}^{D_p}$ is the **shared trainable mask token**, and $p_j\in\mathbb{R}^{D_p}$ is the **positional embedding** for patch $j$.

The mask token is shared across positions, while $p_j$ identifies which position is being predicted. The predictor thus learns a conditional spatial prediction:

> Given the available context, what representation should correspond to this spatial location?

---

## Part II — Understanding the Official Implementation

## 5. Repository structure

The official repository is organized around a few important files:

| File | Responsibility |
|---|---|
| [`src/models/vision_transformer.py`](https://github.com/facebookresearch/ijepa/blob/main/src/models/vision_transformer.py) | Vision Transformer backbone, predictor, attention and blocks |
| [`src/masks/multiblock.py`](https://github.com/facebookresearch/ijepa/blob/main/src/masks/multiblock.py) | Context and target mask sampling |
| [`src/masks/utils.py`](https://github.com/facebookresearch/ijepa/blob/main/src/masks/utils.py) | Token selection from patch indices |
| [`src/helper.py`](https://github.com/facebookresearch/ijepa/blob/main/src/helper.py) | Model/optimizer initialization and learning-rate scheduler |
| [`src/train.py`](https://github.com/facebookresearch/ijepa/blob/main/src/train.py) | Forward passes, loss, gradient update and EMA |
| [`src/transforms.py`](https://github.com/facebookresearch/ijepa/blob/main/src/transforms.py) | Image preprocessing |
| [`configs/in1k_vith14_ep300.yaml`](https://github.com/facebookresearch/ijepa/blob/main/configs/in1k_vith14_ep300.yaml) | Reference training recipe |

Suggested reading order: **`vision_transformer.py` → `multiblock.py` → `train.py` → `helper.py`**.

## 6. ViT-Tiny: understanding the backbone

The repository provides several Vision Transformer configurations, including `vit_tiny`. Its definition is:

```python
def vit_tiny(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model
```

Source: [`vision_transformer.py`](https://github.com/facebookresearch/ijepa/blob/main/src/models/vision_transformer.py).

The architecture is:

| Hyperparameter | ViT-Tiny |
|---|---|
| Input resolution in this example | $224\times224$ |
| Patch size | $16\times16$ |
| Number of patches | 196 |
| Embedding dimension | 192 |
| Transformer blocks | 12 |
| Attention heads | 3 |
| Dimension per head | 64 |
| MLP expansion ratio | 4 |
| MLP hidden dimension | 768 |
| Positional embeddings | Fixed 2D sinusoidal |
| CLS token | None |

The key architectural difference between ViT-Tiny and larger I-JEPA backbones is primarily model capacity. The JEPA training mechanism is the same.

### 6.1 Patch embedding

For a batch of images:

```python
images.shape = [B, 3, 224, 224]
```

The repository implements patch embedding with a convolution:

```python
self.proj = nn.Conv2d(
    in_chans,
    embed_dim,
    kernel_size=patch_size,
    stride=patch_size
)
```

For ViT-Tiny this is equivalent to:

```python
nn.Conv2d(3, 192, kernel_size=16, stride=16)
```

Each $16\times16$ RGB patch contains

$$
16\times16\times3=768
$$

pixel values, and the convolution learns a projection

$$
\mathbb{R}^{768}\longrightarrow\mathbb{R}^{192}.
$$

The convolution produces:

```python
[B, 192, 14, 14]
```

The output is flattened and transposed:

```python
x = self.proj(x).flatten(2).transpose(1, 2)
```

resulting in:

```python
[B, 196, 192]
```

Each of the 196 tokens now represents one image patch.

### 6.2 Positional embeddings

Plain self-attention has no inherent representation of spatial order, so I-JEPA adds positional embeddings to the patch tokens.

Let

$$
E\in\mathbb{R}^{196\times192}
$$

be the positional embedding matrix. For patch $j$:

$$
z_j=e_j+p_j.
$$

The repository uses **fixed 2D sine-cosine positional embeddings**, not learned encoder positions:

```python
self.pos_embed = nn.Parameter(
    torch.zeros(1, num_patches, embed_dim),
    requires_grad=False
)
```

The values are filled with deterministic sinusoidal encodings of 2D patch-grid coordinates. The resulting tensor still has shape `[B, 196, 192]`.

### 6.3 Masking happens before the Transformer blocks

The important order in `VisionTransformer.forward()` is:

```python
x = self.patch_embed(x)

pos_embed = self.interpolate_pos_encoding(
    x, self.pos_embed
)

x = x + pos_embed

if masks is not None:
    x = apply_masks(x, masks)

for blk in self.blocks:
    x = blk(x)
```

In short:

$$
\boxed{
\text{Patch embedding}
\rightarrow\text{Add positions}
\rightarrow\text{Select context}
\rightarrow\text{Transformer}
}
$$

If the context retains $K=85$ patches, the tensor becomes `[B, 85, 192]` **before entering the Transformer**.

Consequently, the context encoder cannot attend to missing target tokens, and visible tokens retain their original spatial positions because positions were added before selection.

### 6.4 Inside a ViT-Tiny Transformer block

Each block contains multi-head self-attention and a feed-forward MLP in a pre-normalization residual architecture:

$$
U=X+\operatorname{MHA}(\operatorname{LN}(X)),
$$

$$
\boxed{X_{\mathrm{out}}=U+\operatorname{MLP}(\operatorname{LN}(U)).}
$$

**Multi-head attention.** With $D=192$ and $H=3$ heads, the head dimension is

$$
d_h=\frac{D}{H}=64.
$$

For an input $X\in\mathbb{R}^{B\times K\times192}$, queries, keys, and values are computed as

$$
Q=XW_Q,\qquad K_{\mathrm{attn}}=XW_K,\qquad V=XW_V.
$$

The implementation combines the three projections:

```python
self.qkv = nn.Linear(dim, dim * 3)
```

which is `nn.Linear(192, 576)` for ViT-Tiny. The projections are rearranged into three heads. Each head computes

$$
A_h=\operatorname{softmax}\left(\frac{Q_hK_h^\top}{\sqrt{64}}\right),
$$

$$
O_h=A_hV_h.
$$

The three head outputs are concatenated and projected back to 192 dimensions. If $K=85$, the per-head attention matrix is $85\times85$ and the attention tensor has shape `[B, 3, 85, 85]`.

**Feed-forward MLP.** Expansion ratio 4 gives

$$
192\rightarrow768\rightarrow192,
$$

with GELU activation. The MLP processes each token independently; attention is what allows tokens to exchange information.

**Depth and normalization.** ViT-Tiny applies 12 such blocks, followed by a final LayerNorm. The context encoder returns

$$
S_C\in\mathbb{R}^{B\times K\times192}.
$$

### 6.5 There is no CLS token

The I-JEPA encoder produces patch-level representations directly, without a classification token. The pretraining loss is also patch-level.

For downstream classification, patch representations can be average-pooled into a global image embedding. The paper's evaluation may also aggregate representations from multiple late Transformer layers.

## 7. The I-JEPA predictor: detailed implementation walkthrough

The predictor is the `VisionTransformerPredictor` class in `src/models/vision_transformer.py`. It has its own width $D_p$ and consists of:

```text
Context features
       |
       v
Linear projection: D -> Dp
       |
       v
Add context position embeddings
       |
       v
Append position-conditioned mask tokens
       |
       v
Transformer blocks
       |
       v
Final LayerNorm
       |
       v
Select target-token outputs
       |
       v
Linear projection: Dp -> D
       |
       v
Predicted target features
```

The two projections let the predictor operate in a feature space with a different dimension from the encoder.

### 7.1 Choosing a predictor for ViT-Tiny

The paper uses a predictor width of 384 for larger backbones. But the Tiny encoder has width 192; copying a predictor width of 384 gives $D_p>D$. This is valid, but not a narrow predictor relative to the backbone.

For an educational lightweight Tiny implementation, choose:

| Parameter | Proposed value |
|---|---:|
| Encoder embedding dimension $D$ | 192 |
| Predictor embedding dimension $D_p$ | 96 |
| Predictor depth | 6 |
| Predictor heads | 3 |
| Predictor MLP hidden dimension | 384 |

**These are proposed illustrative settings, not an officially validated ViT-Tiny recipe.** They give

$$
\frac{D_p}{H_p}=\frac{96}{3}=32
$$

features per predictor attention head.

The predictor can be instantiated as:

```python
predictor = VisionTransformerPredictor(
    num_patches=196,
    embed_dim=192,
    predictor_embed_dim=96,
    depth=6,
    num_heads=3
)
```

The repository's `init_model()` helper and `vit_predictor` factory can also build it.

### 7.2 Step 1: project context embeddings

The predictor receives:

```python
x.shape = [B, K, 192]
```

and applies:

```python
self.predictor_embed = nn.Linear(
    embed_dim,
    predictor_embed_dim
)
```

This maps $192\rightarrow96$, giving `[B, K, 96]`. The predictor adds its own positional embeddings for the visible context positions. These positions are separate from the encoder's positional embeddings, and are expressed in the predictor feature space.

### 7.3 Step 2: construct target mask tokens

For each target position $j$, the predictor constructs

$$
q_j=m+p_j,
$$

where $m$ is the trainable mask token and $p_j$ is the target-position embedding.

The repository initializes a parameter of shape `[1, 1, Dp]`:

```python
self.mask_token = nn.Parameter(
    torch.zeros(1, 1, predictor_embed_dim)
)
```

For a block containing $L=36$ target patches, the target tokens have shape `[B, 36, 96]`. All positions share the learned $m$ but have different positional embeddings.

### 7.4 Step 3: concatenate context and targets

The predictor input is

$$
X_{\mathrm{pred}}=\operatorname{Concat}(S_C',Q_T).
$$

For $K=85$ and $L=36$:

```python
context.shape       # [B, 85, 96]
target_tokens.shape # [B, 36, 96]
combined.shape      # [B, 121, 96]
```

The target part contains **mask tokens**, not image-derived target features.

### 7.5 Step 4: run the predictor Transformer

Six Transformer blocks process the full `[B, 121, 96]` sequence. Self-attention permits context-context, context-target, and target-target interactions.

This is more expressive than predicting each missing embedding with an independent MLP. After the final block and LayerNorm, the tensor still has shape `[B, 121, 96]`.

### 7.6 Step 5: keep target predictions only

The predictor discards the first $K$ outputs associated with context tokens:

```python
x = x[:, N_ctxt:]
```

The remaining tensor has shape `[B, 36, 96]`.

### 7.7 Step 6: project back to encoder dimension

Predictions must match target features, whose dimension is $D=192$. The final predictor projection is:

```python
self.predictor_proj = nn.Linear(
    predictor_embed_dim,
    embed_dim
)
```

This maps $96\rightarrow192$ and produces `[B, 36, 192]`.

### 7.8 How multiple targets are processed

The predictor does **not** put all four target blocks into a single giant target sequence. It solves four prediction tasks conditioned on the same context:

$$
\hat Y_1=g_\phi(S_C,T_1),\quad
\hat Y_2=g_\phi(S_C,T_2),\quad
\hat Y_3=g_\phi(S_C,T_3),\quad
\hat Y_4=g_\phi(S_C,T_4).
$$

The parameters are shared. The repository repeats context embeddings across target blocks and batches these computations for efficiency.

For a batch of $B$ images, $M=4$ blocks and common target length $L$, the predictor returns

$$
\boxed{\hat Y\in\mathbb{R}^{(4B)\times L\times192}.}
$$

Conceptually, these are the same predictions as a tensor of shape `[B, 4, L, 192]`; the official implementation flattens the image and target dimensions.

Source: [`vision_transformer.py`](https://github.com/facebookresearch/ijepa/blob/main/src/models/vision_transformer.py).

## 8. How masking is implemented in PyTorch

The primary class is `MaskCollator` in `src/masks/multiblock.py`.

### 8.1 Masks are patch indices

A mask in this code is a tensor of the **indices of patches to retain**, rather than primarily a binary image.

For example, on a $4\times4$ patch grid:

```text
 0   1   2   3
 4  [5] [6]  7
 8  [9] [10] 11
12  13  14  15
```

The central selected region is represented by:

```python
mask = torch.tensor([5, 6, 9, 10])
```

The mask is used to **gather selected embeddings**, not to overwrite image pixels with zeros.

### 8.2 The `apply_masks` function

The essential mechanism is `torch.gather` along the token dimension. The following is a simplified equivalent of the repository's helper:

```python
def apply_masks(x, masks):
    results = []

    for mask in masks:
        indices = mask.unsqueeze(-1)
        indices = indices.expand(-1, -1, x.size(-1))
        results.append(torch.gather(x, 1, indices))

    return torch.cat(results, dim=0)
```

Given `x.shape == [B, 196, 192]` and `mask.shape == [B, 85]`, this returns `[B, 85, 192]`.

With four target masks each retaining $L=36$ positions, concatenation along the batch dimension produces `[4*B, 36, 192]`.

The same helper extracts target-encoder output embeddings at the required patch positions.

### 8.3 How `MaskCollator` constructs masks

**Target sizes.** A sampled target scale $s_t$ and aspect ratio $r=h/w$ determine approximate area and dimensions. On an $H_g\times W_g$ patch grid:

$$
A_t=s_tH_gW_g,
$$

$$
h\approx\sqrt{A_t r},\qquad
w\approx\sqrt{\frac{A_t}{r}}.
$$

Dimensions are rounded to integer patch units and constrained by grid size.

**Target locations.** The collator samples four target rectangles. Overlap among these targets is allowed.

**Context selection.** It samples a larger context rectangle, then uses the complements of the target masks to remove target positions from the context.

**Return values.** In the standard training pipeline, the batch collator returns:

```python
images, masks_enc, masks_pred
```

where:

```python
len(masks_enc)  == 1
len(masks_pred) == 4
```

Each element of a mask list is a batch of patch-index vectors.

**Common-length truncation.** Different sampled contexts and targets can contain different numbers of patches. To batch the computations without variable-length padding, the collator determines the minimum selected length and truncates index sequences to a common size. If three sampled context masks have lengths 90, 85 and 87, they can all be truncated to 85. Targets are treated similarly. Final lengths may therefore differ slightly from the sampled rectangle areas.

**Sampling fallback.** If repeated attempts to find a sufficiently large context fail, the sampler may relax exclusions. Thus, context/target non-overlap is the intended construction, but the code does not guarantee it absolutely in every fallback case.

Source: [`multiblock.py`](https://github.com/facebookresearch/ijepa/blob/main/src/masks/multiblock.py).

---

## Part III — The Complete Training Procedure

## 9. A full training iteration

Consider a batch

$$
I\in\mathbb{R}^{B\times3\times224\times224}
$$

with Tiny encoder dimension $D=192$ and four target blocks. For tensor examples, suppose the collator retains $K=85$ context patches and $L=36$ target patches per block. These are **illustrative, not fixed I-JEPA settings**.

### Step 1 — Sample masks

The collator generates one context mask and four target masks per image. Let $C_b$ denote context indices for image $b$ and $T_{b,i}$ denote its target block $i$.

### Step 2 — Compute target representations

Run the complete image through the target encoder:

$$
S=f_{\bar\theta}(I).
$$

Its output has shape `[B, 196, 192]`.

The training implementation additionally applies feature-wise LayerNorm:

```python
h = target_encoder(images)
h = F.layer_norm(h, (h.size(-1),))
```

This normalizes over each token's embedding dimension, not over the whole batch. Then extract target embeddings:

```python
h = apply_masks(h, masks_pred)
```

The result has shape `[4*B, 36, 192]`. All target-branch operations are under `torch.no_grad()`, so target features are fixed for the current optimization step.

### Step 3 — Compute context representations

```python
z = encoder(images, masks_enc)
```

The context encoder returns `[B, 85, 192]`. Unlike the target branch, this pass retains its computation graph for gradient updates.

### Step 4 — Predict target representations

```python
z = predictor(z, masks_enc, masks_pred)
```

The predictor inserts position-conditioned mask tokens, runs its Transformer, and returns `[4*B, 36, 192]`. Predicted and target tensors are aligned by image, block and patch position.

### Step 5 — Compute the prediction loss

The **official implementation** uses:

```python
loss = F.smooth_l1_loss(z, h)
```

With default $\beta=1$, PyTorch's Smooth L1 loss for a scalar error $e=\hat s-s$ is

$$
\ell(e)=
\begin{cases}
\tfrac12e^2, & |e|<1,\\[4pt]
|e|-\tfrac12, & |e|\geq1.
\end{cases}
$$

The final loss averages over elements. Relative to squared L2 in the paper, larger errors receive **linear instead of quadratic** penalties, and the PyTorch reduction averages over batch, token and feature dimensions.

Both train the model to match latent target representations.

### Step 6 — Update context encoder and predictor

Only $\theta$ and $\phi$ are optimized by backpropagation:

$$
\theta_{t+1}=\theta_t-\eta\nabla_\theta\mathcal L,
$$

$$
\phi_{t+1}=\phi_t-\eta\nabla_\phi\mathcal L.
$$

These simple SGD-style expressions illustrate which parameters receive gradients; the actual optimizer is **AdamW**. Gradients flow through the predictor into the context encoder, but do not flow through the target encoder.

### Step 7 — Update target encoder using EMA

After the optimizer step:

$$
\bar\theta_{t+1}=m_t\bar\theta_t+(1-m_t)\theta_{t+1}.
$$

With $m_t=0.996$:

$$
\bar\theta_{t+1}=0.996\bar\theta_t+0.004\theta_{t+1}.
$$

The momentum schedule approaches 1 as training progresses, so the target encoder changes progressively more slowly.

### 9.1 Why not train both encoders using the same gradient loss?

Two unconstrained networks minimizing representation similarity can potentially find degenerate solutions, such as constant representations. I-JEPA introduces asymmetry: the context encoder learns by gradient descent, the target encoder supplies stop-gradient targets and follows by EMA, and a separate predictor performs the conditional mapping.

These components empirically help stable training, but do **not** constitute a general mathematical proof that collapse is impossible.

## 10. Tensor shapes: a complete ViT-Tiny example

Assume:

$$
B=2,\quad N=196,\quad K=85,\quad L=36,
$$

$$
M=4,\quad D=192,\quad D_p=96.
$$

The particular values $K=85$ and $L=36$ are illustrative.

| Operation | Tensor shape |
|---|---|
| Input images | `[2, 3, 224, 224]` |
| Patch embeddings | `[2, 196, 192]` |
| Full target encoder output | `[2, 196, 192]` |
| Selected target embeddings | `[8, 36, 192]` |
| Selected context tokens | `[2, 85, 192]` |
| Context encoder output | `[2, 85, 192]` |
| Context projected into predictor space | `[2, 85, 96]` |
| Context repeated for four target blocks | `[8, 85, 96]` |
| Position-conditioned target tokens | `[8, 36, 96]` |
| Combined predictor input | `[8, 121, 96]` |
| Predictor Transformer output | `[8, 121, 96]` |
| Selected predictor target tokens | `[8, 36, 96]` |
| Final predicted representations | `[8, 36, 192]` |
| Target representations for loss | `[8, 36, 192]` |
| Smooth L1 loss | Scalar |

The key alignment is

$$
\boxed{\hat{Y},Y\in\mathbb{R}^{(BM)\times L\times D},}
$$

with batch size $B$, $M$ target blocks, $L$ target patches per block, and encoder dimension $D$.

**Matching shapes are necessary but not sufficient:** the tensors must also have matching image, block, and patch-index ordering.

## 11. A minimal ViT-Tiny training step

The example below uses the official repository's classes and helpers without distributed training. It performs one step with randomly generated images to make the implementation transparent and test tensor compatibility.

The example uses our **illustrative**, lightweight predictor ($D_p=96$, depth 6), not an officially reported Tiny recipe. This code will not produce meaningful pretrained features from one step on random data.

```python
import copy

import torch
import torch.nn.functional as F

from src.helper import init_model
from src.masks.multiblock import MaskCollator
from src.masks.utils import apply_masks


# --------------------------------------------------
# 1. Initialize networks
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

encoder, predictor = init_model(
    device=device,
    model_name="vit_tiny",
    patch_size=16,
    crop_size=224,
    pred_depth=6,
    pred_emb_dim=96,
)

target_encoder = copy.deepcopy(encoder)
target_encoder.requires_grad_(False)


# --------------------------------------------------
# 2. Initialize mask sampler
# --------------------------------------------------

mask_sampler = MaskCollator(
    input_size=224,
    patch_size=16,
    enc_mask_scale=(0.85, 1.0),
    pred_mask_scale=(0.15, 0.20),
    aspect_ratio=(0.75, 1.5),
    nenc=1,
    npred=4,
    min_keep=10,
    allow_overlap=False,
)


# --------------------------------------------------
# 3. Create a dummy batch
# --------------------------------------------------

batch = [
    (torch.randn(3, 224, 224), 0)
    for _ in range(2)
]

data, masks_enc, masks_pred = mask_sampler(batch)
images = data[0].to(device)

masks_enc = [m.to(device) for m in masks_enc]
masks_pred = [m.to(device) for m in masks_pred]


# --------------------------------------------------
# 4. Optimizer
# --------------------------------------------------

optimizer = torch.optim.AdamW(
    list(encoder.parameters()) +
    list(predictor.parameters()),
    lr=1e-4
)


# --------------------------------------------------
# 5. Target encoder forward
# --------------------------------------------------

with torch.no_grad():
    targets = target_encoder(images)
    targets = F.layer_norm(
        targets,
        (targets.size(-1),)
    )
    targets = apply_masks(targets, masks_pred)


# --------------------------------------------------
# 6. Context encoder forward
# --------------------------------------------------

context = encoder(images, masks_enc)


# --------------------------------------------------
# 7. Predictor forward
# --------------------------------------------------

predictions = predictor(
    context,
    masks_enc,
    masks_pred
)


# --------------------------------------------------
# 8. Compute loss
# --------------------------------------------------

loss = F.smooth_l1_loss(
    predictions,
    targets
)


# --------------------------------------------------
# 9. Gradient update
# --------------------------------------------------

optimizer.zero_grad(set_to_none=True)
loss.backward()
optimizer.step()


# --------------------------------------------------
# 10. EMA update
# --------------------------------------------------

momentum = 0.996

with torch.no_grad():
    for online_param, target_param in zip(
        encoder.parameters(),
        target_encoder.parameters()
    ):
        target_param.mul_(momentum).add_(
            online_param,
            alpha=1.0 - momentum
        )


print("Predictions:", predictions.shape)
print("Targets:", targets.shape)
print("Loss:", loss.item())
```

### 11.1 What this code demonstrates

It reconstructs the central learning mechanism:

```text
Full-image target encoding
           |
           v
Visible-context encoding
           |
           v
Position-conditioned prediction
           |
           v
Latent regression loss
           |
           v
Gradient update
           |
           v
Target encoder EMA
```

It intentionally omits learning-rate and weight-decay schedules, mixed precision, real-data loading, and checkpoint management. Those affect the training recipe but not the central architecture.

## 12. Practical configuration and optimization

The official repository initializes models with `init_model()` from `src/helper.py`. It selects the requested backbone and builds a compatible predictor.

For a Tiny experiment, the relevant **illustrative** architecture settings are:

```yaml
meta:
  model_name: vit_tiny
  pred_depth: 6
  pred_emb_dim: 96
```

Example image and masking settings:

```yaml
data:
  crop_size: 224

mask:
  patch_size: 16
  num_enc_masks: 1
  num_pred_masks: 4
  enc_mask_scale: [0.85, 1.0]
  pred_mask_scale: [0.15, 0.2]
  aspect_ratio: [0.75, 1.5]
  allow_overlap: false
  min_keep: 10
```

These are **partial YAML excerpts**, not a complete runnable config. The training script also expects dataset, logging, and optimization settings.

### 12.1 Optimization

The official implementation uses **AdamW** to optimize context encoder and predictor, separating parameters into decay and no-decay groups. Biases and one-dimensional parameters (e.g., LayerNorm scales) are typically excluded from weight decay.

Learning rate uses warmup and cosine decay; weight decay also follows a schedule.

The released ImageNet-1K ViT-H/14 reference YAML contains:

| Parameter | Reference value |
|---|---|
| Optimizer | AdamW |
| Initial learning rate | $2\times10^{-4}$ |
| Peak learning rate | $10^{-3}$ |
| Final learning rate | $10^{-6}$ |
| Warmup | 40 epochs |
| Initial weight decay | 0.04 |
| Final weight decay | 0.4 |
| Initial EMA momentum | 0.996 |
| Final EMA momentum | 1.0 |

The paper's appendix describes somewhat different optimization settings, including a 15-epoch warmup starting at $10^{-4}$. For an exact checkpoint reproduction, follow its specific released configuration and implementation.

The Tiny backbone is much smaller than ViT-Huge, so its best hyperparameters may differ from this reference recipe.

Source: [reference YAML](https://github.com/facebookresearch/ijepa/blob/main/configs/in1k_vith14_ep300.yaml).

### 12.2 Data preprocessing

I-JEPA avoids relying on aggressive, manually designed augmented views, but it still uses cropping and normalization.

The official preprocessing contains operations conceptually like:

```python
transforms.RandomResizedCrop(
    crop_size,
    scale=crop_scale
)
transforms.ToTensor()
transforms.Normalize(mean, std)
```

For the reference configuration:

```yaml
crop_size: 224
crop_scale: [0.3, 1.0]

use_horizontal_flip: false
use_color_distortion: false
use_gaussian_blur: false
```

**No handcrafted strong view augmentations does not mean no preprocessing.** I-JEPA's primary self-supervised signal comes from missing-region latent prediction within an image view, not enforcing agreement between multiple strongly augmented views.

Source: [`src/transforms.py`](https://github.com/facebookresearch/ijepa/blob/main/src/transforms.py).

---

## Part IV — Important Design Choices and Interpretation

## 13. Why the components work together

### 13.1 The target encoder defines the representation space

The target encoder defines **what the predictor should reproduce**. Because it processes the full image, its patch embeddings may contain information from the broader visual scene.

The model thus predicts a hidden patch's representation **as contextualized by the full image**, rather than its isolated features or raw pixels.

The target encoder is not a fixed pretrained teacher: its features evolve during training via EMA. The context encoder and predictor continually adapt to that changing representation space.

### 13.2 The context encoder learns features useful for prediction

The context encoder does not reconstruct RGB images. It creates features that permit the predictor to infer representations of unseen regions. These features must retain information useful for interpreting relationships among image regions—potentially object identity, shape, or spatial arrangement.

The loss therefore rewards visible-context representations that explain unobserved image regions.

### 13.3 The predictor separates representation learning from prediction

The context encoder is intended to learn reusable visual features; the predictor specializes in mapping those features to particular missing-region representations. It mediates the prediction task without being needed for most downstream applications.

At evaluation time, the predictor can be discarded. This also explains why the paper investigates predictor width and depth: predictor capacity affects the difficulty of the optimization problem and the learning signal available to the encoder.

### 13.4 Masking determines what must be learned

Tiny targets can invite shortcuts from local correlations; excessively large targets can remove too much predictable information. A tiny context is ambiguous; overlapping context and targets may leak the answer.

Mask geometry is therefore not just preprocessing. **It changes the learning problem and the resulting inductive bias.**

### 13.5 EMA stabilizes the targets

Because target representations evolve during training, a rapidly changing target encoder can make the regression task unstable. EMA makes the target representation space change more slowly. Stop-gradient, momentum updates, and architecture asymmetry work together empirically to support stable training.

## 14. I-JEPA versus MAE: precise differences

It is tempting to describe I-JEPA as MAE with feature reconstruction instead of pixel reconstruction. That is partly correct, but misses its target network and structured masking.

| Property | MAE | I-JEPA |
|---|---|---|
| Visible input | Unmasked image patches | Unmasked context patches |
| Encoder | ViT | ViT |
| Reconstruction/prediction network | Decoder | Predictor |
| Prediction target | Original patch pixels | Learned target-encoder embeddings |
| Target encoder | Not required | EMA-updated ViT |
| Target contextualization | Pixel values | Features computed using full image |
| Masking strategy | Typically random patch masking | Spatial multi-block masking |
| Training loss | Pixel reconstruction | Feature regression |
| Mask tokens | Decoder | Predictor |
| EMA target network | Not required | Used |
| Main representation | Encoder features | Encoder features |

**The target encoder and multi-block masking strategy are defining parts of I-JEPA**, not details that can be dropped without changing the method.

## 15. Common implementation mistakes

### 15.1 Masking before target encoding

Incorrect:

```python
targets = target_encoder(masked_images)
```

Correct:

```python
features = target_encoder(images)
targets = apply_masks(features, masks_pred)
```

The target encoder must see the complete image before target features are selected.

### 15.2 Adding mask tokens inside the context encoder

The context encoder should process visible image-derived tokens **only**. Mask tokens belong in the predictor. Putting them in the context encoder changes the architecture and what the encoder can attend to.

### 15.3 Using incorrect patch positions

Masks contain indices relative to the **original patch grid**, so positions must continue to refer to that grid even after selecting tokens. Reindexing selected patches as if they formed a new contiguous image can corrupt the predictor's spatial conditioning.

### 15.4 Accidentally training the target encoder

Exclude the target encoder from gradient optimization and update it only with EMA:

```python
target_encoder.requires_grad_(False)
```

Use `torch.no_grad()` for its target-feature forward pass.

### 15.5 Misaligning prediction and target tensors

Equal tensor shapes do not guarantee correct supervision. Predictions must match **the same image, target block, and patch position** in the target tensor. A silent ordering mismatch can produce a valid numerical loss for the wrong learning objective.

### 15.6 Assuming ViT-Tiny has a prescribed official predictor

The repository defines `vit_tiny`, but the paper's primary results and released pretrained checkpoints use larger models. A Tiny educational experiment can choose a smaller predictor, but should report that choice rather than treating it as a published hyperparameter.

## 16. After pretraining: feature extraction

During pretraining, I-JEPA uses

$$
\text{Context encoder}
+\text{Predictor}
+\text{Target encoder}.
$$

After pretraining, the prediction task and predictor are no longer needed for ordinary feature extraction. The paper uses the target encoder for evaluation.

Given an image $I$, compute

$$
S=f_{\bar\theta}(I)\in\mathbb{R}^{N\times D}.
$$

A global image feature can be formed by average pooling:

$$
z=\frac1N\sum_{j=1}^{N}S_j.
$$

For ViT-Tiny, $z\in\mathbb{R}^{192}$. A frozen-feature linear classifier can then compute

$$
\hat y=Wz+b.
$$

For spatial tasks, downstream models may instead use patch-level features. The evaluation setup in the paper can also pool or concatenate representations from multiple final Transformer blocks.

## 17. Final synthesis

I-JEPA constructs a structured latent prediction problem. First, a full-image target encoder produces contextualized patch representations:

$$
S=f_{\bar\theta}(I).
$$

Mask sampling chooses one visible context and several target regions:

$$
C,\quad T_1,\ldots,T_M.
$$

The context encoder receives only visible patches:

$$
S_C=f_\theta(I;C).
$$

The predictor estimates representations for each target region using context features and positional information:

$$
\hat Y_i=g_\phi(S_C,T_i),\qquad Y_i=S[T_i].
$$

The abstract learning objective is

$$
\boxed{
\min_{\theta,\phi}
\mathbb{E}_{I,C,T}\left[
\sum_i
D\!\left(
 g_\phi\!\left(f_\theta(I;C),T_i\right),
 f_{\bar\theta}(I)[T_i]
\right)
\right].
}
$$

Here, $D(\cdot,\cdot)$ denotes a latent regression discrepancy, implemented as Smooth L1 in the official training code. The target encoder follows the context encoder by EMA:

$$
\boxed{
\bar\theta_{t+1}
=m_t\bar\theta_t+(1-m_t)\theta_{t+1}.
}
$$

| Component | Primary role |
|---|---|
| **Context encoder** | Produce visible-context features useful for predicting missing regions |
| **Target encoder** | Provide slowly evolving, full-image-contextualized feature targets |
| **Predictor** | Infer missing-region features from visible context and spatial positions |
| **Multi-block masking** | Define a challenging but informative spatial prediction problem |
| **Positional embeddings** | Tell the predictor where missing features belong |
| **Latent regression loss** | Align predictions with learned target representations |
| **EMA** | Smooth the evolution of target representations |

The context encoder processes only visible patches. The predictor appends position-conditioned mask tokens and uses self-attention to infer missing representations. The target encoder processes the complete image and supplies supervision in feature space.

> **The central insight:** I-JEPA does not learn by reconstructing image pixels. It learns by making representations of visible regions informative enough to predict contextualized representations of unobserved regions.

The geometry of the context and targets, and the way target features are constructed, turn the general JEPA principle into the specific I-JEPA method.

---

## Source references

### Paper

- [Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture (PDF)](https://arxiv.org/pdf/2301.08243)
- Section 3 — Method: architecture, target/context construction, predictor, objective.
- Appendix A.1 — Pretraining: architecture and optimization details.
- Appendix C — Additional Ablations: masking, target encoding, predictor capacity.

### Official implementation

- [Vision Transformer and predictor](https://github.com/facebookresearch/ijepa/blob/main/src/models/vision_transformer.py)
- [Multi-block mask sampler](https://github.com/facebookresearch/ijepa/blob/main/src/masks/multiblock.py)
- [Mask application](https://github.com/facebookresearch/ijepa/blob/main/src/masks/utils.py)
- [Training loop](https://github.com/facebookresearch/ijepa/blob/main/src/train.py)
- [Model and optimizer initialization](https://github.com/facebookresearch/ijepa/blob/main/src/helper.py)
- [Image transforms](https://github.com/facebookresearch/ijepa/blob/main/src/transforms.py)
- [Reference ImageNet-1K ViT-H/14 configuration](https://github.com/facebookresearch/ijepa/blob/main/configs/in1k_vith14_ep300.yaml)

**Recommended code walkthrough:** inspect `VisionTransformerPredictor.forward()`, then the `forward_target()` and `forward_context()` logic in `train.py`. Together, they connect the mathematical objective to the actual implementation.

---

*Equation syntax: inline `$...$`, display `$$...$$` (MathJax-compatible Markdown).*
