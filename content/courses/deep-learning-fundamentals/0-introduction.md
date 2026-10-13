---
title: "Introduction"
weight: 10
categories: ["Theory"]
tags: ["Deep Learning"]
draft: false
---

# Introduction
Machine learning problems are broadly divided into two families according to the information available during training.

## Supervised Learning

In **supervised learning**, each training example is associated with a known **label**. The goal is to learn a mapping from inputs to outputs.

Typical supervised-learning tasks include:

- **classification**, where the output belongs to a discrete set of classes;
- **regression**, where the output is a continuous value.

For example, in spam detection, emails are labeled as either **spam** or **not spam**, and the model learns to predict the appropriate label for new emails.

## Unsupervised Learning

In **unsupervised learning**, training examples do not have labels. The goal is therefore to discover useful structure or patterns in the data.

Typical unsupervised-learning tasks include **clustering** and **dimensionality reduction**.

For example, customers can be grouped according to similarities in their purchasing behavior without having predefined customer categories.

**Logistic regression is a supervised-learning algorithm**, because it is trained using labeled examples $(x_i, y_i)$.

# Training, Validation, and Test Sets

The available dataset is usually divided into three disjoint subsets.

The **training set** is used to estimate the model parameters.

The **validation set** is used to select hyperparameters, such as the learning rate, and to monitor phenomena such as overfitting during model development.

The **test set** is used only after model development has been completed, in order to estimate performance on previously unseen data.

The test set should not be used to make modeling or hyperparameter-selection decisions. Otherwise, information from the test data leaks into the development process and the resulting performance estimate becomes optimistically biased.

# Goal of Supervised Learning

Given a training set of input-output pairs, supervised learning aims to learn a function $f$ mapping an input $x$ to an output $Y$.

The objective is not merely to obtain good performance on the training data. More importantly, the learned function should **generalize** to new examples that were not observed during training.

In practice, $f$ is usually selected from a parameterized family of functions. If the model is represented by $f_W$, where $W$ denotes its parameters, learning consists of finding values of $W$ that fit the training data while generalizing well to unseen data.

# Supervised-Learning Setting

Consider a training dataset

$$
D = \{(x_i,Y_i)\}_{i=1}^{N},
$$

where

$$
x_i \in \mathbb{R}^m,
\qquad
Y_i \in \{0,1\}.
$$

Here:

- $N$ is the number of training examples;
- each input $x_i$ contains $m$ numerical features;
- each label $Y_i$ belongs to one of two classes, $0$ or $1$.

An input can be written as

$$
x_i =
\begin{bmatrix}
x_i^{(1)} \\
x_i^{(2)} \\
\vdots \\
x_i^{(m)}
\end{bmatrix}.
$$

## Features

A **feature** is a measurable property of an input example that represents some aspect of the example numerically.

Each example $x_i$ is therefore described by $m$ features:

$$
x_i^{(1)}, x_i^{(2)}, \ldots, x_i^{(m)}.
$$

Features should capture information that is useful for the prediction task.

For example, when classifying an email as spam or not spam, possible features include the number of occurrences of the word "free", the proportion of capital letters, the number of exclamation marks, whether the sender is in the recipient's contact list, and the total length of the email.

# Logistic Regression

Logistic regression is a **probabilistic discriminative classifier**.

For binary classification, it directly models the posterior probability that the label is $1$:

$$
P(Y=1\mid x) =
\frac{1}{1+e^{-(w^Tx+b)}}.
$$

Since the two class probabilities must sum to one,

$$
P(Y=0\mid x) =
1-P(Y=1\mid x) =
\frac{1}{1+e^{w^Tx+b}}.
$$

The predicted class is the one with the largest posterior probability:

$$
\hat{Y} =
f_{\mathrm{LR}}(x) =
\arg\max_{c\in\{0,1\}} P(Y=c\mid x).
$$

## Sigmoid Function

Logistic regression applies the **sigmoid function** to an affine transformation of the input:

$$
f(x_i,w,b) =
\sigma(w^Tx_i+b),
$$

where

$$
\sigma(z) =
\frac{1}{1+e^{-z}} =
\frac{e^z}{1+e^z}.
$$

The output of the sigmoid satisfies

$$
0 < \sigma(z) < 1,
$$

which allows it to be interpreted as a probability.

The weight vector is

$$
w\in\mathbb{R}^m,
$$

while $b\in\mathbb{R}$ is the **bias** or **intercept**.

During training, the parameters are iteratively adjusted to improve the model's predictions.

# The Affine Transformation $w^Tx+b$

The quantity

$$
z=w^Tx+b
$$

can be expanded as

$$
z =
\sum_{k=1}^{m}w^{(k)}x^{(k)}+b.
$$

Strictly speaking, $w^Tx+b$ is an **affine transformation**, because it consists of the linear map $w^Tx$ followed by the translation $b$.

In machine learning, it is nevertheless common to informally call this transformation "linear."

The set

$$
\{x\in\mathbb{R}^m:w^Tx+b=0\}
$$

is a hyperplane in $\mathbb{R}^m$ and constitutes the decision boundary of binary logistic regression when the classification threshold is $0.5$.

# Learning Logistic Regression

The parameters of logistic regression are estimated by maximizing the likelihood of the observed training labels.

Given

$$
D=\{(x_i,y_i)\}_{i=1}^{N},
$$

the likelihood is

$$
L(W\mid D) =
\prod_{i=1}^{N}
P(Y=y_i\mid X=x_i;W),
$$

assuming the training examples are conditionally independent given the model parameters.

Maximum-likelihood estimation chooses

$$
W^* =
\arg\max_W L(W\mid D).
$$

Direct analytical maximization is generally unavailable, so numerical optimization algorithms are used.

# Log-Likelihood

Instead of maximizing the likelihood directly, it is more convenient to maximize its logarithm:

$$
\ell(W) =
\log L(W\mid D).
$$

Because the logarithm converts products into sums,

$$
\ell(W) =
\sum_{i=1}^{N}
\log P(Y=y_i\mid X=x_i;W).
$$

Since $\log(\cdot)$ is strictly increasing,

$$
\arg\max_W L(W\mid D) =
\arg\max_W \ell(W).
$$

Using the log-likelihood provides two important practical advantages.

First, it improves **numerical stability**. Multiplying many probabilities smaller than $1$ can produce extremely small numbers and cause floating-point underflow. Summing their logarithms avoids this problem.

Second, it simplifies optimization because derivatives of sums are much easier to compute than derivatives of large products.

# Absorbing the Bias into the Weight Vector

The affine transformation

$$
w^Tx+b
$$

can be rewritten as a single dot product by appending a constant feature equal to $1$ to every input vector.

Define

$$
\tilde{x} =
\begin{bmatrix}
x\\
1
\end{bmatrix},
\qquad
W =
\begin{bmatrix}
w\\
b
\end{bmatrix}.
$$

Then

$$
W^T\tilde{x} =
w^Tx+b.
$$

Thus, the bias can be treated as an additional model weight. This changes only the notation, not the model itself.

From this point onward, $x$ can be understood as the augmented input vector and $W$ as the corresponding bias-absorbed weight vector.

# Binary Logistic Regression

After absorbing the bias into $W$,

$$
P(Y=1\mid x) =
\sigma(W^Tx) =
\frac{1}{1+e^{-W^Tx}},
$$

and

$$
P(Y=0\mid x) =
1-P(Y=1\mid x) =
\frac{1}{1+e^{W^Tx}}.
$$

For a binary label $y_i\in\{0,1\}$, the probability of the observed label can be written compactly as

$$
P(Y=y_i\mid x_i) =
P(Y=1\mid x_i)^{y_i}
P(Y=0\mid x_i)^{1-y_i}.
$$

Taking logarithms gives

$$
\log P(Y=y_i\mid x_i) =
y_i\log P(Y=1\mid x_i)
+
(1-y_i)\log P(Y=0\mid x_i).
$$

Therefore, the total log-likelihood is

$$
\ell(W) =
\sum_{i=1}^{N}
\left[
y_i\log p_i
+
(1-y_i)\log(1-p_i)
\right],
$$

where

$$
p_i=P(Y=1\mid x_i)=\sigma(W^Tx_i).
$$

The negative log-likelihood is the **binary cross-entropy loss**, up to whether a sum or mean over examples is used.

Therefore,

$$
\text{maximizing log-likelihood}
\iff
\text{minimizing binary cross-entropy}.
$$

# Binary Cross-Entropy Loss

A common training objective is the mean binary cross-entropy:

$$
\mathcal{L}(W) =
-\frac{1}{N}
\sum_{i=1}^{N}
\left[
Y_i\log f(x_i,W)
+
(1-Y_i)\log\left(1-f(x_i,W)\right)
\right],
$$

where

$$
f(x_i,W)=\sigma(W^Tx_i).
$$

The objective of training is to find parameters $W$ that minimize this loss.

# Gradient Descent

**Gradient descent** is an iterative optimization algorithm used to minimize a differentiable function.

Given a loss $\mathcal{L}(W)$, gradient descent moves the parameters in the direction opposite to the gradient because the gradient identifies the direction of steepest local increase.

## Gradient of a Multivariate Function

For

$$
W=(W_1,\ldots,W_m)^T,
$$

the gradient is

$$
\nabla\mathcal{L}(W) =
\begin{bmatrix}
\dfrac{\partial\mathcal{L}}{\partial W_1}\\
\dfrac{\partial\mathcal{L}}{\partial W_2}\\
\vdots\\
\dfrac{\partial\mathcal{L}}{\partial W_m}
\end{bmatrix}.
$$

Each component

$$
\frac{\partial\mathcal{L}}{\partial W_j}
$$

measures the rate at which the loss changes as $W_j$ varies while the other parameters are held fixed.

The gradient $\nabla\mathcal{L}(W)$ points in the direction of steepest local ascent. Consequently,

$$
-\nabla\mathcal{L}(W)
$$

is the direction of steepest local descent under the Euclidean norm.

# Gradient-Descent Update

For each parameter,

$$
W_j
\leftarrow
W_j -
\alpha
\frac{\partial\mathcal{L}(W)}{\partial W_j},
$$

or, in vector form,

$$
W
\leftarrow
W-\alpha\nabla\mathcal{L}(W).
$$

The positive scalar $\alpha$ is called the **learning rate** and controls the size of each optimization step.

# Gradient Descent as a Local Optimization Problem

The gradient-descent update can be derived from a local approximation of the loss.

At iteration $t$, approximate $\mathcal{L}(W)$ around $W_t$ using its first-order Taylor expansion:

$$
\mathcal{L}(W)
\approx
\mathcal{L}(W_t)
+
\nabla\mathcal{L}(W_t)^T(W-W_t).
$$

Because this approximation is reliable only locally, introduce a quadratic penalty that discourages moving too far away from $W_t$:

$$
W_{t+1} =
\arg\min_W
\left[
\mathcal{L}(W_t)
+
\nabla\mathcal{L}(W_t)^T(W-W_t)
+
\frac{1}{2\alpha}
\|W-W_t\|_2^2
\right].
$$

The quadratic term controls the size of the step. A smaller $\alpha$ corresponds to a stronger penalty and therefore produces a smaller update.

To solve this optimization problem, differentiate the objective with respect to $W$:

$$
\nabla\mathcal{L}(W_t)
+
\frac{1}{\alpha}(W-W_t) =
0.
$$

Therefore,

$$
W-W_t =
-\alpha\nabla\mathcal{L}(W_t),
$$

and hence

$$
W_{t+1} =
W_t-\alpha\nabla\mathcal{L}(W_t).
$$

This is exactly the standard gradient-descent update.

# Learning Rate

The learning rate $\alpha$ strongly affects optimization.

If $\alpha$ is too small, the parameter updates are very small and convergence can be extremely slow.

If $\alpha$ is too large, the algorithm can overshoot the minimum, oscillate, or even diverge.

An appropriate learning rate therefore balances optimization speed and stability.

# Convexity

A function $\mathcal{L}$ is **convex** if, for any $x$, $y$, and $\lambda\in[0,1]$,

$$
\mathcal{L}(\lambda x+(1-\lambda)y)
\leq
\lambda\mathcal{L}(x)
+
(1-\lambda)\mathcal{L}(y).
$$

For a differentiable convex function, every local minimum is also a global minimum.

Under suitable conditions on the learning rate and the objective, gradient-based optimization can converge to a global minimizer of a convex loss.

For non-convex objectives, optimization may instead reach local minima, saddle points, or other stationary regions.

The unregularized binary logistic-regression cross-entropy loss is convex with respect to its parameters.

# Simplifying the Logistic-Regression Loss

For logistic regression,

$$
f(x_i,W) =
\sigma(W^Tx_i) =
\frac{e^{W^Tx_i}}
{1+e^{W^Tx_i}}.
$$

Also,

$$
1-f(x_i,W) =
\frac{1}
{1+e^{W^Tx_i}}.
$$

Starting from

$$
\mathcal{L}(W) =
-\frac{1}{N}
\sum_{i=1}^{N}
\left[
Y_i\log f(x_i,W)
+
(1-Y_i)\log(1-f(x_i,W))
\right],
$$

substitute the logistic probabilities:

$$
\mathcal{L}(W) =
-\frac{1}{N}
\sum_{i=1}^{N}
\left[
Y_i
\log
\left(
\frac{e^{W^Tx_i}}
{1+e^{W^Tx_i}}
\right)
+
(1-Y_i)
\log
\left(
\frac{1}
{1+e^{W^Tx_i}}
\right)
\right].
$$

Using logarithm properties,

$$
\log
\left(
\frac{e^{W^Tx_i}}
{1+e^{W^Tx_i}}
\right) =
W^Tx_i -
\log(1+e^{W^Tx_i}),
$$

while

$$
\log
\left(
\frac{1}
{1+e^{W^Tx_i}}
\right) =
-\log(1+e^{W^Tx_i}).
$$

Therefore,

$$
\mathcal{L}(W) =
-\frac{1}{N}
\sum_{i=1}^{N}
\left[
Y_iW^Tx_i -
\log(1+e^{W^Tx_i})
\right].
$$

Equivalently,

$$
\mathcal{L}(W) =
\frac{1}{N}
\sum_{i=1}^{N}
\left[
\log(1+e^{W^Tx_i}) -
Y_iW^Tx_i
\right].
$$

# Derivative of the Logistic-Regression Loss

Consider

$$
\mathcal{L}(W) =
-\frac{1}{N}
\sum_{i=1}^{N}
\left[
Y_iW^Tx_i -
\log(1+e^{W^Tx_i})
\right].
$$

For parameter $W_j$,

$$
\frac{\partial(W^Tx_i)}{\partial W_j} =
x_i^{(j)}.
$$

Moreover,

$$
\frac{\partial}{\partial W_j}
\log(1+e^{W^Tx_i}) =
\frac{e^{W^Tx_i}}
{1+e^{W^Tx_i}}
x_i^{(j)}.
$$

Since

$$
\frac{e^{W^Tx_i}}
{1+e^{W^Tx_i}} =
\sigma(W^Tx_i) =
f(x_i,W),
$$

we obtain

$$
\frac{\partial\mathcal{L}(W)}
{\partial W_j} =
-\frac{1}{N}
\sum_{i=1}^{N}
\left[
Y_i-f(x_i,W)
\right]
x_i^{(j)}.
$$

Equivalently,

$$
\frac{\partial\mathcal{L}(W)}
{\partial W_j} =
\frac{1}{N}
\sum_{i=1}^{N}
\left[
f(x_i,W)-Y_i
\right]
x_i^{(j)}.
$$

# Final Gradient and Update

Collecting all partial derivatives gives

$$
\nabla\mathcal{L}(W) =
\frac{1}{N}
\sum_{i=1}^{N}
\left(
f(x_i,W)-Y_i
\right)x_i.
$$

If the examples are arranged row-wise in a design matrix

$$
X\in\mathbb{R}^{N\times m},
$$

with

$$
y=
\begin{bmatrix}
Y_1\\
\vdots\\
Y_N
\end{bmatrix},
\qquad
\hat{y} =
\begin{bmatrix}
f(x_1,W)\\
\vdots\\
f(x_N,W)
\end{bmatrix},
$$

then the gradient is

$$
\nabla\mathcal{L}(W) =
\frac{1}{N}X^T(\hat{y}-y).
$$

The gradient-descent update is therefore

$$
W
\leftarrow
W-\alpha\nabla\mathcal{L}(W),
$$

or explicitly,

$$
W
\leftarrow
W -
\frac{\alpha}{N}
X^T(\hat{y}-y).
$$

# Decision Boundary

The decision boundary is the set of points at which the two classes are equally probable:

$$
P(Y=1\mid x) =
P(Y=0\mid x).
$$

Equivalently,

$$
\log
\frac{P(Y=1\mid x)}
{P(Y=0\mid x)} =
0.
$$

For logistic regression, the log-odds are

$$
\log
\frac{P(Y=1\mid x)}
{P(Y=0\mid x)} =
W^Tx.
$$

Therefore, the decision boundary satisfies

$$
W^Tx=0.
$$

Hence,

$$
\{x\in\mathbb{R}^m:W^Tx=0\}
$$

is a hyperplane.

If the bias is written explicitly rather than absorbed into $W$, the boundary is

$$
w^Tx+b=0.
$$

Although the predicted probability is obtained through the nonlinear sigmoid function, the decision boundary is linear in the input features. Logistic regression is therefore a **linear classifier**.

# Multiclass Logistic Regression

Logistic regression can be generalized from two classes to $K$ classes using **multinomial logistic regression**, also called **softmax regression**.

A common parameterization assigns one weight vector $W_c$ to each class and defines

$$
P(Y=c\mid x) =
\frac{\exp(W_c^Tx)}
{\sum_{l=1}^{K}\exp(W_l^Tx)},
\qquad
c=1,\ldots,K.
$$

Because adding the same vector contribution to every class score leaves the probabilities unchanged, one class can alternatively be chosen as a reference class. Setting its score to zero gives

$$
P(Y=c\mid x) =
\frac{\exp(W_c^Tx)}
{1+\sum_{l=1}^{K-1}\exp(W_l^Tx)},
\qquad
c=1,\ldots,K-1,
$$

and

$$
P(Y=K\mid x) =
\frac{1}
{1+\sum_{l=1}^{K-1}\exp(W_l^Tx)}.
$$

The predicted class is

$$
\hat{Y} =
\arg\max_{c\in\{1,\ldots,K\}}
P(Y=c\mid x).
$$

# Evaluating a Classifier: Accuracy

Given true labels $Y_i$ and predicted labels $\hat{Y}_i$, **accuracy** is the proportion of correctly classified examples:

$$
\operatorname{Accuracy} =
\frac{1}{N}
\sum_{i=1}^{N}
\mathbb{1}[Y_i=\hat{Y}_i],
$$

where $\mathbb{1}[\cdot]$ is the indicator function:

$$
\mathbb{1}[\text{condition}] =
\begin{cases}
1, & \text{if the condition is true},\\
0, & \text{otherwise}.
\end{cases}
$$

Accuracy satisfies

$$
0\leq\operatorname{Accuracy}\leq1.
$$

It is often expressed as a percentage.

Training accuracy can be useful for diagnosing optimization and overfitting, but estimates of generalization performance should be computed on validation or test data rather than inferred from training accuracy.

Accuracy can also be misleading on highly imbalanced datasets. For example, if $95\%$ of the examples belong to one class, a classifier that always predicts that class already achieves $95\%$ accuracy despite learning no useful distinction between the classes.

# Optimization Algorithms

## Batch Gradient Descent

In **batch gradient descent**, the gradient used for every parameter update is computed from the entire training dataset.

For a dataset containing $N$ examples,

$$
\nabla\mathcal{L}(W) =
\frac{1}{N}
\sum_{i=1}^{N}
\nabla\mathcal{L}_i(W).
$$

This provides an exact gradient of the empirical loss but can be computationally expensive when $N$ is very large.

## Mini-Batch Stochastic Gradient Descent

In **mini-batch stochastic gradient descent**, each update uses only a subset of $B$ examples, where typically

$$
B\ll N.
$$

For a mini-batch $\mathcal{B}$,

$$
g_t =
\frac{1}{B}
\sum_{i\in\mathcal{B}}
\nabla\mathcal{L}_i(W_t),
$$

and the update is

$$
W_{t+1} =
W_t-\alpha g_t.
$$

This gradient is an estimate of the full-dataset gradient.

Mini-batch training is particularly convenient because it provides a compromise between computational efficiency and gradient stability.

## Stochastic Gradient Descent

In the strict sense, **stochastic gradient descent (SGD)** uses a single randomly selected training example for each update:

$$
B=1.
$$

In modern machine-learning terminology, however, "SGD" is also commonly used more broadly to include mini-batch stochastic gradient methods.

# SGD with Momentum

Momentum incorporates information from previous gradients rather than relying only on the current gradient.

One common formulation is

$$
v_t =
\beta v_{t-1}
+
(1-\beta)g_t,
$$

followed by

$$
W_{t+1} =
W_t-\alpha v_t,
$$

where

$$
g_t=\nabla\mathcal{L}(W_t).
$$

The parameter

$$
\beta\in[0,1)
$$

determines how strongly past gradients are retained. A common value is

$$
\beta=0.9.
$$

Momentum tends to smooth the optimization trajectory, reduce oscillations in directions with rapidly changing gradients, and accelerate movement in directions where gradients are consistently aligned.

Different libraries may use slightly different but equivalent momentum conventions, for example omitting the factor $(1-\beta)$ and compensating through the effective learning rate.

# Adam Optimizer

**Adam**, or **Adaptive Moment Estimation**, combines two central ideas:

**Momentum-like averaging** uses an exponentially weighted average of past gradients to smooth the update direction.

**Adaptive scaling** uses an exponentially weighted average of squared gradients to assign different effective step sizes to different parameters.

Let

$$
g_t =
\nabla\mathcal{L}(W_t)
$$

be the gradient at iteration $t$.

## First-Moment Estimate

Adam maintains an exponential moving average of gradients:

$$
m_t =
\beta_1m_{t-1}
+
(1-\beta_1)g_t.
$$

This quantity estimates the first moment of the gradient.

## Second-Moment Estimate

Adam also maintains an exponential moving average of the element-wise squared gradients:

$$
v_t =
\beta_2v_{t-1}
+
(1-\beta_2)g_t^{\odot 2},
$$

where $g_t^{\odot 2}$ denotes element-wise squaring.

This quantity estimates the uncentered second moment of the gradient.

## Bias Correction

Because $m_0$ and $v_0$ are initialized to zero, the moving averages are biased toward zero during the first optimization steps.

Adam therefore computes

$$
\hat{m}_t =
\frac{m_t}
{1-\beta_1^t},
$$

and

$$
\hat{v}_t =
\frac{v_t}
{1-\beta_2^t}.
$$

## Adam Update

The parameters are updated element-wise according to

$$
W_{t+1} =
W_t -
\alpha
\frac{\hat{m}_t}
{\sqrt{\hat{v}_t}+\epsilon}.
$$

The small constant $\epsilon$ prevents numerical instability and division by zero.

Typical default values are

$$
\beta_1=0.9,
\qquad
\beta_2=0.999,
\qquad
\epsilon=10^{-8},
\qquad
\alpha=10^{-3}.
$$

# Adam: Advantages and Disadvantages

## Advantages

Adam often performs well with relatively little hyperparameter tuning. It can handle noisy and sparse gradients effectively and frequently reaches useful solutions quickly in practical machine-learning problems.

## Disadvantages

Adam requires additional memory because it stores both $m_t$ and $v_t$ for every parameter.

Depending on the model and task, solutions obtained using Adam may also generalize differently from those obtained with SGD or SGD with momentum. Consequently, Adam is not universally superior, and optimizer choice remains problem-dependent.

# References

1. J. Duchi, E. Hazan, and Y. Singer. *Adaptive Subgradient Methods for Online Learning and Stochastic Optimization*. Journal of Machine Learning Research, 12:2121–2159, 2011.

2. D. P. Kingma and J. Ba. *Adam: A Method for Stochastic Optimization*. arXiv:1412.6980, 2014.

3. M. D. Zeiler. *ADADELTA: An Adaptive Learning Rate Method*. arXiv:1212.5701, 2012.