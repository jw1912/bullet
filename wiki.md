# Training a Neural Network for Chess

__Warning: Assumes a basic knowledge of linear algebra and differentiation.__

This document covers how to train a neural network architectures of the form `768 -> N -> 1`, which is then
extended to architectures of the form `768 -> Nx2 -> 1`.

## How does the Network work?

### Input

The input of a basic neural network is a vector of `768 = 2 x 6 x 64` zeros or ones, where a one at a certain index
represents the presence of a particular piece on a particular square, and a zero represents an absence of that piece.

The standard way to do this is to set `white_pawn = 0, white_knight = 1, ..., black_pawn = 6, ..., black_king = 11` and
for each piece `Piece` on square `Square`, you set the `64 * Piece + Square`th element of the input vector to 1.

From now on we denote this input vector by $\mathbf{x}$.

### Hidden Layer

The hidden layer in a `768 -> N -> 1` network is an N-dimensional vector, and is a function of the input vector for which
we have to define some new things:

- $H$, the hidden weights, an `Nx768` dimensional matrix
- $\mathbf{b}$, the hidden bias, an N-dimensional vector
- $\sigma$, the activation function, which acts element-wise on any vector

With these we define the accumulator, $a = H \mathbf{x} + b$, and then the hidden layer itself is $\mathbf{h} = \sigma (\mathbf{a})$.

### Output Layer

The output $y$ is a scalar value, as it needs to be used in a chess engine. We define:

- $O$, the output weight, an `1xN` dimensional matrix
- $c$, the output bias, a scalar constant

Then the output is defined as $y = O \mathbf{h} + c$, notice that the product of a `1xN` matrix with an N-dimensional vector is a
`1x1` matrix, i.e a scalar (we could have also defined $O$ as a vector and used a dot product, but this maintains consistency).

### Bringing it all together

Writing out the full calculation, we have

$$
y(\mathbf{x}) = O \sigma( H \mathbf{x} + \mathbf{b} ) + c
$$

## What is Gradient Descent?

### Definition

You have a (differentiable) function $f$ of a vector of parameters $\mathbf{p} = (p_1, p_2, ..., p_n)^T$, and you want to find its minimum.

You pick some starting parameters $\mathbf{x_0}$ and a learning rate $\epsilon$, and then proceed by iterating

$$
\mathbf{p_{i + 1}} = \mathbf{p_i} - \epsilon \nabla f(\mathbf{p_i})
$$

until you (hopefully) converge on a minimum, where $\nabla f$ is the gradient of $f$, and is defined by

$$
\nabla f = (\frac{\partial f}{\partial p_1}, \frac{\partial f}{\partial p_2}, ..., \frac{\partial f}{\partial p_n})
$$

### How do we apply it to chess engines?

We decide our function $f$ is going to be the loss over a dataset, but what do we mean by that?

Our dataset consists of a large number of pairs $(\mathbf{x_i}, R_i)$, where $\mathbf{x_i}$ is a position represented as an input vector,
and $R_i$ is the corresponding expected result for the position, what we want our network's output $y$ to be as close as possible to.

Our parameters are $H, \mathbf{b}, O, c$ concatenated into a vector $\mathbf{p}$.

We denote the output of our network with an input $\mathbf{x}$ and parameters $\mathbf{p}$ by $y(\mathbf{x}; \mathbf{p})$.

Now we define our loss, $f$, by

$$
f(\mathbf{p}) = \frac{1}{N} \sum_{i=1}^{N} (sigmoid(y(\mathbf{x_i}; \mathbf{p})) - R_i)^2
$$

where 

$$
sigmoid(x) = \frac{1}{1 + e^{-x}}
$$

Now to calculate the gradient vector we just need 

## Derivatives w.r.t Weights

## The Final Gradient

## A Perspective Network
