# 1. NNUE Basics

## Simple Feed-Forward Network

### Input

The input of a basic neural network for chess is a vector of `768 = 2 x 6 x 64` zeros or ones, where a one at a certain index
represents the presence of a particular piece on a particular square, and a zero represents an absence of that piece.

The standard way to do this is to set `white_pawn = 0, white_knight = 1, ..., black_pawn = 6, ..., black_king = 11` and
for each piece `Piece` on square `Square`, you set the `64 * Piece + Square`th element of the input vector to 1.

From now on we denote this input vector by $\mathbf{x}$.

### Hidden Layer

The hidden layer in a `768 -> N -> 1` network is an `N`-dimensional vector, and is a function of the input vector for which
we have to define some new things:

- $H$, the hidden weights, an `Nx768` dimensional matrix
- $\mathbf{b}$, the hidden bias, an N-dimensional vector
- $\rho$, the activation function, which acts element-wise on any vector

With these we define the accumulator, $a = H \mathbf{x} + b$, and then the hidden layer itself is $\mathbf{h} = \rho (\mathbf{a})$.

### Output Layer

The output $y$ is a scalar value, as it needs to be used in a chess engine. We define:

- $O$, the output weight, an `1xN` dimensional matrix
- $c$, the output bias, a scalar constant

Then the output is defined as $y = O \mathbf{h} + c$, notice that the product of a `1xN` matrix with an N-dimensional vector is a
`1x1` matrix, i.e a scalar (we could have also defined $O$ as a vector and used a dot product, but this maintains consistency).

### Bringing it all together

Writing out the full calculation, we have

$$
y(\mathbf{x}) = O \rho( H \mathbf{x} + \mathbf{b} ) + c
$$

## Perspective Networks

A perspective network architecture `768 -> Nx2 -> 1` is very similar, except there are two sets of inputs,
$\mathbf{x}$ and $\mathbf{\hat{x}}$.

Unlike in the previous network, $\mathbf{x}$ is not from from white perspective, instead the piece types are labelled
`friendly_pawn = 0, ..., enemy_king = 11` from the perspective of the side to move, and then $\mathbf{\hat{x}}$ is the same
but from the perspective of the opposite side.

You have two accumulators now, $\mathbf{a} = H \mathbf{x} + \mathbf{b}$ and $\mathbf{\hat{a}} = H \mathbf{\hat{x}} + \mathbf{b}$,
and the output is now given by

$$
y = O \rho(concat(\mathbf{a}, \mathbf{\hat{a}})) + c
$$

In this case you can split $O$ into $O_1$ and $O_2$ for equivalently

$$
y = O_1 \rho(\mathbf{a}) + O_2 \rho(\mathbf{\hat{a}}) + c
$$

which is generally the form you will use in inference.

## Beginner Traps

### Poor Beginner Resources

#### Almost any article/blogpost/book on NNUE that isn't backed by the author's strong NNUE engine

- Sure, they may explain the core NNUE concept well
- Everything else will usually be bad advice, backed by little/no relevant evidence

#### Stockfish network architectures

- Specifically referring to the **architecture**, not actual SF networks (which are obviously very good)
- SF architectures have been parodied by many an engine
- Many aspects of the SF architectures require **significant** effort, amounts of data, and/or training time/complexity to actually gain elo
- As a result, an engine may (and likely will for a beginner) actually *lose* elo with an SF architecture vs a much simpler one

#### [nnue-pytorch's nnue.md](https://github.com/official-stockfish/nnue-pytorch/blob/master/docs/nnue.md)

- Good resource for general SF NNUE history + some more advanced stuff
- Follow on from the above
    - This is a document about Stockfish NNUE
    - You are not writing Stockfish
- Contains way more information than necessary for a beginner so it can be generally confusing
- Treats arguably the most reasonable starting architecture as a [toy example](https://github.com/official-stockfish/nnue-pytorch/blob/master/docs/nnue.md#a-simple-input-feature-set)
- Is generally outdated w.r.t the vast majority of its contents

### Skipped Progression

In general, it is good practise to start simple, and then incrementally increase complexity, whilst verifying that
each incremental step actually gains ELO.

Of particular relevance to the below beginner traps: at the time of writing, Alexandria 7.0.0 is 6th on CCRL Blitz 1CPU,
and its network architecture is `(768->1536)x2->1x8` - that is *no* input buckets and only a single hidden layer.

#### Massive input featureset

- Just start with basic 768 inputs
- You won't have enough data for things like HalfKA/HalfKP at first (or perhaps ever, custom bucket schemes
will generally serve better with less data)

#### More than 1 hidden layer

- Usually more beneficial to just increase the size of the first hidden layer (up to at least 1024)
- Whilst further layers gain at fixed nodes, it is non-trivial to get them to *not* lose lots of elo due to the speed hit
- Requires manual SIMD and well considered quantisation tech

## Good NNUE Resources

- [NNUE Performance Improvements](https://cosmo.tardis.ac/files/2024-06-01-nnue.html) - Cosmo Bobak (author of Viridithas)
