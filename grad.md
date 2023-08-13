Model:

$$
y(\mathbf{x}) = O \sigma (H \mathbf{x} + \mathbf{b}) + c
$$

Partials:

$H_i$ denotes row $i$ of $H$.

$$
\frac{\partial y}{\partial c} = 1
$$

$$
\frac{\partial y}{\partial O_i} = \sigma (H_i \cdot \mathbf{x} + b_i)
$$

$$
\frac{\partial y}{\partial b_i} = O_i \sigma ' (H_i \cdot \mathbf{x} + b_i)
$$

$$
\frac{\partial y}{\partial H_{ij}} = O_i \sigma ' (H_i \cdot \mathbf{x} + b_i) x_j
$$

Error:

$$
E = \frac{1}{N} \sum_{i=1}^{N} (y(\mathbf{x}_i) - R_i)^2
$$

W.r.t arbitrary parameter $w$

$$
\frac{\partial E}{\partial w}
    = \frac{2}{N} \sum_{i=1}^{N} \left [
        (y(\mathbf{x}_i) - R_i) \frac{\partial y}{\partial w}(\mathbf{x}_i)
    \right ]
$$