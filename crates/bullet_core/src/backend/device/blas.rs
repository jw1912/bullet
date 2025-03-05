#[derive(Clone, Copy, Debug)]
pub struct GemmConfig {
    pub alpha: f32,
    pub beta: f32,
    pub shape_a: Shape,
    pub trans_a: bool,
    pub shape_b: Shape,
    pub trans_b: bool,
}

impl GemmConfig {
    pub fn new(alpha: f32, beta: f32, shape_a: Shape, trans_a: bool, shape_b: Shape, trans_b: bool) -> Self {
        Self { alpha, beta, shape_a, trans_a, shape_b, trans_b }
    }

    pub fn output_shape(&self) -> Shape {
        self.shape_a.maybe_transpose(self.trans_a) * self.shape_b.maybe_transpose(self.trans_b)
    }
}

pub trait BlasOperations {
    type BlasError;

    fn gemm(&mut self, config: &GemmConfig, a: &Self, b: &Self) -> Result<(), Self::BlasError>;

    fn gebmm(&mut self, config: &GemmConfig, batch_size: usize, a: &Self, b: &Self) -> Result<(), Self::BlasError>;

    /// If `input_a = None`, then take `input_a = output`, i.e. perform the
    /// in place operation `output = alpha * output + beta * input_b`.
    ///
    /// If `input_b = None` then this is equivalent to a scaling operation.
    fn geam(
        &mut self,
        size: usize,
        alpha: f32,
        a: Option<&Self>,
        beta: f32,
        b: Option<&Self>,
    ) -> Result<(), Self::BlasError>;
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct Shape {
    rows: usize,
    cols: usize,
}

impl std::fmt::Display for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} x {}", self.rows, self.cols)
    }
}

impl std::ops::Mul<Shape> for Shape {
    type Output = Shape;
    fn mul(self, rhs: Shape) -> Self::Output {
        self.matmul(rhs).unwrap()
    }
}

impl Shape {
    pub fn matmul(self, rhs: Shape) -> Option<Self> {
        let shape = Self { cols: rhs.cols, rows: self.rows };
        (self.cols == rhs.rows).then_some(shape)
    }

    pub fn new(rows: usize, cols: usize) -> Self {
        assert!(cols > 0, "Cannot have 0 columns!");
        assert!(rows > 0, "Cannot have 0 rows!");
        Self { cols, rows }
    }

    pub fn transpose(&self) -> Self {
        Self { cols: self.rows, rows: self.cols }
    }

    pub fn maybe_transpose(&self, trans: bool) -> Self {
        if trans {
            self.transpose()
        } else {
            *self
        }
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn size(&self) -> usize {
        self.cols * self.rows
    }
}
