use crate::graph::ir::shape::Shape;

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

    // If `input_a = None`, then take `input_a = output`, i.e. perform the
    // in place operation `output = alpha * output + beta * input_b`.
    //
    // If `input_b = None` then this is equivalent to a scaling operation.
    //fn geam(
    //    &mut self,
    //    size: usize,
    //    alpha: f32,
    //    a: Option<&Self>,
    //    beta: f32,
    //    b: Option<&Self>,
    //) -> Result<(), Self::BlasError>;
}
