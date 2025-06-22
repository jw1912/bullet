use crate::graph::ir::op::DiffableFromOutput;

#[derive(Clone, Copy, Debug)]
pub struct AdamConfig {
    pub beta1: f32,
    pub beta2: f32,
    pub gradient_factor: f32,
    pub learning_rate: f32,
    pub denom: bool,
    pub decay: f32,
    pub clip: Option<(f32, f32)>,
}

pub trait BaseOperations {
    type BaseError;

    fn set_to(&mut self, size: usize, val: f32) -> Result<(), Self::BaseError>;

    #[allow(clippy::too_many_arguments)]
    fn copy_or_add_strided(
        &mut self,
        add: bool,
        rows: usize,
        cols: usize,
        offset: usize,
        stride: usize,
        a: &Self,
        offset_a: usize,
        stride_a: usize,
    ) -> Result<(), Self::BaseError>;

    fn diffable_from_output_fwd(
        &mut self,
        size: usize,
        a: &Self,
        act: DiffableFromOutput,
    ) -> Result<(), Self::BaseError>;

    fn diffable_from_output_bwd(
        &mut self,
        size: usize,
        a: &Self,
        grd: &Self,
        act: DiffableFromOutput,
    ) -> Result<(), Self::BaseError>;

    fn add_scalar(&mut self, size: usize, alpha: f32, input: &Self) -> Result<(), Self::BaseError>;

    fn abs_pow_scalar(&mut self, size: usize, alpha: f32, input: &Self) -> Result<(), Self::BaseError>;

    fn abs_pow_scalar_backward(
        &mut self,
        size: usize,
        alpha: f32,
        input: &Self,
        grd: &Self,
    ) -> Result<(), Self::BaseError>;

    fn pairwise_fwd(&mut self, size: usize, batch_size: usize, a: &Self) -> Result<(), Self::BaseError>;

    fn pairwise_bwd(&mut self, size: usize, batch_size: usize, a: &Self, grd: &Self) -> Result<(), Self::BaseError>;

    fn power_error_fwd(&mut self, power: f32, size: usize, a: &Self, b: &Self) -> Result<(), Self::BaseError>;

    fn power_error_bwd(
        &mut self,
        power: f32,
        size: usize,
        a: &Self,
        b: &Self,
        grd: &Self,
    ) -> Result<(), Self::BaseError>;

    fn clip(&mut self, size: usize, min: f32, max: f32) -> Result<(), Self::BaseError>;

    fn adam(
        &mut self,
        config: &AdamConfig,
        size: usize,
        grd: &Self,
        mom: &mut Self,
        vel: &mut Self,
    ) -> Result<(), Self::BaseError>;
}
