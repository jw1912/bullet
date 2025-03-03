/// List of supported activation functions.
#[repr(i32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Activation {
    Identity = 0,
    ReLU = 1,
    CReLU = 2,
    SCReLU = 3,
    SqrReLU = 4,
    Sigmoid = 5,
    Square = 6,
}

#[derive(Clone, Copy, Debug)]
pub struct AdamConfig {
    pub beta1: f32,
    pub beta2: f32,
    pub gradient_factor: f32,
    pub learning_rate: f32,
    pub denom: bool,
}

impl AdamConfig {
    pub fn new(beta1: f32, beta2: f32, gradient_factor: f32, learning_rate: f32, denom: bool) -> Self {
        Self { beta1, beta2, gradient_factor, learning_rate, denom }
    }
}

pub trait BaseOperations {
    type BaseError;

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

    fn activate_fwd(&mut self, size: usize, a: &Self, act: Activation) -> Result<(), Self::BaseError>;

    fn activate_bwd(&mut self, size: usize, a: &Self, grd: &Self, act: Activation) -> Result<(), Self::BaseError>;

    fn pairwise_fwd(
        &mut self,
        size: usize,
        batch_size: usize,
        a: &Self,
        //post_concat: bool,
    ) -> Result<(), Self::BaseError>;

    fn pairwise_bwd(
        &mut self,
        size: usize,
        batch_size: usize,
        a: &Self,
        grd: &Self,
        //post_concat: bool,
    ) -> Result<(), Self::BaseError>;

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
