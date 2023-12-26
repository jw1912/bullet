mod trainer;

pub use bullet_core::inputs::Chess768;
pub use bullet_tensor::Activation;
pub use trainer::{Trainer, TrainerBuilder};

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_build() {
        let net = TrainerBuilder::<Chess768>::default()
            .set_batch_size(16_384)
            .ft(512)
            .activate(Activation::SCReLU)
            .add_layer(1)
            .build();

        println!("{net}");
    }
}
