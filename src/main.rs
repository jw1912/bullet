use bullet::{TrainerBuilder, run_training};
use bullet_core::inputs::Chess768;
use bullet_tensor::{Activation, device_synchronise};

fn main() {
    let mut net = TrainerBuilder::<Chess768>::default()
        .set_batch_size(16_384)
        .ft(512)
        .activate(Activation::SCReLU)
        .add_layer(1)
        .build();

    device_synchronise();

    println!("Network Architecture: {net}");

    run_training(&mut net, 4, 10, 400.0, "../../data/wha.data");
}