use bullet::{TrainerBuilder, run_training};
use bullet_core::inputs::Chess768;
use bullet_tensor::{Activation, device_synchronise};

fn main() {
    let mut net = TrainerBuilder::<Chess768>::default()
        .set_batch_size(16_384)
        .ft(32)
        .activate(Activation::ReLU)
        .add_layer(1)
        .build();

    device_synchronise();

    println!("Network Architecture: {net}");

    run_training(&mut net, 4, 5, 400.0, "../../data/10m.data");
}