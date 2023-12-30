use bullet::{TrainerBuilder, run_training};
use bullet_core::inputs::Chess768;
use bullet_tensor::{Activation, device_synchronise};

static INC: [f32; 24673] = unsafe {
    std::mem::transmute(*include_bytes!("../checkpoints/blah-epoch1/params.bin"))
};

fn main() {
    let mut net = TrainerBuilder::<Chess768>::default()
        .set_batch_size(16_384)
        .ft(32)
        .activate(Activation::ReLU)
        .add_layer(1)
        .build();

    device_synchronise();

    println!("Network Architecture: {net}");

    net.optimiser.load_weights_from_cpu(&INC);

    run_training(&mut net, 4, 1, 400.0, "../../data/batch.data");
}