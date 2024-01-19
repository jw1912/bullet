use bullet::{
    inputs, Activation, TrainerBuilder
};

const HIDDEN_SIZE: usize = 256;
const QA: i32 = 181;
const QB: i32 = 64;

fn main() {
    let trainer = TrainerBuilder::default()
        .quantisations(&[QA, QB])
        .input(inputs::Chess768)
        .feature_transformer(HIDDEN_SIZE)
        .activate(Activation::SCReLU)
        .add_layer(1)
        .build();

    trainer.load_from_checkpoint("checkpoints/my-checkpoint");
    trainer.save_quantised("requantised.bin");
}