use bullet::{
    inputs, Activation, LrScheduler, TrainerBuilder, TrainingSchedule, WdlScheduler,
};

const HIDDEN_SIZE: usize = 768;
const SCALE: i32 = 400;
const QA: i32 = 181;
const QB: i32 = 64;

fn main() {
    let mut trainer = TrainerBuilder::default()
        .set_batch_size(16_384)
        .set_eval_scale(SCALE as f32)
        .set_quantisations(&[QA, QB])
        .set_input(inputs::Chess768)
        .ft(HIDDEN_SIZE)
        .activate(Activation::SCReLU)
        .add_layer(1)
        .build();

    let schedule = TrainingSchedule {
        net_id: "net-01.01.24".to_string(),
        start_epoch: 1,
        end_epoch: 17,
        wdl_scheduler: WdlScheduler::Linear {
            start: 0.2,
            end: 0.5,
        },
        lr_scheduler: LrScheduler::Step {
            start: 0.001,
            gamma: 0.1,
            step: 8,
        },
        save_rate: 1,
    };

    trainer.run(
        &schedule,
        4,
        "../../data/akimbo3-9.data",
        "checkpoints",
    );
}

/*
Inference Example Notes:
- For speed, you would want to convert these weights to i16s, as for
    this specific architecture they won't cause any overflow issues.
- ReLU and CReLU do not introduce the additional quantisation factor
    that SCReLU does.
- `QA = 181` to utilise an additional trick with manual SIMD, not included
    here and the compiler will not be able to do it automatically.

This is how you would load the network in rust.
Commented out because it will error if it can't find the file.
static NNUE: Network =
    unsafe { std::mem::transmute(*include_bytes!("../resources/net-epoch30.bin")) };
*/

#[inline]
/// Squared Clipped ReLU - Activation Function
fn screlu(x: i32) -> i32 {
    x.clamp(0, QA).pow(2)
}

/// This is the quantised format that bullet outputs.
#[repr(C)]
pub struct Network {
    /// Column-Major `HIDDEN_SIZE x 768` matrix.
    feature_weights: [Accumulator; 768],
    /// Vector with dimension `HIDDEN_SIZE`.
    feature_bias: Accumulator,
    /// Column-Major `1 x (2 * HIDDEN_SIZE)`
    /// matrix, we use it like this to make the
    /// code nicer in `Network::evaluate`.
    output_weights: [Accumulator; 2],
    output_bias: i32,
}

impl Network {
    /// Calculates the output of the network, starting from the already
    /// calculated hidden layer (done efficiently during makemoves).
    pub fn evaluate(&self, us: &Accumulator, them: &Accumulator) -> i32 {
        let mut output = 0;

        // Side-To-Move Accumulator -> Output
        for (&input, &weight) in us.vals.iter().zip(&self.output_weights[0].vals) {
            output += screlu(input) * weight;
        }

        // Not-Side-To-Move Accumulator -> Output
        for (&input, &weight) in them.vals.iter().zip(&self.output_weights[1].vals) {
            output += screlu(input) * weight;
        }

        // SCReLU introduces an additional factor of `QA`,
        // which must be removed before adding the bias
        output /= QA;

        // Add the bias
        output += self.output_bias;

        // Apply eval scale
        output *= SCALE;

        // Remove quantisation
        output /= QA * QB;

        output
    }
}

/// A column of the feature-weights matrix.
/// Note the `align(64)`.
#[derive(Clone, Copy)]
#[repr(C, align(64))]
pub struct Accumulator {
    vals: [i32; HIDDEN_SIZE],
}

impl Accumulator {
    /// Initialised with bias so we can just efficiently
    /// operate on it afterwards.
    pub fn new(net: &Network) -> Self {
        net.feature_bias
    }

    /// Add a feature to an accumulator.
    pub fn add_feature(&mut self, feature_idx: usize, net: &Network) {
        for (i, d) in self
            .vals
            .iter_mut()
            .zip(&net.feature_weights[feature_idx].vals)
        {
            *i += *d
        }
    }

    /// Remove a feature from an accumulator.
    pub fn remove_feature(&mut self, feature_idx: usize, net: &Network) {
        for (i, d) in self
            .vals
            .iter_mut()
            .zip(&net.feature_weights[feature_idx].vals)
        {
            *i -= *d
        }
    }
}
