use bullet_lib::{
    nn::{optimiser, Activation},
    trainer::{
        default::{formats::bulletformat::AtaxxBoard, inputs::SparseInputType, loader, Loss, TrainerBuilder},
        schedule::{lr, wdl, TrainingSchedule, TrainingSteps},
        settings::LocalSettings,
    },
};

const HIDDEN_SIZE: usize = 128;
const PER_TUPLE: usize = 3usize.pow(4);
const NUM_TUPLES: usize = 36;

#[derive(Clone, Copy, Default)]
pub struct Ataxx2Tuples;
impl SparseInputType for Ataxx2Tuples {
    type RequiredDataType = AtaxxBoard;

    fn num_inputs(&self) -> usize {
        NUM_TUPLES * PER_TUPLE
    }

    fn max_active(&self) -> usize {
        NUM_TUPLES
    }

    fn map_features<F: FnMut(usize, usize)>(&self, pos: &Self::RequiredDataType, mut f: F) {
        let [boys, opps, _] = pos.bbs();

        for i in 0..6 {
            for j in 0..6 {
                const POWERS: [usize; 4] = [1, 3, 9, 27];
                const MASK: u64 = 0b0001_1000_0011;

                let tuple = 6 * i + j;
                let mut stm = PER_TUPLE * tuple;
                let mut ntm = stm;

                let offset = 7 * i + j;
                let mut b = (boys >> offset) & MASK;
                let mut o = (opps >> offset) & MASK;

                while b > 0 {
                    let mut sq = b.trailing_zeros() as usize;
                    if sq > 6 {
                        sq -= 5;
                    }

                    stm += POWERS[sq];
                    ntm += 2 * POWERS[sq];

                    b &= b - 1;
                }

                while o > 0 {
                    let mut sq = o.trailing_zeros() as usize;
                    if sq > 6 {
                        sq -= 5;
                    }

                    stm += 2 * POWERS[sq];
                    ntm += POWERS[sq];

                    o &= o - 1;
                }

                f(stm, ntm);
            }
        }
    }

    fn shorthand(&self) -> String {
        "2-tuples".to_string()
    }

    fn description(&self) -> String {
        "Ataxx 2x2-typles".to_string()
    }
}

fn main() {
    let mut trainer = TrainerBuilder::default()
        .single_perspective()
        .quantisations(&[255, 64])
        .optimiser(optimiser::AdamW)
        .loss_fn(Loss::SigmoidMSE)
        .input(Ataxx2Tuples)
        .feature_transformer(HIDDEN_SIZE)
        .activate(Activation::SCReLU)
        .add_layer(1)
        .build();

    let schedule = TrainingSchedule {
        net_id: "net006".to_string(),
        eval_scale: 400.0,
        steps: TrainingSteps {
            batch_size: 16_384,
            batches_per_superbatch: 6104,
            start_superbatch: 1,
            end_superbatch: 40,
        },
        wdl_scheduler: wdl::ConstantWDL { value: 0.5 },
        lr_scheduler: lr::StepLR { start: 0.001, gamma: 0.1, step: 15 },
        save_rate: 10,
    };

    trainer.set_optimiser_params(optimiser::AdamWParams::default());

    let settings = LocalSettings { threads: 4, test_set: None, output_directory: "checkpoints", batch_queue_size: 512 };

    let data_loader = loader::DirectSequentialDataLoader::new(&["../../data/ataxx/005.data"]);

    trainer.run(&schedule, &settings, &data_loader);

    println!("{}", 400.0 * trainer.eval("x5o/7/7/7/7/7/o5x x 0 1"));
    println!("{}", 400.0 * trainer.eval("5oo/7/x6/x6/7/7/o5x o 0 2"));
}
