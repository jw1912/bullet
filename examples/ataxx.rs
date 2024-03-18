use bullet_lib::{
    format::AtaxxBoard,
    inputs::InputType, outputs, Activation, LocalSettings, LrScheduler, TrainerBuilder, TrainingSchedule,
    WdlScheduler,
};

const HIDDEN_SIZE: usize = 128;
const PER_TUPLE: usize = 3usize.pow(4);

#[derive(Clone, Copy, Default)]
pub struct Ataxx2Tuples;
impl InputType for Ataxx2Tuples {
    type RequiredDataType = AtaxxBoard;
    type FeatureIter = ThisIterator;

    fn max_active_inputs(&self) -> usize {
        36
    }

    fn buckets(&self) -> usize {
        1
    }

    fn inputs(&self) -> usize {
        self.max_active_inputs() * PER_TUPLE
    }

    fn feature_iter(&self, pos: &Self::RequiredDataType) -> Self::FeatureIter {
        let mut res = [(0, 0); 36];

        let [boys, opps, _] = pos.bbs();

        for i in 0..6 {
            for j in 0..6 {
                const POWERS: [usize; 4] = [1, 3, 9, 27];
                const MASK: u64 = 0b0001_1000_0011;

                let tuple = 6 * i + j;
                let mut feat = PER_TUPLE * tuple;

                let offset = 7 * i + j;
                let mut b = (boys >> offset) & MASK;
                let mut o = (opps >> offset) & MASK;

                while b > 0 {
                    let mut sq = b.trailing_zeros() as usize;
                    if sq > 6 {
                        sq -= 5;
                    }

                    feat += POWERS[sq];

                    b &= b - 1;
                }

                while o > 0 {
                    let mut sq = o.trailing_zeros() as usize;
                    if sq > 6 {
                        sq -= 5;
                    }

                    feat += 2 * POWERS[sq];

                    o &= o - 1;
                }

                res[tuple] = (feat, feat);
            }
        }

        ThisIterator {
            inner: res,
            index: 0,
        }
    }
}

pub struct ThisIterator {
    inner: [(usize, usize); 36],
    index: usize,
}

impl Iterator for ThisIterator {
    type Item = (usize, usize);

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= 25 {
            return None;
        }

        let res = self.inner[self.index];
        self.index += 1;
        Some(res)
    }
}

fn main() {
    let mut trainer = TrainerBuilder::default()
        .single_perspective()
        .quantisations(&[255, 64])
        .input(Ataxx2Tuples)
        .output_buckets(outputs::Single)
        .feature_transformer(HIDDEN_SIZE)
        .activate(Activation::SCReLU)
        .add_layer(1)
        .build();

    let schedule = TrainingSchedule {
        net_id: "net005".to_string(),
        batch_size: 16_384,
        eval_scale: 400.0,
        batches_per_superbatch: 6104,
        start_superbatch: 1,
        end_superbatch: 40,
        wdl_scheduler: WdlScheduler::Constant { value: 0.5 },
        lr_scheduler: LrScheduler::Step {
            start: 0.001,
            gamma: 0.1,
            step: 15,
        },
        save_rate: 10,
    };

    let settings = LocalSettings {
        threads: 4,
        data_file_paths: vec!["../../data/ataxx/005.data"],
        output_directory: "checkpoints",
    };

    trainer.run(&schedule, &settings);

    println!("{}", trainer.eval("x5o/7/7/7/7/7/o5x x 0 1"));
    println!("{}", trainer.eval("5oo/7/x6/x6/7/7/o5x o 0 2"));
}
