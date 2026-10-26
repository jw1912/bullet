pub(crate) mod builder;
pub mod loader;
pub mod save;

use std::cell::RefCell;

pub use builder::{NoOutputBuckets, ValueTrainerBuilder, split_inputs};
use bullet_compiler::tensor::TValue;
use bullet_trainer::{
    model::{ModelEvaluator, ModelInputs, ModelInputsMapper, SavedFormat},
    optimiser::{Optimiser, OptimiserState},
    reader::{DataReader, ReadMapLoader},
    run::{self, HostPool, Step, logger},
};

use crate::{
    game::{
        inputs::{InputSplit, SparseInputType},
        outputs::OutputBuckets,
    },
    nn::ExecutionContext,
    trainer::{
        schedule::{TrainingSchedule, lr::LrScheduler, wdl::WdlScheduler},
        settings::LocalSettings,
    },
    wdl,
};

use loader::LoadableDataType;

/// Value network trainer, generally for training NNUE networks.
pub struct ValueTrainer<Opt: OptimiserState<ExecutionContext>, Inp: SparseInputType, Out> {
    pub optimiser: Optimiser<ExecutionContext, Opt>,
    state: ValueTrainerState<Inp, Out>,
    evaluator: Option<ModelEvaluator<ExecutionContext>>,
}

type B<I> = fn(&<I as SparseInputType>::RequiredDataType, f32) -> f32;
type Wgt<I> = fn(&<I as SparseInputType>::RequiredDataType) -> f32;

#[derive(Clone)]
pub struct ValueTrainerState<Inp: SparseInputType, Out> {
    input_getter: Inp,
    output_getter: Out,
    blend_getter: B<Inp>,
    weight_getter: Option<Wgt<Inp>>,
    saved_format: Vec<SavedFormat>,
    use_win_rate_model: bool,
    wdl: bool,
}

impl<I, O> ValueTrainerState<I, O>
where
    I: SparseInputType,
    I::RequiredDataType: LoadableDataType,
    O: OutputBuckets<I::RequiredDataType>,
{
    fn make_mapper(&self, scale: f32, wdl: impl WdlScheduler) -> ModelInputsMapper<I::RequiredDataType> {
        let nnz = self.input_getter.max_active();
        let num = self.input_getter.num_inputs();
        let inp = self.input_getter.clone();
        let out = self.output_getter;
        let wget = self.weight_getter;
        let target_wdl = self.wdl;
        let blend_getter = self.blend_getter;
        let use_win_rate_model = self.use_win_rate_model;
        let rscale = 1.0 / scale;

        fn sigmoid(x: f32) -> f32 {
            1. / (1. + (-x).exp())
        }

        // Everything but the sparse features: output bucket, entry weight, targets.
        let fill_rest = move |pos: &I::RequiredDataType,
                              step: Step,
                              buckets: &mut [i32],
                              targets: &mut [f32],
                              weights: &mut [f32]| {
            buckets[0] = i32::from(out.bucket(pos));
            weights[0] = wget.map_or(1.0, |w| w(pos));

            if target_wdl {
                for target in targets.iter_mut() {
                    *target = 0.0;
                }

                targets[usize::from(pos.result() as u8)] = 1.0;
            } else {
                let score = f32::from(pos.score());
                let score = if use_win_rate_model {
                    let p = (score - 270.0) / 380.0;
                    let pm = (-score - 270.0) / 380.0;
                    0.5 * (1.0 + sigmoid(p) - sigmoid(pm))
                } else {
                    sigmoid(rscale * score)
                };

                let result = f32::from(pos.result() as u8) / 2.0;
                let blend = blend_getter(pos, wdl.blend(step.batch(), step.superbatch(), step.final_superbatch()));
                assert!((0.0..=1.0).contains(&blend), "WDL proportion must be in [0, 1]");
                targets[0] = blend * result + (1. - blend) * score;
            }
        };

        if let Some(split) = inp.split() {
            let (at, nnz1, nnz2) = (split.at, split.max_active_first, split.max_active_second);
            let inputs = ModelInputs::default()
                .add_sparse("stm", (at, 1), nnz1)
                .add_sparse("nstm", (at, 1), nnz1)
                .add_sparse("stm2", (num - at, 1), nnz2)
                .add_sparse("nstm2", (num - at, 1), nnz2)
                .add_sparse("buckets", (1, 1), 1)
                .add_dense("targets", (if target_wdl { 3 } else { 1 }, 1))
                .add_dense("entry_weights", (1, 1));

            return ModelInputsMapper::build(
                &inputs,
                move |pos, step, ((((((stm, ntm), stm2), ntm2), buckets), targets), weights)| {
                    map_split(&inp, pos, split, (stm, ntm), (stm2, ntm2));
                    fill_rest(pos, step, buckets, targets, weights);
                },
            );
        }

        let inputs = ModelInputs::default()
            .add_sparse("stm", (num, 1), nnz)
            .add_sparse("nstm", (num, 1), nnz)
            .add_sparse("buckets", (1, 1), 1)
            .add_dense("targets", (if target_wdl { 3 } else { 1 }, 1))
            .add_dense("entry_weights", (1, 1));

        ModelInputsMapper::build(&inputs, move |pos, step, ((((stm, ntm), buckets), targets), weights)| {
            let mut cnt = 0;
            inp.map_features(pos, |our, opp| {
                assert!(our < num && opp < num, "Input feature index exceeded input size!");
                stm[cnt] = our as i32;
                ntm[cnt] = opp as i32;
                cnt += 1;
            });

            if cnt < nnz {
                stm[cnt] = -1;
                ntm[cnt] = -1;
            }

            assert!(cnt <= nnz, "More inputs provided than the specified maximum!");

            fill_rest(pos, step, buckets, targets, weights);
        })
    }

    pub fn make_read_map_loader<D>(
        &self,
        reader: D,
        scale: f32,
        wdl: impl WdlScheduler,
        threads: u8,
    ) -> ReadMapLoader<D, I::RequiredDataType>
    where
        D: DataReader<I::RequiredDataType>,
    {
        ReadMapLoader::new(reader, self.make_mapper(scale, wdl), threads)
    }
}

/// Writes the features of `pos` into the sparse input buffers of the two groups
/// of `split`, second-group indices relative to `split.at`, each list ended by -1
/// when shorter than its maximum.
fn map_split<I: SparseInputType>(
    inp: &I,
    pos: &I::RequiredDataType,
    split: InputSplit,
    (stm, ntm): (&mut [i32], &mut [i32]),
    (stm2, ntm2): (&mut [i32], &mut [i32]),
) {
    let (at, nnz1, nnz2, num) = (split.at, split.max_active_first, split.max_active_second, inp.num_inputs());
    let (mut c1, mut c2) = (0, 0);
    inp.map_features(pos, |our, opp| {
        assert!(our < num && opp < num, "Input feature index exceeded input size!");
        assert_eq!(our >= at, opp >= at, "Perspectives disagree on the feature group!");
        if our < at {
            assert!(c1 < nnz1, "More first-group inputs provided than the specified maximum!");
            stm[c1] = our as i32;
            ntm[c1] = opp as i32;
            c1 += 1;
        } else {
            assert!(c2 < nnz2, "More second-group inputs provided than the specified maximum!");
            stm2[c2] = (our - at) as i32;
            ntm2[c2] = (opp - at) as i32;
            c2 += 1;
        }
    });
    if c1 < nnz1 {
        stm[c1] = -1;
        ntm[c1] = -1;
    }
    if c2 < nnz2 {
        stm2[c2] = -1;
        ntm2[c2] = -1;
    }
}

impl<Opt, Inp, Out> ValueTrainer<Opt, Inp, Out>
where
    Opt: OptimiserState<ExecutionContext>,
    Inp: SparseInputType,
    Inp::RequiredDataType: LoadableDataType,
    Out: OutputBuckets<Inp::RequiredDataType>,
{
    pub fn run(
        &mut self,
        schedule: &TrainingSchedule<impl LrScheduler, impl WdlScheduler>,
        settings: &LocalSettings,
        dataloader: &impl DataReader<Inp::RequiredDataType>,
    ) {
        logger::clear_colours();
        println!("{}", logger::ansi("Training Preamble", "34;1"));

        schedule.display();
        settings.display();

        if settings.test_set.is_some() {
            println!(
                "{}",
                logger::ansi("Warning: Validation data not currently implemented! Please bother me on discord.", "31")
            )
        }

        let steps = schedule.steps;

        let dataloader = self.state.make_read_map_loader(
            dataloader.clone(),
            schedule.eval_scale,
            schedule.wdl_scheduler.clone(),
            settings.loader_threads(),
        );

        let _ = std::fs::create_dir(settings.output_directory);

        let lr_scheduler = schedule.lr_scheduler.clone();
        let saved_format = self.state.saved_format.clone();

        let error_record = RefCell::new(Vec::new());
        let mut loss_sum = 0.0;
        let mut ticks_since_last = 0.0;

        run::train(
            &mut self.optimiser,
            run::TrainingSchedule { steps, log_rate: 128, lr_schedule: lr_scheduler.boxed(), track_grad_norms: false },
            dataloader,
            |_, step, error| {
                loss_sum += error;
                ticks_since_last += 1.0;

                if step.batch().is_multiple_of(32) || step.batch() + 1 == step.batches_per_superbatch() {
                    let normalised_loss = loss_sum / ticks_since_last;

                    error_record.borrow_mut().push((step.superbatch(), step.batch(), normalised_loss));

                    loss_sum = 0.0;
                    ticks_since_last = 0.0;
                }
            },
            |trainer, step| {
                let superbatch = step.superbatch();
                if superbatch.is_multiple_of(schedule.save_rate) || superbatch == step.final_superbatch() {
                    let name = format!("{}-{superbatch}", schedule.net_id);
                    let path = format!("{}/{name}", settings.output_directory);
                    std::fs::create_dir(path.as_str()).unwrap_or(());
                    save::save_to_checkpoint(trainer, &saved_format, &path);
                    save::write_losses(&format!("{path}/log.txt"), &error_record.borrow());

                    println!("Saved [{}]", logger::ansi(name, 31));
                }
            },
        )
        .unwrap();
    }

    pub fn eval_raw_output(&mut self, fen: &str) -> Vec<f32>
    where
        Inp::RequiredDataType: std::str::FromStr<Err: std::fmt::Debug>,
    {
        let pos = format!("{fen} | 0 | 0.0").parse::<Inp::RequiredDataType>().unwrap();

        let mapper = self.state.make_mapper(1.0, wdl::ConstantWDL { value: 1.0 });
        let pool = HostPool::new(self.optimiser.device());
        let host_data = mapper.map(&pool, &[pos], Step::default(), 1).unwrap();

        let device_data = host_data.to_device(&self.optimiser.device()).unwrap();

        if self.evaluator.is_none() {
            let mut evaluator = ModelEvaluator::new(self.optimiser.definition(), self.optimiser.device()).unwrap();
            evaluator.load_device_weights(self.optimiser.weights()).unwrap();
            self.evaluator = Some(evaluator);
        }

        let outputs = self.evaluator.as_mut().unwrap().evaluate(&device_data).unwrap();

        let output = outputs.get("output").unwrap().clone();
        let TValue::F32(output) = output.to_host().unwrap() else { panic!() };
        output
    }

    pub fn eval(&mut self, fen: &str) -> f32
    where
        Inp::RequiredDataType: std::str::FromStr<Err: std::fmt::Debug>,
    {
        let vals = self.eval_raw_output(fen);

        match vals[..] {
            [mut loss, mut draw, mut win] => {
                let max = win.max(draw).max(loss);
                win = (win - max).exp();
                draw = (draw - max).exp();
                loss = (loss - max).exp();

                (win + draw / 2.0) / (win + draw + loss)
            }
            [score] => score,
            _ => panic!("Invalid output size!"),
        }
    }

    pub fn measure_max_cpu_throughput(
        &self,
        schedule: &TrainingSchedule<impl LrScheduler, impl WdlScheduler>,
        settings: &LocalSettings,
        dataloader: &impl DataReader<Inp::RequiredDataType>,
    ) {
        let steps = schedule.steps;
        let dataloader = self.state.make_read_map_loader(
            dataloader.clone(),
            schedule.eval_scale,
            schedule.wdl_scheduler.clone(),
            settings.loader_threads(),
        );

        run::measure_max_cpu_throughput(dataloader, steps).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::formats::bulletformat::ChessBoard;

    /// Features 0..10, split at 6; the dummy emits 2, 7, 3, 9 for both perspectives.
    #[derive(Clone, Copy)]
    struct Dummy;

    impl SparseInputType for Dummy {
        type RequiredDataType = ChessBoard;
        fn num_inputs(&self) -> usize {
            10
        }
        fn max_active(&self) -> usize {
            6
        }
        fn map_features<F: FnMut(usize, usize)>(&self, _: &ChessBoard, mut f: F) {
            for i in [2, 7, 3, 9] {
                f(i, i);
            }
        }
        fn shorthand(&self) -> String {
            "dummy".into()
        }
        fn description(&self) -> String {
            "dummy".into()
        }
        fn split(&self) -> Option<InputSplit> {
            Some(InputSplit { at: 6, max_active_first: 3, max_active_second: 3 })
        }
    }

    #[test]
    fn split_routes_features() {
        let (mut stm, mut ntm, mut stm2, mut ntm2) = ([9; 3], [9; 3], [9; 3], [9; 3]);
        map_split(&Dummy, &ChessBoard::default(), Dummy.split().unwrap(), (&mut stm, &mut ntm), (&mut stm2, &mut ntm2));
        assert_eq!(stm, [2, 3, -1]);
        assert_eq!(ntm, [2, 3, -1]);
        assert_eq!(stm2, [1, 3, -1]);
        assert_eq!(ntm2, [1, 3, -1]);
    }
}
