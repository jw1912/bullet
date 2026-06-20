pub(crate) mod builder;
pub mod loader;
pub mod save;

use std::cell::RefCell;

pub use builder::{NoOutputBuckets, ValueTrainerBuilder};
use bullet_compiler::tensor::TValue;
use bullet_trainer::{
    model::{ModelEvaluator, ModelInputs, ModelInputsMapper, SavedFormat},
    optimiser::{Optimiser, OptimiserState},
    reader::{DataReader, ReadMapLoader},
    run::{self, Step, logger},
};

use crate::{
    game::{inputs::SparseInputType, outputs::OutputBuckets},
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
            settings.threads as u8,
        );

        let _ = std::fs::create_dir(settings.output_directory);

        let lr_scheduler = schedule.lr_scheduler.clone();
        let saved_format = self.state.saved_format.clone();

        let error_record = RefCell::new(Vec::new());
        let mut loss_sum = 0.0;
        let mut ticks_since_last = 0.0;

        run::train(
            &mut self.optimiser,
            run::TrainingSchedule { steps, log_rate: 128, lr_schedule: lr_scheduler.boxed() },
            dataloader,
            |_, step, error| {
                loss_sum += error;
                ticks_since_last += 1.0;

                if step.batch().is_multiple_of(32)
                    || (step.batches_per_superbatch() < 32 && step.batch() == step.batches_per_superbatch())
                {
                    let normalised_loss = loss_sum / f32::min(ticks_since_last, step.batches_per_superbatch() as f32);

                    error_record.borrow_mut().push((step.superbatch(), step.batch(), normalised_loss));

                    loss_sum = 0.0;
                    ticks_since_last = 0.0;
                }
            },
            |trainer, step| {
                let superbatch = step.superbatch();
                if superbatch % schedule.save_rate == 0 || superbatch == step.final_superbatch() {
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
        let host_data = mapper.map(&[pos], Step::default(), 1);

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
            settings.threads as u8,
        );

        run::measure_max_cpu_throughput(dataloader, steps).unwrap()
    }
}
