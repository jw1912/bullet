pub(crate) mod builder;
mod dataloader;
pub mod loader;
mod save;

use std::cell::RefCell;

pub use builder::{NoOutputBuckets, ValueTrainerBuilder};

use acyclib::{
    graph::Node,
    trainer::{self, Trainer, logger, optimiser::OptimiserState},
};

use acyclib::{
    graph::{GraphNodeId, GraphNodeIdTy, like::GraphLike, save::SavedFormat},
    trainer::dataloader::{PreparedBatchDevice, PreparedBatchHost},
};

use crate::{
    game::{inputs::SparseInputType, outputs::OutputBuckets},
    nn::{ExecutionContext, Graph},
    trainer::{
        schedule::{TrainingSchedule, lr::LrScheduler, wdl::WdlScheduler},
        settings::LocalSettings,
    },
    value::{
        dataloader::ValueDataLoader,
        loader::{DefaultDataLoader, LoadableDataType},
    },
};

use crate::value::loader::PreparedData;

/// Value network trainer, generally for training NNUE networks.
pub struct ValueTrainer<
    Opt: OptimiserState<ExecutionContext>,
    Inp: SparseInputType,
    Out: OutputBuckets<Inp::RequiredDataType>,
>(Trainer<ExecutionContext, Graph, Opt, ValueTrainerState<Inp, Out>>);

impl<Opt, Inp, Out> std::ops::Deref for ValueTrainer<Opt, Inp, Out>
where
    Opt: OptimiserState<ExecutionContext>,
    Inp: SparseInputType,
    Out: OutputBuckets<Inp::RequiredDataType>,
{
    type Target = Trainer<ExecutionContext, Graph, Opt, ValueTrainerState<Inp, Out>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<Opt, Inp, Out> std::ops::DerefMut for ValueTrainer<Opt, Inp, Out>
where
    Opt: OptimiserState<ExecutionContext>,
    Inp: SparseInputType,
    Out: OutputBuckets<Inp::RequiredDataType>,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

type B<I> = fn(&<I as SparseInputType>::RequiredDataType, f32) -> f32;
type Wgt<I> = fn(&<I as SparseInputType>::RequiredDataType) -> f32;

#[derive(Clone)]
pub struct ValueTrainerState<Inp: SparseInputType, Out> {
    input_getter: Inp,
    output_getter: Out,
    blend_getter: B<Inp>,
    weight_getter: Option<Wgt<Inp>>,
    output_node: Node,
    saved_format: Vec<SavedFormat>,
    use_win_rate_model: bool,
    wdl: bool,
}

impl<Inp: SparseInputType, Out> ValueTrainerState<Inp, Out>
where
    Inp: SparseInputType,
    Inp::RequiredDataType: LoadableDataType,
    Out: OutputBuckets<Inp::RequiredDataType>,
{
    pub fn prepare(
        &self,
        batch: &[Inp::RequiredDataType],
        threads: usize,
        blend: f32,
        scale: f32,
    ) -> PreparedBatchHost {
        PreparedBatchHost::from(PreparedData::new(
            self.input_getter.clone(),
            self.output_getter,
            self.blend_getter,
            self.weight_getter,
            self.use_win_rate_model,
            self.wdl,
            batch,
            threads,
            blend,
            scale,
        ))
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
        dataloader: &impl loader::DataLoader<Inp::RequiredDataType>,
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

        let dataloader = DefaultDataLoader::new(
            self.state.input_getter.clone(),
            self.state.output_getter,
            self.state.blend_getter,
            self.state.weight_getter,
            self.state.use_win_rate_model,
            self.state.wdl,
            schedule.eval_scale,
            dataloader.clone(),
        );

        let _ = std::fs::create_dir(settings.output_directory);

        let lr_scheduler = schedule.lr_scheduler.clone();

        let steps = schedule.steps;

        let error_record = RefCell::new(Vec::new());
        let mut loss_sum = 0.0;
        let mut ticks_since_last = 0.0;

        self.train_custom(
            trainer::schedule::TrainingSchedule {
                steps,
                log_rate: 128,
                lr_schedule: Box::new(|a, b| lr_scheduler.lr(a, b)),
            },
            ValueDataLoader { steps, threads: settings.threads, dataloader, wdl: schedule.wdl_scheduler.clone() },
            |_, superbatch, curr_batch, error| {
                loss_sum += error;
                ticks_since_last += 1.0;

                if curr_batch % 32 == 0
                    || (steps.batches_per_superbatch < 32 && curr_batch == steps.batches_per_superbatch)
                {
                    let normalised_loss = loss_sum / f32::min(ticks_since_last, steps.batches_per_superbatch as f32);

                    error_record.borrow_mut().push((superbatch, curr_batch, normalised_loss));

                    loss_sum = 0.0;
                    ticks_since_last = 0.0;
                }
            },
            |trainer, superbatch| {
                if superbatch % schedule.save_rate == 0 || superbatch == steps.end_superbatch {
                    let name = format!("{}-{superbatch}", schedule.net_id);
                    let path = format!("{}/{name}", settings.output_directory);
                    std::fs::create_dir(path.as_str()).unwrap_or(());
                    save::save_to_checkpoint(trainer, &path);
                    save::write_losses(&format!("{path}/log.txt"), &error_record.borrow());

                    println!("Saved [{}]", logger::ansi(name, 31));
                }
            },
        )
        .unwrap();
    }

    pub fn get_output_values(&self) -> Vec<f32> {
        let id = GraphNodeId::new(self.state.output_node.idx(), GraphNodeIdTy::Values);

        #[cfg(not(any(feature = "multigpu", feature = "cpu")))]
        {
            self.optimiser.graph.get(id).unwrap().get_dense_vals().unwrap()
        }

        #[cfg(any(feature = "multigpu", feature = "cpu"))]
        self.optimiser.graph.get_all(id).unwrap().iter().flat_map(|x| x.get_dense_vals().unwrap()).collect()
    }

    pub fn eval_raw_output(&mut self, fen: &str) -> Vec<f32>
    where
        Inp::RequiredDataType: std::str::FromStr<Err: std::fmt::Debug> + LoadableDataType,
    {
        let pos = format!("{fen} | 0 | 0.0").parse::<Inp::RequiredDataType>().unwrap();

        let host_data = self.state.prepare(&[pos], 1, 1.0, 1.0);

        #[cfg(not(any(feature = "multigpu", feature = "cpu")))]
        let graph = &mut self.optimiser.graph;

        #[cfg(any(feature = "multigpu", feature = "cpu"))]
        let graph = self.optimiser.graph.primary_mut();

        let mut device_data = PreparedBatchDevice::new(graph.devices(), &host_data).unwrap();

        device_data.load_into_graph(graph).unwrap();

        graph.synchronise().unwrap();
        graph.forward().unwrap();

        self.get_output_values()
    }

    pub fn eval(&mut self, fen: &str) -> f32
    where
        Inp::RequiredDataType: std::str::FromStr<Err: std::fmt::Debug> + LoadableDataType,
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
        dataloader: &impl loader::DataLoader<Inp::RequiredDataType>,
    ) {
        let steps = schedule.steps;
        let threads = settings.threads;
        let wdl = schedule.wdl_scheduler.clone();
        let dataloader = DefaultDataLoader::new(
            self.state.input_getter.clone(),
            self.state.output_getter,
            self.state.blend_getter,
            self.state.weight_getter,
            self.state.use_win_rate_model,
            self.state.wdl,
            schedule.eval_scale,
            dataloader.clone(),
        );

        let dataloader = ValueDataLoader { steps, threads, dataloader, wdl };

        self.0.measure_max_cpu_throughput(dataloader, steps).unwrap()
    }
}
