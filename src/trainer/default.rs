mod builder;
pub mod cutechess;
/// Contains the `InputType` trait for implementing custom input types,
/// as well as several premade input formats that are commonly used.
pub mod inputs;
mod loader;
/// Contains the `OutputBuckets` trait for implementing custom output bucket types,
/// as well as several premade output buckets that are commonly used.
pub mod outputs;
mod quant;
pub mod testing;

use bulletformat::BulletFormat;

pub use builder::{Loss, TrainerBuilder};
pub use loader::DefaultDataPreparer;
pub use quant::QuantTarget;

use inputs::InputType;
use loader::DefaultDataLoader;
use outputs::OutputBuckets;
use testing::{EngineType, TestSettings};

use std::{
    collections::HashSet,
    fs::File,
    io::{self, Write},
};

use diffable::Node;

use super::{
    logger,
    schedule::{lr::LrScheduler, wdl::WdlScheduler, TrainingSteps},
    LocalSettings, TrainingSchedule,
};

use crate::{
    loader::{CanBeDirectlySequentiallyLoaded, DataLoader, DirectSequentialDataLoader},
    optimiser::Optimiser,
    trainer::NetworkTrainer,
    Graph,
};

/// Holy unsound code batman!
/// Needs refactor.
unsafe impl<T: BulletFormat + 'static> CanBeDirectlySequentiallyLoaded for T {}

#[derive(Clone, Copy)]
pub struct AdditionalTrainerInputs {
    nstm: bool,
    output_buckets: bool,
    wdl: bool,
    dense_inputs: bool,
}

pub struct Trainer<Opt, Inp, Out = outputs::Single> {
    optimiser: Opt,
    input_getter: Inp,
    output_getter: Out,
    output_node: Node,
    additional_inputs: AdditionalTrainerInputs,
    saved_format: Vec<(String, QuantTarget)>,
}

impl<Opt: Optimiser, Inp: InputType, Out: OutputBuckets<Inp::RequiredDataType>> NetworkTrainer
    for Trainer<Opt, Inp, Out>
{
    type Optimiser = Opt;
    type PreparedData = DefaultDataPreparer<Inp, Out>;

    fn load_batch(&mut self, prepared: &Self::PreparedData) -> usize {
        let batch_size = prepared.batch_size;

        let graph = self.optimiser.graph_mut();

        unsafe {
            if self.additional_inputs.dense_inputs {
                let input = &prepared.dstm;
                graph.get_input_mut("stm").load_dense_from_slice(input.shape, &input.value);

                if self.additional_inputs.nstm {
                    let input = &prepared.dnstm;
                    graph.get_input_mut("nstm").load_dense_from_slice(input.shape, &input.value);
                }
            } else {
                let input = &prepared.stm;
                graph.get_input_mut("stm").load_sparse_from_slice(input.shape, input.max_active, &input.value);

                if self.additional_inputs.nstm {
                    let input = &prepared.nstm;
                    graph.get_input_mut("nstm").load_sparse_from_slice(input.shape, input.max_active, &input.value);
                }
            }

            if self.additional_inputs.output_buckets {
                let input = &prepared.buckets;
                graph.get_input_mut("buckets").load_sparse_from_slice(input.shape, input.max_active, &input.value);
            }
        }

        graph.get_input_mut("targets").load_dense_from_slice(prepared.targets.shape, &prepared.targets.value);

        batch_size
    }

    fn optimiser(&self) -> &Self::Optimiser {
        &self.optimiser
    }

    fn optimiser_mut(&mut self) -> &mut Self::Optimiser {
        &mut self.optimiser
    }

    fn save_to_checkpoint(&self, path: &str) {
        std::fs::create_dir(path).unwrap_or(());

        let optimiser_path = format!("{path}/optimiser_state");
        std::fs::create_dir(optimiser_path.as_str()).unwrap_or(());
        self.optimiser().write_to_checkpoint(&optimiser_path);

        self.save_unquantised(&format!("{path}/raw.bin")).unwrap();
        if let Err(e) = self.save_quantised(&format!("{path}/quantised.bin")) {
            println!("{e}");
        }
    }
}

impl<Opt: Optimiser, Inp: InputType, Out: OutputBuckets<Inp::RequiredDataType>> Trainer<Opt, Inp, Out> {
    pub fn new(
        graph: Graph,
        output_node: Node,
        params: Opt::Params,
        input_getter: Inp,
        output_getter: Out,
        saved_format: Vec<(String, QuantTarget)>,
        dense_inputs: bool,
    ) -> Self {
        let inputs = graph.input_ids();
        let inputs = inputs.iter().map(String::as_str).collect::<HashSet<_>>();

        assert!(inputs.contains("stm"), "Graph does not contain stm inputs!");
        assert!(inputs.contains("targets"), "Graph does not contain targets!");

        let nstm = inputs.contains("nstm");
        let output_buckets = inputs.contains("buckets");
        let expected = 2 + usize::from(nstm) + usize::from(output_buckets);

        let output_shape = graph.get_node(output_node).values.shape();

        assert_eq!(output_shape.cols(), 1, "Output cannot have >1 column!");
        assert!(output_shape.rows() == 1 || output_shape.rows() == 3, "Only supports 1 or 3 outputs!");

        let wdl = output_shape.rows() == 3;

        if inputs.len() != expected {
            println!("WARNING: The network graph contains an unexpected number of inputs!")
        };

        Self {
            optimiser: Opt::new(graph, params),
            input_getter,
            output_getter,
            output_node,
            additional_inputs: AdditionalTrainerInputs { nstm, output_buckets, wdl, dense_inputs },
            saved_format,
        }
    }

    pub fn load_from_checkpoint(&mut self, path: &str) {
        <Self as NetworkTrainer>::load_from_checkpoint(self, path);
    }

    pub fn save_to_checkpoint(&self, path: &str) {
        <Self as NetworkTrainer>::save_to_checkpoint(self, path);
    }

    fn preamble<D: DataLoader<Inp::RequiredDataType>, LR: LrScheduler, WDL: WdlScheduler>(
        &self,
        schedule: &TrainingSchedule<LR, WDL>,
        settings: &LocalSettings,
        data_loader: &D,
    ) -> (DefaultDataLoader<Inp, Out, D>, Option<DirectLoader<Inp, Out>>) {
        logger::clear_colours();
        println!("{}", logger::ansi("Beginning Training", "34;1"));
        schedule.display();
        settings.display();
        let preparer = DefaultDataLoader::new(
            self.input_getter,
            self.output_getter,
            self.additional_inputs.wdl,
            schedule.eval_scale,
            data_loader.clone(),
            self.additional_inputs.dense_inputs,
        );
        let test_preparer = settings.test_set.map(|test| {
            DefaultDataLoader::new(
                self.input_getter,
                self.output_getter,
                self.additional_inputs.wdl,
                schedule.eval_scale,
                DirectSequentialDataLoader::new(&[test.path]),
                self.additional_inputs.dense_inputs,
            )
        });

        display_total_positions(data_loader, schedule.steps);

        (preparer, test_preparer)
    }

    pub fn run<D: DataLoader<Inp::RequiredDataType>, LR: LrScheduler, WDL: WdlScheduler>(
        &mut self,
        schedule: &TrainingSchedule<LR, WDL>,
        settings: &LocalSettings,
        data_loader: &D,
    ) {
        let (preparer, test_preparer) = self.preamble(schedule, settings, data_loader);

        self.train_custom(&preparer, &test_preparer, schedule, settings, |_, _, _, _| {});
    }

    pub fn run_and_test<D: DataLoader<Inp::RequiredDataType>, LR: LrScheduler, WDL: WdlScheduler, T: EngineType>(
        &mut self,
        schedule: &TrainingSchedule<LR, WDL>,
        settings: &LocalSettings,
        data_loader: &D,
        testing: &TestSettings<T>,
    ) {
        let (preparer, test_preparer) = self.preamble(schedule, settings, data_loader);

        testing.setup(schedule);

        let mut handles = Vec::new();

        self.train_custom(&preparer, &test_preparer, schedule, settings, |superbatch, trainer, schedule, _| {
            if superbatch % testing.test_rate == 0 || superbatch == schedule.steps.end_superbatch {
                trainer.save_to_checkpoint(&format!("{}/nets/{}-{superbatch}", testing.out_dir, schedule.net_id));
                let handle = testing.dispatch(&schedule.net_id, superbatch);
                handles.push(handle);
            }
        });

        println!("# [Waiting for Tests]");
        for handle in handles {
            if let Err(err) = handle.join() {
                println!("{err:?}");
            }
        }
    }

    pub fn eval(&mut self, fen: &str) -> f32
    where
        Inp::RequiredDataType: std::str::FromStr<Err = String>,
    {
        let pos = format!("{fen} | 0 | 0.0").parse::<Inp::RequiredDataType>().unwrap();

        let prepared = DefaultDataPreparer::prepare(
            self.input_getter,
            self.output_getter,
            self.additional_inputs.wdl,
            &[pos],
            1,
            1.0,
            1.0,
            self.additional_inputs.dense_inputs,
        );

        self.load_batch(&prepared);
        self.optimiser.graph_mut().forward();

        let eval = self.optimiser.graph().get_node(self.output_node);

        let mut val = vec![0.0; eval.values.dense().allocated_size()];
        eval.values.dense().write_to_slice(&mut val);
        val[0]
    }

    pub fn set_optimiser_params(&mut self, params: Opt::Params) {
        self.optimiser.set_params(params);
    }

    pub fn save_quantised(&self, path: &str) -> io::Result<()> {
        let mut file = File::create(path).unwrap();

        let mut buf = Vec::new();

        for (id, quant) in &self.saved_format {
            let weights = self.optimiser.graph().get_weights(id);
            let weights = weights.values.dense();

            let mut weight_buf = vec![0.0; weights.shape().size()];
            let written = weights.write_to_slice(&mut weight_buf);
            assert_eq!(written, weights.shape().size());

            let quantised = quant.quantise(&weight_buf)?;
            buf.extend_from_slice(&quantised);
        }

        let bytes = buf.len() % 64;
        if bytes > 0 {
            let chs = [b'b', b'u', b'l', b'l', b'e', b't'];

            for i in 0..64 - bytes {
                buf.push(chs[i % chs.len()]);
            }
        }

        file.write_all(&buf)?;

        Ok(())
    }

    pub fn save_unquantised(&self, path: &str) -> io::Result<()> {
        let mut file = File::create(path).unwrap();

        let mut buf = Vec::new();

        for (id, _) in &self.saved_format {
            let weights = self.optimiser.graph().get_weights(id);
            let weights = weights.values.dense();

            let mut weight_buf = vec![0.0; weights.shape().size()];
            let written = weights.write_to_slice(&mut weight_buf);
            assert_eq!(written, weights.shape().size());

            let quantised = QuantTarget::Float.quantise(&weight_buf)?;
            buf.extend_from_slice(&quantised);
        }

        file.write_all(&buf)?;

        Ok(())
    }
}

fn display_total_positions<T, D: DataLoader<T>>(data_loader: &D, steps: TrainingSteps) {
    if let Some(num) = data_loader.count_positions() {
        let pos_per_sb = steps.batch_size * steps.batches_per_superbatch;
        let sbs = steps.end_superbatch - steps.start_superbatch + 1;
        let total_pos = pos_per_sb * sbs;
        let iters = total_pos as f64 / num as f64;

        println!("Positions              : {}", logger::ansi(num, 31));
        println!("Total Epochs           : {}", logger::ansi(format!("{iters:.2}"), 31));
    }
}

type DirectLoader<Inp, Out> = DefaultDataLoader<Inp, Out, DirectSequentialDataLoader>;
