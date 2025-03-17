mod builder;
pub mod gamerunner;
/// Contains the `InputType` trait for implementing custom input types,
/// as well as several premade input formats that are commonly used.
pub mod inputs;
pub mod loader;
/// Contains the `OutputBuckets` trait for implementing custom output bucket types,
/// as well as several premade output buckets that are commonly used.
pub mod outputs;
pub mod testing;

/// Re-exports crates for certain file formats (e.g. Bulletformat)
pub mod formats {
    pub use bulletformat;
    pub use montyformat;
    pub use sfbinpack;
}

pub use super::save::{Layout, QuantTarget, SavedFormat};
pub use builder::{Loss, TrainerBuilder};

use inputs::SparseInputType;
use loader::{
    CanBeDirectlySequentiallyLoaded, DataLoader, DefaultDataLoader, DefaultDataPreparer, DirectSequentialDataLoader,
};
use outputs::OutputBuckets;
use testing::{EngineType, TestSettings};

use std::{
    collections::HashSet,
    fs::File,
    io::{self, Write},
};

use super::{
    logger,
    schedule::{lr::LrScheduler, wdl::WdlScheduler, TrainingSteps},
    LocalSettings, NetworkTrainer, TrainingSchedule,
};

use crate::{nn::DeviceError, ExecutionContext};

use bullet_core::{
    backend::device::OperationError,
    graph::{Graph, Node},
    optimiser::{Optimiser, OptimiserState},
};

unsafe impl CanBeDirectlySequentiallyLoaded for bulletformat::ChessBoard {}
unsafe impl CanBeDirectlySequentiallyLoaded for bulletformat::AtaxxBoard {}
unsafe impl CanBeDirectlySequentiallyLoaded for bulletformat::chess::CudADFormat {}
unsafe impl CanBeDirectlySequentiallyLoaded for bulletformat::chess::MarlinFormat {}

#[derive(Clone, Copy)]
pub struct AdditionalTrainerInputs {
    wdl: bool,
}

pub struct Trainer<Opt: OptimiserState<ExecutionContext>, Inp, Out = outputs::Single> {
    optimiser: Optimiser<ExecutionContext, Opt>,
    input_getter: Inp,
    output_getter: Out,
    output_node: Node,
    additional_inputs: AdditionalTrainerInputs,
    saved_format: Vec<SavedFormat>,
    factorised_weights: Option<Vec<String>>,
}

impl<Opt: OptimiserState<ExecutionContext>, Inp: SparseInputType, Out: OutputBuckets<Inp::RequiredDataType>>
    NetworkTrainer for Trainer<Opt, Inp, Out>
{
    type OptimiserState = Opt;
    type PreparedData = DefaultDataPreparer<Inp, Out>;

    fn load_batch(&mut self, prepared: &Self::PreparedData) -> usize {
        unsafe { load_into_graph(&mut self.optimiser.graph, prepared).unwrap() }
    }

    fn optimiser(&self) -> &Optimiser<ExecutionContext, Self::OptimiserState> {
        &self.optimiser
    }

    fn optimiser_mut(&mut self) -> &mut Optimiser<ExecutionContext, Self::OptimiserState> {
        &mut self.optimiser
    }

    fn save_to_checkpoint(&self, path: &str) {
        std::fs::create_dir(path).unwrap_or(());

        let optimiser_path = format!("{path}/optimiser_state");
        std::fs::create_dir(optimiser_path.as_str()).unwrap_or(());
        self.optimiser().write_to_checkpoint(&optimiser_path).unwrap();

        if let Err(e) = self.save_unquantised(&format!("{path}/raw.bin")) {
            println!("Failed to write raw network weights:");
            println!("{e}");
        }

        if let Err(e) = self.save_quantised(&format!("{path}/quantised.bin")) {
            println!("Failed to write quantised network weights:");
            println!("{e}");
        }
    }
}

impl<Opt: OptimiserState<ExecutionContext>, Inp: SparseInputType, Out: OutputBuckets<Inp::RequiredDataType>>
    Trainer<Opt, Inp, Out>
{
    pub fn new(
        graph: Graph<ExecutionContext>,
        output_node: Node,
        params: Opt::Params,
        input_getter: Inp,
        output_getter: Out,
        saved_format: Vec<SavedFormat>,
        dense_inputs: bool,
    ) -> Self {
        let inputs = graph.input_ids();
        let inputs = inputs.iter().map(String::as_str).collect::<HashSet<_>>();

        assert!(!dense_inputs, "Inputs are now always sparse and must be converted to dense in your network builder!");

        assert!(inputs.contains("stm"), "Graph does not contain stm inputs!");
        assert!(inputs.contains("targets"), "Graph does not contain targets!");

        let nstm = inputs.contains("nstm");
        let output_buckets = inputs.contains("buckets");
        let expected = 2 + usize::from(nstm) + usize::from(output_buckets);

        let output_shape = output_node.shape;

        assert_eq!(output_shape.cols(), 1, "Output cannot have >1 column!");
        assert!(output_shape.rows() == 1 || output_shape.rows() == 3, "Only supports 1 or 3 outputs!");

        let wdl = output_shape.rows() == 3;

        if inputs.len() != expected {
            println!("WARNING: The network graph contains an unexpected number of inputs!")
        };

        Self {
            optimiser: Optimiser::new(graph, params).unwrap(),
            input_getter,
            output_getter,
            output_node,
            additional_inputs: AdditionalTrainerInputs { wdl },
            saved_format,
            factorised_weights: None,
        }
    }

    pub fn load_from_checkpoint(&mut self, path: &str) {
        <Self as NetworkTrainer>::load_from_checkpoint(self, path);
    }

    pub fn save_to_checkpoint(&self, path: &str) {
        <Self as NetworkTrainer>::save_to_checkpoint(self, path);
    }

    pub fn eval_raw_output(&mut self, fen: &str) -> Vec<f32>
    where
        Inp::RequiredDataType: std::str::FromStr<Err = String>,
    {
        let pos = format!("{fen} | 0 | 0.0").parse::<Inp::RequiredDataType>().unwrap();

        let prepared = DefaultDataPreparer::prepare(
            self.input_getter.clone(),
            self.output_getter,
            self.additional_inputs.wdl,
            &[pos],
            1,
            1.0,
            1.0,
        );

        self.load_batch(&prepared);
        self.optimiser.graph.forward().unwrap();

        let eval = self.optimiser.graph.get_node(self.output_node);

        let dense_vals = eval.values.dense().unwrap();
        let mut vals = vec![0.0; dense_vals.size()];
        dense_vals.write_to_slice(&mut vals).unwrap();
        vals
    }

    pub fn eval(&mut self, fen: &str) -> f32
    where
        Inp::RequiredDataType: std::str::FromStr<Err = String>,
    {
        let vals = self.eval_raw_output(fen);

        match &vals[..] {
            [mut loss, mut draw, mut win] => {
                let max = win.max(draw).max(loss);
                win = (win - max).exp();
                draw = (draw - max).exp();
                loss = (loss - max).exp();

                (win + draw / 2.0) / (win + draw + loss)
            }
            [score] => *score,
            _ => panic!("Invalid output size!"),
        }
    }

    pub fn set_optimiser_params(&mut self, params: Opt::Params) {
        self.optimiser.set_params(params);
    }

    pub fn sanity_check(&self) {
        self.optimiser.graph.sanity_check();
    }

    pub fn mark_weights_as_input_factorised(&mut self, weights: &[&str]) {
        if self.factorised_weights.is_none() {
            self.factorised_weights = Some(Vec::new())
        }

        for weight in weights {
            self.factorised_weights.as_mut().unwrap().push(weight.to_string());
        }
    }

    pub fn profile_node(&mut self, node: Node, id: &str) {
        self.optimiser.graph.profile_node(node, id);
    }

    pub fn profile_all_nodes(&mut self) {
        self.optimiser.graph.profile_all_nodes();
    }

    pub fn report_profiles(&self) {
        self.optimiser.graph.report_profiles();
    }

    pub fn save_quantised(&self, path: &str) -> io::Result<()> {
        let mut file = File::create(path).unwrap();

        let mut buf = Vec::new();

        for SavedFormat { id, quant, layout, transforms } in &self.saved_format {
            let weights = self.optimiser.graph.get_weights(id);
            let weights = weights.values.dense().unwrap();

            let mut weight_buf = vec![0.0; weights.size()];
            let written = weights.write_to_slice(&mut weight_buf).unwrap();
            assert_eq!(written, weights.size());

            if let Some(factorised) = &self.factorised_weights {
                if factorised.contains(id) {
                    assert!(self.input_getter.is_factorised(), "Attempting to merge in unfactorised weights!");
                    weight_buf = self.input_getter.merge_factoriser(weight_buf);

                    if let Layout::Transposed(_) = layout {
                        unimplemented!(
                            "Transposing post-factoriser merge is not currently supported - why do you want to do this?"
                        );
                    }
                }
            }

            if let Layout::Transposed(shape) = layout {
                assert_eq!(shape.size(), weights.size());
                weight_buf = SavedFormat::transpose(*shape, &weight_buf);
            }

            for transform in transforms {
                weight_buf = transform(&self.optimiser.graph, weight_buf);
            }

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

        for SavedFormat { id, .. } in &self.saved_format {
            let weights = self.optimiser.graph.get_weights(id);
            let weights = weights.values.dense().unwrap();

            let mut weight_buf = vec![0.0; weights.size()];
            let written = weights.write_to_slice(&mut weight_buf).unwrap();
            assert_eq!(written, weights.size());

            let quantised = QuantTarget::Float.quantise(&weight_buf)?;
            buf.extend_from_slice(&quantised);
        }

        file.write_all(&buf)?;

        Ok(())
    }

    pub fn training_preamble<D, D2, LR: LrScheduler, WDL: WdlScheduler>(
        &self,
        schedule: &TrainingSchedule<LR, WDL>,
        settings: &LocalSettings,
        data_loader: &D,
        test_loader: &Option<D2>,
    ) -> PairedLoaders<Inp, Out, D, D2>
    where
        D: DataLoader<Inp::RequiredDataType>,
        D2: DataLoader<Inp::RequiredDataType>,
    {
        logger::clear_colours();
        println!("{}", logger::ansi("Beginning Training", "34;1"));

        schedule.display();
        settings.display();

        let preparer = DefaultDataLoader::new(
            self.input_getter.clone(),
            self.output_getter,
            self.additional_inputs.wdl,
            schedule.eval_scale,
            data_loader.clone(),
        );

        let test_preparer = test_loader.as_ref().map(|loader| {
            DefaultDataLoader::new(
                self.input_getter.clone(),
                self.output_getter,
                self.additional_inputs.wdl,
                schedule.eval_scale,
                loader.clone(),
            )
        });

        display_total_positions(data_loader, schedule.steps);

        (preparer, test_preparer)
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

impl<Opt: OptimiserState<ExecutionContext>, Inp: SparseInputType, Out: OutputBuckets<Inp::RequiredDataType>>
    Trainer<Opt, Inp, Out>
where
    Inp::RequiredDataType: CanBeDirectlySequentiallyLoaded,
{
    pub fn run<D: DataLoader<Inp::RequiredDataType>, LR: LrScheduler, WDL: WdlScheduler>(
        &mut self,
        schedule: &TrainingSchedule<LR, WDL>,
        settings: &LocalSettings,
        data_loader: &D,
    ) {
        let test_loader = settings.test_set.map(|test| DirectSequentialDataLoader::new(&[test.path]));
        let (preparer, test_preparer) = self.training_preamble(schedule, settings, data_loader, &test_loader);

        self.train_custom(&preparer, &test_preparer, schedule, settings, |_, _, _, _| {});
    }

    pub fn run_and_test<D: DataLoader<Inp::RequiredDataType>, LR: LrScheduler, WDL: WdlScheduler, T: EngineType>(
        &mut self,
        schedule: &TrainingSchedule<LR, WDL>,
        settings: &LocalSettings,
        data_loader: &D,
        testing: &TestSettings<T>,
    ) {
        let test_loader = settings.test_set.map(|test| DirectSequentialDataLoader::new(&[test.path]));
        let (preparer, test_preparer) = self.training_preamble(schedule, settings, data_loader, &test_loader);

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
}

type PairedLoaders<Inp, Out, D, D2> = (DefaultDataLoader<Inp, Out, D>, Option<DefaultDataLoader<Inp, Out, D2>>);

/// # Safety
///
/// The graph needs to take sparse `stm` and optionally `nstm` inputs
/// in the correct format
pub unsafe fn load_into_graph<Inp, Out>(
    graph: &mut Graph<ExecutionContext>,
    prepared: &DefaultDataPreparer<Inp, Out>,
) -> Result<usize, OperationError<DeviceError>>
where
    Inp: SparseInputType,
    Out: OutputBuckets<Inp::RequiredDataType>,
{
    let batch_size = prepared.batch_size;
    let expected_inputs = prepared.input_getter.num_inputs();

    unsafe {
        let input = &prepared.stm;
        let mut stm = graph.get_input_mut("stm");

        if stm.values.single_size() != expected_inputs {
            return Err(OperationError::InvalidTensorFormat);
        }

        stm.load_sparse_from_slice(input.max_active, Some(batch_size), &input.value)?;

        drop(stm);
        let input_ids = graph.input_ids();

        if input_ids.contains(&"nstm".to_string()) {
            let input = &prepared.nstm;
            let ntm = &mut *graph.get_input_mut("nstm");

            if ntm.values.single_size() != expected_inputs {
                return Err(OperationError::InvalidTensorFormat);
            }

            ntm.load_sparse_from_slice(input.max_active, Some(batch_size), &input.value)?;
        }
    }

    if graph.input_ids().contains(&"buckets".to_string()) {
        let input = &prepared.buckets;
        let mut buckets = graph.get_input_mut("buckets");

        if buckets.values.single_size() != Out::BUCKETS {
            return Err(OperationError::InvalidTensorFormat);
        }

        buckets.load_sparse_from_slice(input.max_active, Some(batch_size), &input.value)?;
    }

    graph.get_input_mut("targets").load_dense_from_slice(Some(batch_size), &prepared.targets.value)?;

    Ok(batch_size)
}
