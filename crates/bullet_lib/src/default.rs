#![allow(deprecated)]

mod builder;
pub mod gamerunner;
pub mod testing;

/// Re-exports crates for certain file formats (e.g. Bulletformat)
pub mod formats {
    pub use bulletformat;
    pub use montyformat;
    pub use sfbinpack;
}

pub use crate::{
    game::{inputs, outputs},
    value::loader,
};
pub use builder::{Loss, TrainerBuilder};

use crate::value::loader::{
    load_into_graph, CanBeDirectlySequentiallyLoaded, DataLoader, DefaultDataLoader, DefaultDataPreparer,
    DirectSequentialDataLoader, LoadableDataType, B,
};
use testing::{EngineType, TestSettings};

use std::{
    collections::HashSet,
    fs::File,
    io::{self, Write},
};

use crate::{
    game::{inputs::SparseInputType, outputs::OutputBuckets},
    trainer::{
        logger,
        save::{Layout, QuantTarget, SavedFormat},
        schedule::{lr::LrScheduler, wdl::WdlScheduler, TrainingSchedule, TrainingSteps},
        settings::LocalSettings,
        NetworkTrainer,
    },
    ExecutionContext,
};

use bullet_core::{
    graph::{Graph, Node, NodeId, NodeIdTy},
    optimiser::{Optimiser, OptimiserState},
};

#[derive(Clone, Copy)]
pub struct AdditionalTrainerInputs {
    pub wdl: bool,
}

pub(crate) type Wgt<I> = fn(&<I as SparseInputType>::RequiredDataType) -> f32;

pub struct Trainer<Opt: OptimiserState<ExecutionContext>, Inp: SparseInputType, Out> {
    pub(crate) optimiser: Optimiser<ExecutionContext, Opt>,
    pub(crate) input_getter: Inp,
    pub(crate) output_getter: Out,
    pub(crate) blend_getter: B<Inp>,
    pub(crate) weight_getter: Option<Wgt<Inp>>,
    pub(crate) use_win_rate_model: bool,
    pub(crate) output_node: Node,
    pub(crate) additional_inputs: AdditionalTrainerInputs,
    pub(crate) saved_format: Vec<SavedFormat>,
    pub(crate) factorised_weights: Option<Vec<String>>,
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
    #[deprecated(note = "You should use `ValueTrainerBuilder` instead of this!")]
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
            blend_getter: |_, wdl| wdl,
            weight_getter: None,
            use_win_rate_model: false,
            output_node,
            additional_inputs: AdditionalTrainerInputs { wdl },
            saved_format,
            factorised_weights: None,
        }
    }

    pub fn set_wdl_adjust(&mut self, func: B<Inp>) {
        self.blend_getter = func;
    }

    pub fn load_from_checkpoint(&mut self, path: &str) {
        <Self as NetworkTrainer>::load_from_checkpoint(self, path);
    }

    pub fn load_weights_from_file(&mut self, path: &str) {
        self.optimiser.load_weights_from_file(path).unwrap()
    }

    pub fn save_to_checkpoint(&self, path: &str) {
        <Self as NetworkTrainer>::save_to_checkpoint(self, path);
    }

    pub fn eval_raw_output(&mut self, fen: &str) -> Vec<f32>
    where
        Inp::RequiredDataType: std::str::FromStr<Err = String> + LoadableDataType,
    {
        let pos = format!("{fen} | 0 | 0.0").parse::<Inp::RequiredDataType>().unwrap();

        let prepared = DefaultDataPreparer::prepare(
            self.input_getter.clone(),
            self.output_getter,
            self.blend_getter,
            self.weight_getter,
            self.use_win_rate_model,
            self.additional_inputs.wdl,
            &[pos],
            1,
            1.0,
            1.0,
        );

        self.load_batch(&prepared);
        self.optimiser.graph.forward().unwrap();

        let id = NodeId::new(self.output_node.idx(), NodeIdTy::Values);
        let eval = self.optimiser.graph.get(id).unwrap();

        let dense_vals = eval.dense().unwrap();
        let mut vals = vec![0.0; dense_vals.size()];
        dense_vals.write_to_slice(&mut vals).unwrap();
        vals
    }

    pub fn eval(&mut self, fen: &str) -> f32
    where
        Inp::RequiredDataType: std::str::FromStr<Err = String> + LoadableDataType,
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

    pub fn save_quantised(&self, path: &str) -> io::Result<()> {
        let mut file = File::create(path).unwrap();

        let mut buf = Vec::new();

        for SavedFormat { custom: _, id, quant, layout, transforms, round } in &self.saved_format {
            let id = id.as_ref().unwrap();
            let idx = NodeId::new(self.optimiser.graph.weight_idx(id).unwrap(), NodeIdTy::Values);
            let weights = self.optimiser.graph.get(idx).unwrap();
            let weights = weights.dense().unwrap();

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
                weight_buf = SavedFormat::transpose_impl(*shape, &weight_buf);
            }

            for transform in transforms {
                weight_buf = transform(&self.optimiser.graph, id, weight_buf);
            }

            let quantised = match quant.quantise(*round, &weight_buf) {
                Ok(q) => q,
                Err(err) => {
                    println!("Quantisation failed for id: {id}");
                    return Err(err);
                }
            };

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
            let id = id.as_ref().unwrap();
            let id = NodeId::new(self.optimiser.graph.weight_idx(id).unwrap(), NodeIdTy::Values);
            let weights = self.optimiser.graph.get(id).unwrap();
            let weights = weights.dense().unwrap();

            let mut weight_buf = vec![0.0; weights.size()];
            let written = weights.write_to_slice(&mut weight_buf).unwrap();
            assert_eq!(written, weights.size());

            let quantised = QuantTarget::Float.quantise(false, &weight_buf)?;
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
            self.blend_getter,
            self.weight_getter,
            self.use_win_rate_model,
            self.additional_inputs.wdl,
            schedule.eval_scale,
            data_loader.clone(),
        );

        let test_preparer = test_loader.as_ref().map(|loader| {
            DefaultDataLoader::new(
                self.input_getter.clone(),
                self.output_getter,
                self.blend_getter,
                self.weight_getter,
                self.use_win_rate_model,
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
    Inp::RequiredDataType: CanBeDirectlySequentiallyLoaded + LoadableDataType,
{
    pub fn run(
        &mut self,
        schedule: &TrainingSchedule<impl LrScheduler, impl WdlScheduler>,
        settings: &LocalSettings,
        data_loader: &impl DataLoader<Inp::RequiredDataType>,
    ) {
        self.run_with_callback(schedule, settings, data_loader, |_, _, _, _| {});
    }

    pub fn run_with_callback<LR: LrScheduler, WDL: WdlScheduler>(
        &mut self,
        schedule: &TrainingSchedule<LR, WDL>,
        settings: &LocalSettings,
        data_loader: &impl DataLoader<Inp::RequiredDataType>,
        mut callback: impl FnMut(usize, &Self, &TrainingSchedule<LR, WDL>, &LocalSettings),
    ) {
        let test_loader = settings.test_set.map(|test| DirectSequentialDataLoader::new(&[test.path]));
        let (preparer, test_preparer) = self.training_preamble(schedule, settings, data_loader, &test_loader);

        self.train_custom(&preparer, &test_preparer, schedule, settings, |superbatch, trainer, schedule, settings| {
            callback(superbatch, trainer, schedule, settings)
        });
    }

    pub fn run_and_test(
        &mut self,
        schedule: &TrainingSchedule<impl LrScheduler, impl WdlScheduler>,
        settings: &LocalSettings,
        data_loader: &impl DataLoader<Inp::RequiredDataType>,
        testing: &TestSettings<impl EngineType>,
    ) {
        testing.setup(schedule);

        let mut handles = Vec::new();

        self.run_with_callback(schedule, settings, data_loader, |superbatch, trainer, schedule, _| {
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
