pub(crate) mod builder;
mod dataloader;

use std::{
    fs::File,
    io::{self, Write},
    ops::{Deref, DerefMut},
};

pub use builder::{NoOutputBuckets, ValueTrainerBuilder};

use crate::{nn::ExecutionContext, value::loader::DefaultDataPreparer};
use bullet_core::{
    graph::Node,
    optimiser::OptimiserState,
    trainer::{
        self,
        dataloader::{PreparedBatchDevice, PreparedBatchHost},
        logger, Trainer,
    },
};

use crate::{
    game::{inputs::SparseInputType, outputs::OutputBuckets},
    lr::LrScheduler,
    trainer::save::{Layout, QuantTarget, SavedFormat},
    value::{
        dataloader::ValueDataLoader,
        loader::{DefaultDataLoader, LoadableDataType},
    },
    wdl::WdlScheduler,
    LocalSettings, TrainingSchedule,
};

pub use crate::default::loader;

/// For now `ValueTrainer` just aliases the existing `Trainer`,
/// because the only improvements for now are in the **construction**
/// of the trainer via `ValueTrainerBuilder`.
pub struct ValueTrainer<
    Opt: OptimiserState<ExecutionContext>,
    Inp: SparseInputType,
    Out: OutputBuckets<Inp::RequiredDataType>,
>(Trainer<ExecutionContext, Opt, ValueTrainerState<Inp, Out>>);

impl<Opt, Inp, Out> Deref for ValueTrainer<Opt, Inp, Out>
where
    Opt: OptimiserState<ExecutionContext>,
    Inp: SparseInputType,
    Out: OutputBuckets<Inp::RequiredDataType>,
{
    type Target = Trainer<ExecutionContext, Opt, ValueTrainerState<Inp, Out>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<Opt, Inp, Out> DerefMut for ValueTrainer<Opt, Inp, Out>
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

pub struct ValueTrainerState<Inp: SparseInputType, Out> {
    input_getter: Inp,
    output_getter: Out,
    blend_getter: B<Inp>,
    weight_getter: Option<Wgt<Inp>>,
    output_node: Node,
    saved_format: Vec<SavedFormat>,
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

        let dataloader = DefaultDataLoader::new(
            self.state.input_getter.clone(),
            self.state.output_getter,
            self.state.blend_getter,
            self.state.weight_getter,
            false,
            false,
            schedule.eval_scale,
            dataloader.clone(),
        );

        let lr_scheduler = schedule.lr_scheduler.clone();

        self.train_custom(
            trainer::schedule::TrainingSchedule {
                steps: schedule.steps,
                save_rate: schedule.save_rate,
                out_dir: settings.output_directory.to_string(),
                log_rate: 128,
                lr_schedule: Box::new(|a, b| lr_scheduler.lr(a, b)),
                net_id: schedule.net_id.clone(),
            },
            ValueDataLoader {
                steps: schedule.steps,
                threads: settings.threads,
                dataloader,
                wdl: schedule.wdl_scheduler.clone(),
            },
            |_, _, _| {},
            |_, _| {},
        )
        .unwrap();
    }

    pub fn load_from_checkpoint(&mut self, path: &str) {
        let err = self.optimiser.load_from_checkpoint(&format!("{path}/optimiser_state"));
        if let Err(e) = err {
            println!();
            println!("Error loading from checkpoint:");
            println!("{e:?}");
            std::process::exit(1);
        }
    }

    pub fn eval_raw_output(&mut self, fen: &str) -> Vec<f32>
    where
        Inp::RequiredDataType: std::str::FromStr<Err = String> + LoadableDataType,
    {
        let pos = format!("{fen} | 0 | 0.0").parse::<Inp::RequiredDataType>().unwrap();

        let prepared = DefaultDataPreparer::prepare(
            self.state.input_getter.clone(),
            self.state.output_getter,
            self.state.blend_getter,
            self.state.weight_getter,
            false,
            false,
            &[pos],
            1,
            1.0,
            1.0,
        );

        let host_data = PreparedBatchHost::from(prepared);
        let mut device_data = PreparedBatchDevice::new(self.optimiser.graph.device(), &host_data).unwrap();

        device_data.load_into_graph(&mut self.optimiser.graph).unwrap();

        self.optimiser.graph.synchronise().unwrap();
        self.optimiser.graph.forward().unwrap();

        let eval = self.optimiser.graph.get_node(self.state.output_node);

        let dense_vals = eval.values.dense().unwrap();
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

    pub fn save_quantised(&self, path: &str) -> io::Result<()> {
        let mut file = File::create(path).unwrap();

        let mut buf = Vec::new();

        for SavedFormat { id, quant, layout, transforms, round } in &self.state.saved_format {
            let weights = self.optimiser.graph.get_weights(id);
            let weights = weights.values.dense().unwrap();

            let mut weight_buf = vec![0.0; weights.size()];
            let written = weights.write_to_slice(&mut weight_buf).unwrap();
            assert_eq!(written, weights.size());

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
                    println!("Quantisation failed for id: {}", id);
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

        for SavedFormat { id, .. } in &self.state.saved_format {
            let weights = self.optimiser.graph.get_weights(id);
            let weights = weights.values.dense().unwrap();

            let mut weight_buf = vec![0.0; weights.size()];
            let written = weights.write_to_slice(&mut weight_buf).unwrap();
            assert_eq!(written, weights.size());

            let quantised = QuantTarget::Float.quantise(false, &weight_buf)?;
            buf.extend_from_slice(&quantised);
        }

        file.write_all(&buf)?;

        Ok(())
    }
}
