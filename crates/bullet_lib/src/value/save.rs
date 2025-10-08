use std::{
    fs::File,
    io::{self, Write},
};

use crate::{
    nn::{ExecutionContext, Graph},
    value::ValueTrainerState,
};
use acyclib::{
    graph::{
        GraphNodeId, GraphNodeIdTy,
        like::GraphLike,
        save::{GraphWeights, QuantTarget},
    },
    trainer::{Trainer, optimiser::OptimiserState},
};

use crate::{
    game::{inputs::SparseInputType, outputs::OutputBuckets},
    value::{ValueTrainer, loader::LoadableDataType},
};

type ValueTrainerInner<Opt, Inp, Out> = Trainer<ExecutionContext, Graph, Opt, ValueTrainerState<Inp, Out>>;

pub(super) fn write_losses(path: &str, error_record: &[(usize, usize, f32)]) {
    use std::io::Write;

    let mut writer = std::io::BufWriter::new(std::fs::File::create(path).expect("Opening log file failed!"));
    for (superbatch, batch, loss) in error_record {
        writeln!(writer, "{superbatch},{batch},{loss}",).expect("Writing to log file failed!");
    }
}

pub(super) fn save_to_checkpoint<Opt, Inp, Out>(trainer: &ValueTrainerInner<Opt, Inp, Out>, path: &str)
where
    Opt: OptimiserState<ExecutionContext>,
    Inp: SparseInputType,
    Inp::RequiredDataType: LoadableDataType,
    Out: OutputBuckets<Inp::RequiredDataType>,
{
    std::fs::create_dir(path).unwrap_or(());

    let optimiser_path = format!("{path}/optimiser_state");
    std::fs::create_dir(optimiser_path.as_str()).unwrap_or(());
    trainer.optimiser.write_to_checkpoint(&optimiser_path).unwrap();

    if let Err(e) = save_unquantised(trainer, &format!("{path}/raw.bin")) {
        println!("Failed to write raw network weights:");
        println!("{e}");
    }

    if let Err(e) = save_quantised(trainer, &format!("{path}/quantised.bin")) {
        println!("Failed to write quantised network weights:");
        println!("{e}");
    }
}

pub(super) fn save_unquantised<Opt, Inp, Out>(trainer: &ValueTrainerInner<Opt, Inp, Out>, path: &str) -> io::Result<()>
where
    Opt: OptimiserState<ExecutionContext>,
    Inp: SparseInputType,
    Inp::RequiredDataType: LoadableDataType,
    Out: OutputBuckets<Inp::RequiredDataType>,
{
    let mut file = File::create(path).unwrap();

    let mut buf = Vec::new();

    for fmt in &trainer.state.saved_format {
        if let Some(id) = &fmt.get_id() {
            let idx =
                GraphNodeId::new(trainer.optimiser.graph.primary().weight_idx(id).unwrap(), GraphNodeIdTy::Values);
            let weights = trainer.optimiser.graph.primary().get(idx).unwrap();
            let weights = weights.dense();

            let mut weight_buf = vec![0.0; weights.size()];
            let written = weights.write_to_slice(&mut weight_buf).unwrap();
            assert_eq!(written, weights.size());

            let quantised = QuantTarget::Float.quantise(false, &weight_buf)?;
            buf.extend_from_slice(&quantised);
        }
    }

    file.write_all(&buf)?;

    Ok(())
}

pub(super) fn save_quantised<Opt, Inp, Out>(trainer: &ValueTrainerInner<Opt, Inp, Out>, path: &str) -> io::Result<()>
where
    Opt: OptimiserState<ExecutionContext>,
    Inp: SparseInputType,
    Inp::RequiredDataType: LoadableDataType,
    Out: OutputBuckets<Inp::RequiredDataType>,
{
    let weight_store = GraphWeights::from(trainer.optimiser.graph.primary());

    let mut file = File::create(path).unwrap();
    let mut buf = Vec::new();

    for fmt in &trainer.state.saved_format {
        buf.extend_from_slice(&fmt.write_to_byte_buffer(&weight_store)?);
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

impl<Opt, Inp, Out> ValueTrainer<Opt, Inp, Out>
where
    Opt: OptimiserState<ExecutionContext>,
    Inp: SparseInputType,
    Inp::RequiredDataType: LoadableDataType,
    Out: OutputBuckets<Inp::RequiredDataType>,
{
    pub fn load_from_checkpoint(&mut self, path: &str) {
        let err = self.optimiser.load_from_checkpoint(&format!("{path}/optimiser_state"));
        if let Err(e) = err {
            println!();
            println!("Error loading from checkpoint:");
            println!("{e:?}");
            std::process::exit(1);
        }
    }

    pub fn save_to_checkpoint(&self, path: &str) {
        save_to_checkpoint(&self.0, path);
    }

    pub fn save_quantised(&self, path: &str) -> io::Result<()> {
        save_quantised(self, path)
    }

    pub fn save_unquantised(&self, path: &str) -> io::Result<()> {
        save_unquantised(self, path)
    }
}
