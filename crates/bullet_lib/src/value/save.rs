use std::{
    fs::File,
    io::{self, Write},
};

use crate::nn::ExecutionContext;
use bullet_compiler::tensor::TValue;
use bullet_trainer::{
    model::{QuantTarget, SavedFormat},
    optimiser::{Optimiser, OptimiserState},
};

use crate::{game::inputs::SparseInputType, value::ValueTrainer};

pub fn write_losses(path: &str, error_record: &[(usize, usize, f32)]) {
    use std::io::Write;

    let mut writer = std::io::BufWriter::new(std::fs::File::create(path).expect("Opening log file failed!"));
    for (superbatch, batch, loss) in error_record {
        writeln!(writer, "{superbatch},{batch},{loss}",).expect("Writing to log file failed!");
    }
}

pub fn save_to_checkpoint<O>(optimiser: &Optimiser<ExecutionContext, O>, saved_format: &[SavedFormat], path: &str)
where
    O: OptimiserState<ExecutionContext>,
{
    std::fs::create_dir(path).unwrap_or(());

    let optimiser_path = format!("{path}/optimiser_state");
    std::fs::create_dir(optimiser_path.as_str()).unwrap_or(());
    optimiser.write_to_checkpoint(&optimiser_path).unwrap();

    if let Err(e) = save_unquantised(optimiser, saved_format, &format!("{path}/raw.bin")) {
        println!("Failed to write raw network weights:");
        println!("{e}");
    }

    if let Err(e) = save_quantised(optimiser, saved_format, &format!("{path}/quantised.bin")) {
        println!("Failed to write quantised network weights:");
        println!("{e}");
    }
}

pub fn save_unquantised<O>(
    optimiser: &Optimiser<ExecutionContext, O>,
    saved_format: &[SavedFormat],
    path: &str,
) -> io::Result<()>
where
    O: OptimiserState<ExecutionContext>,
{
    let mut file = File::create(path).unwrap();

    let mut buf = Vec::new();

    for fmt in saved_format {
        if let Some(id) = &fmt.get_id() {
            let weights = optimiser.weights().get(id).unwrap();
            let TValue::F32(weights) = weights.to_host().unwrap() else { panic!() };
            let quantised = QuantTarget::Float.quantise(false, &weights)?;
            buf.extend_from_slice(&quantised);
        }
    }

    file.write_all(&buf)?;

    Ok(())
}

pub fn save_quantised<O>(
    optimiser: &Optimiser<ExecutionContext, O>,
    saved_format: &[SavedFormat],
    path: &str,
) -> io::Result<()>
where
    O: OptimiserState<ExecutionContext>,
{
    let weight_store = &*optimiser.cpu_weights().unwrap();

    let mut file = File::create(path).unwrap();
    let buf = weight_store.to_quantised_buffer(saved_format, true)?;
    file.write_all(&buf)?;

    Ok(())
}

impl<Opt, Inp, Out> ValueTrainer<Opt, Inp, Out>
where
    Opt: OptimiserState<ExecutionContext>,
    Inp: SparseInputType,
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
        save_to_checkpoint(&self.optimiser, &self.state.saved_format, path);
    }

    pub fn save_quantised(&self, path: &str) -> io::Result<()> {
        save_quantised(&self.optimiser, &self.state.saved_format, path)
    }

    pub fn save_unquantised(&self, path: &str) -> io::Result<()> {
        save_unquantised(&self.optimiser, &self.state.saved_format, path)
    }
}
