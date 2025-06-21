use std::{
    fs::File,
    io::{self, Write},
};

use crate::nn::ExecutionContext;
use bullet_core::optimiser::OptimiserState;

use crate::{
    game::{inputs::SparseInputType, outputs::OutputBuckets},
    trainer::save::{Layout, QuantTarget, SavedFormat},
    value::{loader::LoadableDataType, ValueTrainer},
};

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
        std::fs::create_dir(path).unwrap_or(());

        let optimiser_path = format!("{path}/optimiser_state");
        std::fs::create_dir(optimiser_path.as_str()).unwrap_or(());
        self.optimiser.write_to_checkpoint(&optimiser_path).unwrap();

        if let Err(e) = self.save_unquantised(&format!("{path}/raw.bin")) {
            println!("Failed to write raw network weights:");
            println!("{e}");
        }

        if let Err(e) = self.save_quantised(&format!("{path}/quantised.bin")) {
            println!("Failed to write quantised network weights:");
            println!("{e}");
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
