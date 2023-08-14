use data::PackedPosition;

use crate::{
    arch::{update_single_grad, NNUEParams, QuantisedNNUE, K},
    position::Position,
};

use std::{
    fs::File,
    io::{BufRead, BufReader},
    path::Path,
    thread,
    time::Instant,
};

pub struct Trainer {
    file: File,
    pub data: Vec<Position>,
    threads: usize,
    rate: f64,
}

impl Trainer {
    pub fn run(
        &mut self,
        nnue: &mut NNUEParams,
        max_epochs: usize,
        net_name: &str,
        report_rate: usize,
        save_rate: usize,
        batch_size: usize,
    ) {
        self.load_data();

        let mut velocity = Box::<NNUEParams>::default();
        let mut momentum = Box::<NNUEParams>::default();

        let timer = Instant::now();

        let mut error;

        for epoch in 1..=max_epochs {
            error = 0.0;

            for batch in self.data.chunks(batch_size) {
                self.update_weights(nnue, batch, &mut velocity, &mut momentum, &mut error);
            }

            error /= self.data.len() as f64;

            if epoch % report_rate == 0 {
                let eps = epoch as f64 / timer.elapsed().as_secs_f64();
                println!("epoch {epoch} error {error:.6} eps {eps:.2}/sec");
            }

            if epoch % save_rate == 0 {
                let qnnue = QuantisedNNUE::from_unquantised(nnue);
                qnnue
                    .write_to_bin(&format!("{net_name}-{epoch}.bin"))
                    .unwrap();
            }
        }
    }

    #[must_use]
    pub fn new(path: impl AsRef<Path>, threads: usize, rate: f64) -> Self {
        Self {
            file: File::open(path).unwrap(),
            data: Vec::new(),
            threads,
            rate,
        }
    }

    pub fn load_data(&mut self) {
        let timer = Instant::now();

        let mut file = BufReader::with_capacity(16384, &self.file);

        while let Ok(buf) = file.fill_buf() {
            // finished reading file
            if buf.is_empty() {
                break;
            }

            let buf_ref: &[PackedPosition] = unsafe { util::to_slice_with_lifetime(buf) };

            for &packed in buf_ref {
                let pos = Position::from(packed);
                self.data.push(pos);
            }

            let consumed = buf.len();
            file.consume(consumed);
        }

        let elapsed = timer.elapsed().as_secs_f64();
        let pps = self.data.len() as f64 / elapsed;
        println!(
            "{} positions in {elapsed:.2} seconds, {pps:.2} pos/sec",
            self.data.len()
        );
    }

    fn gradients(&self, nnue: &NNUEParams, batch: &[Position], error: &mut f64) -> Box<NNUEParams> {
        let size = batch.len() / self.threads;
        let mut errors = vec![0.0; self.threads];
        let mut grad = Box::default();
        thread::scope(|s| {
            batch
                .chunks(size)
                .zip(errors.iter_mut())
                .map(|(chunk, error)| s.spawn(|| gradients_batch(chunk, nnue, error)))
                .collect::<Vec<_>>()
                .into_iter()
                .map(|p| p.join().unwrap_or_default())
                .for_each(|part| *grad += *part);
        });
        *error += errors.iter().sum::<f64>();
        grad
    }

    fn update_weights(
        &self,
        nnue: &mut NNUEParams,
        batch: &[Position],
        velocity: &mut NNUEParams,
        momentum: &mut NNUEParams,
        error: &mut f64,
    ) {
        let adj = 2. * K / batch.len() as f64;
        let gradients = self.gradients(nnue, batch, error);

        for (i, param) in nnue.feature_weights.iter_mut().enumerate() {
            let grad = adj * gradients.feature_weights[i];
            adam(
                param,
                &mut momentum.feature_weights[i],
                &mut velocity.feature_weights[i],
                grad,
                self.rate,
            );
        }

        for (i, param) in nnue.output_weights.iter_mut().enumerate() {
            let grad = adj * gradients.output_weights[i];
            adam(
                param,
                &mut momentum.output_weights[i],
                &mut velocity.output_weights[i],
                grad,
                self.rate,
            );
        }

        for (i, param) in nnue.feature_bias.iter_mut().enumerate() {
            let grad = adj * gradients.feature_bias[i];
            adam(
                param,
                &mut momentum.feature_bias[i],
                &mut velocity.feature_bias[i],
                grad,
                self.rate,
            );
        }

        let grad = adj * gradients.output_bias;
        adam(
            &mut nnue.output_bias,
            &mut momentum.output_bias,
            &mut velocity.output_bias,
            grad,
            self.rate,
        );
    }
}

fn gradients_batch(positions: &[Position], nnue: &NNUEParams, error: &mut f64) -> Box<NNUEParams> {
    let mut grad = Box::default();
    for pos in positions {
        update_single_grad(pos, nnue, &mut grad, error);
    }
    grad
}

const B1: f64 = 0.9;
const B2: f64 = 0.999;

fn adam(p: &mut f64, m: &mut f64, v: &mut f64, grad: f64, rate: f64) {
    *m = B1 * *m + (1. - B1) * grad;
    *v = B2 * *v + (1. - B2) * grad * grad;
    *p -= rate * *m / (v.sqrt() + 0.000_000_01);
}
