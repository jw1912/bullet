use data::Position;

use crate::{
    arch::{update_single_grad, NNUEParams, QuantisedNNUE, K, test_eval},
    activation::Activation, optimiser::Optimiser,
};

use std::{
    fs::File,
    io::{BufRead, BufReader},
    thread,
    time::Instant,
};

pub struct Trainer<Opt: Optimiser> {
    file: String,
    threads: usize,
    rate: f64,
    blend: f64,
    optimiser: Opt,
}

impl<Opt: Optimiser> Trainer<Opt> {
    #[must_use]
    pub fn new(file: String, threads: usize, rate: f64, blend: f64, optimiser: Opt) -> Self {
        Self {
            file,
            threads,
            rate,
            blend,
            optimiser,
        }
    }

    pub fn run<Act: Activation>(
        &mut self,
        nnue: &mut NNUEParams,
        max_epochs: usize,
        net_name: &str,
        report_rate: usize,
        save_rate: usize,
        batch_size: usize,
    ) {
        let timer = Instant::now();

        let mut error;

        for epoch in 1..=max_epochs {
            error = 0.0;
            let mut num = 0;

            let cap = 1024 * batch_size * std::mem::size_of::<Position>();
            let mut file = BufReader::with_capacity(cap, File::open(&self.file).unwrap());

            while let Ok(buf) = file.fill_buf() {
                // finished reading file
                if buf.is_empty() {
                    break;
                }

                let buf_ref: &[Position] = unsafe { data::util::to_slice_with_lifetime(buf) };

                for batch in buf_ref.chunks(batch_size) {
                    let adj = 2. * K / batch.len() as f64;
                    let gradients = self.gradients::<Act>(nnue, batch, &mut error);

                    self.optimiser.update_weights(nnue, &gradients, adj, self.rate);
                }

                num += buf_ref.len();
                let consumed = buf.len();
                file.consume(consumed);
            }

            error /= num as f64;

            if epoch == 1 {
                println!("Positions: {num}");
            }

            if epoch % report_rate == 0 {
                let elapsed = timer.elapsed().as_secs_f64();
                let eps = epoch as f64 / elapsed;
                println!("epoch {epoch} error {error:.6} time {elapsed:.6} eps {eps:.2}/sec");
            }

            if epoch % save_rate == 0 {
                let qnnue = QuantisedNNUE::from_unquantised(nnue);
                qnnue
                    .write_to_bin(&format!("{net_name}-{epoch}.bin"))
                    .unwrap();
            }
        }

        test_eval::<Act>(nnue);
    }

    fn gradients<Act: Activation>(
        &self,
        nnue: &NNUEParams,
        batch: &[Position],
        error: &mut f64,
    ) -> Box<NNUEParams> {
        let size = batch.len() / self.threads;
        let mut errors = vec![0.0; self.threads];
        let mut grad = NNUEParams::new();
        let blend = self.blend;
        thread::scope(|s| {
            batch
                .chunks(size)
                .zip(errors.iter_mut())
                .map(|(chunk, error)| s.spawn(|| gradients_batch::<Act>(chunk, nnue, error, blend)))
                .collect::<Vec<_>>()
                .into_iter()
                .map(|p| p.join().unwrap())
                .for_each(|part| *grad += &part);
        });
        *error += errors.iter().sum::<f64>();
        grad
    }
}

fn gradients_batch<Act: Activation>(
    positions: &[Position],
    nnue: &NNUEParams,
    error: &mut f64,
    blend: f64,
) -> Box<NNUEParams> {
    let mut grad = NNUEParams::new();
    for pos in positions {
        update_single_grad::<Act>(pos, nnue, &mut grad, error, blend);
    }
    grad
}
