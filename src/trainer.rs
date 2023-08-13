use crate::{
    arch::{update_single_grad, NNUEParams, QuantisedNNUE, K},
    position::Position,
};
use std::{
    fs::File,
    io::{BufRead, BufReader},
    thread,
    time::Instant,
};

#[derive(Default)]
pub struct Trainer {
    data: Vec<Position>,
    threads: usize,
}

impl Trainer {
    #[must_use]
    pub fn new(threads: usize) -> Self {
        Self {
            data: Vec::new(),
            threads,
        }
    }

    #[must_use]
    fn num(&self) -> f64 {
        self.data.len() as f64
    }

    pub fn add_data(&mut self, file_name: &str) {
        let timer = Instant::now();
        let (mut wins, mut losses, mut draws) = (0, 0, 0);
        let file = File::open(file_name).unwrap();
        for line in BufReader::new(file).lines().map(Result::unwrap) {
            let res: Position = line.parse().unwrap();
            match (2. * res.result) as i64 {
                2 => wins += 1,
                0 => losses += 1,
                1 => draws += 1,
                _ => unreachable!(),
            }
            self.data.push(res);
        }
        let elapsed = timer.elapsed().as_secs_f64();
        let pps = self.num() / elapsed;
        println!(
            "{} positions in {elapsed:.2} seconds, {pps:.2} pos/sec",
            self.num()
        );
        println!("wins {wins} losses {losses} draws {draws}");
    }

    fn gradients(&self, nnue: &NNUEParams, error: &mut f64) -> Box<NNUEParams> {
        let size = self.data.len() / self.threads;
        let mut errors = vec![0.0; self.threads];
        let mut grad = Box::default();
        thread::scope(|s| {
            self.data
                .chunks(size)
                .zip(errors.iter_mut())
                .map(|(chunk, error)| s.spawn(|| gradients_batch(chunk, nnue, error)))
                .collect::<Vec<_>>()
                .into_iter()
                .map(|p| p.join().unwrap_or_default())
                .for_each(|part| *grad += *part);
        });
        *error = errors.iter().sum::<f64>() / self.num();
        grad
    }

    pub fn run(
        &self,
        nnue: &mut NNUEParams,
        max_epochs: usize,
        rate: f64,
        net_name: &str,
        report_rate: usize,
        save_rate: usize,
    ) {
        let mut velocity = Box::<NNUEParams>::default();
        let mut momentum = Box::<NNUEParams>::default();

        let timer = Instant::now();

        let adj = 2. * K / self.num();
        let mut error = 0.0;

        for epoch in 1..=max_epochs {
            let gradients = self.gradients(nnue, &mut error);

            for (i, param) in nnue.feature_weights.iter_mut().enumerate() {
                let grad = adj * gradients.feature_weights[i];
                adam(
                    param,
                    &mut momentum.feature_weights[i],
                    &mut velocity.feature_weights[i],
                    grad,
                    rate,
                );
            }

            for (i, param) in nnue.output_weights.iter_mut().enumerate() {
                let grad = adj * gradients.output_weights[i];
                adam(
                    param,
                    &mut momentum.output_weights[i],
                    &mut velocity.output_weights[i],
                    grad,
                    rate,
                );
            }

            for (i, param) in nnue.feature_bias.iter_mut().enumerate() {
                let grad = adj * gradients.feature_bias[i];
                adam(
                    param,
                    &mut momentum.feature_bias[i],
                    &mut velocity.feature_bias[i],
                    grad,
                    rate,
                );
            }

            let grad = adj * gradients.output_bias;
            adam(
                &mut nnue.output_bias,
                &mut momentum.output_bias,
                &mut velocity.output_bias,
                grad,
                rate,
            );

            if epoch % report_rate == 0 {
                let eps = epoch as f64 / timer.elapsed().as_secs_f64();
                println!("epoch {epoch} error {error:.6} rate {rate:.3} eps {eps:.2}/sec");
            }

            if epoch % save_rate == 0 {
                let qnnue = QuantisedNNUE::from_unquantised(nnue);
                qnnue
                    .write_to_bin(&format!("{net_name}-{epoch}.bin"))
                    .unwrap();
            }
        }
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
