pub mod position;
pub mod arch;
pub mod data;

use data::Data;
use arch::NNUEParams;


use std::time::Instant;

const B1: f64 = 0.9;
const B2: f64 = 0.999;

fn adam(p: &mut f64, m: &mut f64, v: &mut f64, grad: f64, rate: f64) {
    *m = B1 * *m + (1. - B1) * grad;
    *v = B2 * *v + (1. - B2) * grad * grad;
    *p -= rate * *m / (v.sqrt() + 0.00000001);
}

pub fn gd_tune(data: &Data, nnue: &mut NNUEParams, max_epochs: usize, rate: f64, net_name: &str) {
    let mut velocity = NNUEParams::default();
    let mut momentum = NNUEParams::default();

    let timer = Instant::now();

    let adj = 2. / data.num();
    let mut error = 0.0;

    for epoch in 1..=max_epochs {
        let gradients = data.gradients(nnue, &mut error);
        println!("error {error:.6}");

        for (i, param) in nnue.feature_weights.iter_mut().enumerate() {
            let grad = adj * gradients.feature_weights[i];
            adam(param, &mut momentum.feature_weights[i], &mut velocity.feature_weights[i], grad, rate)
        }

        for (i, param) in nnue.output_weights.iter_mut().enumerate() {
            let grad = adj * gradients.output_weights[i];
            adam(param, &mut momentum.output_weights[i], &mut velocity.output_weights[i], grad, rate)
        }

        for (i, param) in nnue.feature_bias.iter_mut().enumerate() {
            let grad = adj * gradients.feature_bias[i];
            adam(param, &mut momentum.feature_bias[i], &mut velocity.feature_bias[i], grad, rate)
        }

        let grad = adj * gradients.output_bias;
        adam(&mut nnue.output_bias, &mut momentum.output_bias, &mut velocity.output_bias, grad, rate);

        let eps = epoch as f64 / timer.elapsed().as_secs_f64();
        println!("epoch {epoch} rate {rate:.3} eps {eps:.2}/sec");
        if epoch % 10 == 0 {
            nnue.write_to_bin(&format!("{net_name}-{epoch}.bin")).unwrap()
        }
    }
}
