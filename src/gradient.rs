use std::thread;

pub fn gradients_batch_cpu(
    batch: &[Data],
    nnue: &Network,
    error: &mut f32,
    scale: f32,
    blend: f32,
    threads: usize,
) -> Box<Network> {
    let size = batch.len() / threads;
    let mut errors = vec![0.0; threads];
    let mut grad = Network::new();

    thread::scope(|s| {
        batch
            .chunks(size)
            .zip(errors.iter_mut())
            .map(|(chunk, error)| {
                s.spawn(move || {
                    let mut grad = Network::new();
                    for pos in chunk {
                        update_single_grad_cpu(pos, nnue, &mut grad, error, blend, scale);
                    }
                    grad
                })
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(|p| p.join().unwrap())
            .for_each(|part| *grad += &part);
    });
    let batch_error = errors.iter().sum::<f32>();
    *error += batch_error;
    grad
}
