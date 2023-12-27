mod trainer;

pub use bullet_core::inputs::Chess768;
pub use bullet_tensor::Activation;
pub use trainer::{Trainer, TrainerBuilder};

use std::{io::{Write, stdout}, time::Instant};
use bulletformat::DataLoader;
use bullet_core::data::BoardCUDA;
use bullet_tensor::device_synchronise;

#[allow(clippy::too_many_arguments)]
pub fn run_training(
    trainer: &mut Trainer<Chess768>,
    threads: usize,
    max_epochs: usize,
    scale: f32,
    file: &str,
) {
    let rscale = 1.0 / scale;
    let file_size = std::fs::metadata(file).unwrap().len();
    let num = (file_size / 32) as usize;
    let batch_size = trainer.batch_size();
    let batches = (num + batch_size - 1) / batch_size;

    println!("Positions: {num}");

    let timer = Instant::now();

    trainer.eval("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 | 0 | 0.5");

    device_synchronise();

    for epoch in 1..=max_epochs {
        trainer.prep_for_epoch();
        let epoch_timer = Instant::now();
        let mut finished_batches = 0;
        let loader = DataLoader::new(file, 1_024).unwrap();
        let blend = 0.5;

        loader.map_batches_threaded_loading(batch_size, |batch| {
            trainer.clear_data();
            let batch_size = batch.len();
            let chunk_size = (batch.len() + threads - 1) / threads;

            device_synchronise();

            std::thread::scope(|s| {
                batch
                    .chunks(chunk_size)
                    .map(|chunk| {
                        s.spawn(move || {
                            let num = chunk.len();
                            let mut our_inputs = Vec::with_capacity(num);
                            let mut opp_inputs = Vec::with_capacity(num);
                            let mut results = Vec::with_capacity(num);

                            for pos in chunk {
                                BoardCUDA::push(
                                    pos,
                                    &mut our_inputs,
                                    &mut opp_inputs,
                                    &mut results,
                                    blend,
                                    rscale
                                );
                            }

                            (our_inputs, opp_inputs, results)
                        })
                    })
                    .collect::<Vec<_>>()
                    .into_iter()
                    .map(|p| p.join().unwrap())
                    .for_each(|(our_inputs, opp_inputs, results)| {
                        trainer.append_data(&our_inputs, &opp_inputs, &results);
                    });
            });

            device_synchronise();

            trainer.train_on_batch(0.01, 0.001);

            device_synchronise();

            if finished_batches % 128 == 0 {
                let pct = finished_batches as f32 / batches as f32 * 100.0;
                let positions = finished_batches * batch_size;
                let pos_per_sec = positions as f32 / epoch_timer.elapsed().as_secs_f32();
                print!(
                    "epoch {epoch} [{pct}% ({finished_batches}/{batches} batches, {pos_per_sec} pos/sec)]\r",
                );
                let _ = stdout().flush();
            }

            finished_batches += 1;
        });

        let error = trainer.error() / num as f32;

        let epoch_time = epoch_timer.elapsed().as_secs_f32();

        trainer.eval("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 | 0 | 0.5");

        println!(
            "epoch {epoch} | time {epoch_time:.2} | running loss {error:.6} | {} pos/sec | total time {}",
            num.max(1) as f32 / epoch_time,
            timer.elapsed().as_secs_f32(),
        );

        let net_path = format!("net_test-epoch{epoch}");

        println!("Saved [{net_path}]");
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_build() {
        let net = TrainerBuilder::<Chess768>::default()
            .set_batch_size(16_384)
            .ft(512)
            .activate(Activation::ReLU)
            .add_layer(1)
            .build();

        println!("Network Architecture: {net}");
    }
}
