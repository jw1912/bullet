mod schedule;
mod trainer;

pub use bullet_core::inputs;
pub use bullet_tensor::Activation;
pub use schedule::{TrainingSchedule, WdlScheduler, LrScheduler, LrSchedulerType};
pub use trainer::{Trainer, TrainerBuilder};

use std::{io::{Write, stdout}, time::Instant};
use bulletformat::DataLoader;
use bullet_core::data::BoardCUDA;
use bullet_tensor::device_synchronise;

#[macro_export]
macro_rules! ansi {
    ($x:expr, $y:expr) => {
        format!("\x1b[{}m{}\x1b[0m", $y, $x)
    };
    ($x:expr, $y:expr, $esc:expr) => {
        format!("\x1b[{}m{}\x1b[0m{}", $y, $x, $esc)
    };
}

#[allow(clippy::too_many_arguments)]
pub fn run_training<T>(
    trainer: &mut Trainer<T>,
    schedule: &mut TrainingSchedule,
    threads: usize,
    file: &str,
    cbcs: bool,
    start_epoch: usize,
) {
    device_synchronise();

    let esc = if cbcs { "\x1b[38;5;225m" } else { "" };
    let num_cs = if cbcs { 35 } else { 36 };
    print!("{esc}");

    println!("{}", ansi!("Beginning Training", "34;1", esc));

    let rscale = 1.0 / trainer.eval_scale();
    let file_size = std::fs::metadata(file).unwrap().len();
    let num = (file_size / 32) as usize;
    let batch_size = trainer.batch_size();
    let batches = (num + batch_size - 1) / batch_size;

    println!("Positions: {num}");

    for i in 1..start_epoch {
        schedule.update(i, num_cs, esc)
    }

    let timer = Instant::now();

    device_synchronise();

    for epoch in start_epoch..=schedule.num_epochs() {
        trainer.prep_for_epoch();
        let epoch_timer = Instant::now();
        let mut finished_batches = 0;
        let loader = DataLoader::new(file, 1_024).unwrap();
        let blend = schedule.wdl(epoch);

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

        println!(
            "epoch {} | time {} | running loss {} | {} pos/sec | total time {}",
            ansi!(epoch, num_cs, esc),
            ansi!(format!("{epoch_time:.2}"), num_cs, esc),
            ansi!(format!("{error:.6}"), num_cs, esc),
            ansi!(
                format!("{:.0}", num.max(1) as f32 / epoch_time),
                num_cs,
                esc
            ),
            ansi!(format!("{:.2}", timer.elapsed().as_secs_f32()), num_cs, esc),
        );

        schedule.update(epoch, num_cs, esc);

        if schedule.should_save(epoch) {
            let net_path = format!("net_test-epoch{epoch}");

            trainer.save(schedule.net_id(), epoch).unwrap();

            println!("Saved [{net_path}]");
        }
    }
}
