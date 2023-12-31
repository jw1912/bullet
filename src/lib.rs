mod schedule;
mod trainer;

pub use bullet_core::inputs;
pub use bullet_tensor::Activation;
pub use schedule::{TrainingSchedule, WdlScheduler, LrScheduler, LrSchedulerType};
pub use trainer::{Trainer, TrainerBuilder};

use std::{io::{Write, stdout}, time::Instant, sync::atomic::{AtomicBool, Ordering::SeqCst}};
use bulletformat::DataLoader;
use bullet_core::{inputs::InputType, GpuDataLoader};
use bullet_tensor::device_synchronise;

static CBCS: AtomicBool = AtomicBool::new(false);

pub fn set_cbcs(val: bool) {
    CBCS.store(val, SeqCst)
}

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
pub fn run_training<T: InputType>(
    trainer: &mut Trainer<T>,
    schedule: &mut TrainingSchedule,
    threads: usize,
    file: &str,
) {
    device_synchronise();

    let cbcs = CBCS.load(SeqCst);
    let esc = if cbcs { "\x1b[38;5;225m" } else { "" };
    let num_cs = if cbcs { 35 } else { 36 };
    print!("{esc}");

    println!("{}", ansi!("Beginning Training", "34;1", esc));

    let rscale = 1.0 / trainer.eval_scale();
    let file_size = std::fs::metadata(file).unwrap().len();
    let num = (file_size / 32) as usize;
    let batch_size = trainer.batch_size();
    let batches = (num + batch_size - 1) / batch_size;

    println!("Arch           : {trainer}");
    println!("Data File Path : {}", ansi!(file, "32;1", esc));
    println!("Threads        : {}", ansi!(threads, 31, esc));
    println!("WDL Proportion : start {} end {}",
        ansi!(schedule.wdl_scheduler.start(), 31, esc),
        ansi!(schedule.wdl_scheduler.end(), 31, esc),
    );
    println!("Max Epochs     : {}", ansi!(schedule.num_epochs, 31, esc));
    println!("Save Rate      : {}", ansi!(schedule.save_rate, 31, esc));
    println!("Batch Size     : {}", ansi!(trainer.batch_size(), 31, esc));
    println!("Net Name       : {}", ansi!(schedule.net_id, "32;1", esc));
    println!("LR Scheduler   : {}", schedule.lr_scheduler.colourful(esc));
    println!("Scale          : {}", ansi!(format!("{:.0}", trainer.eval_scale()), 31, esc));
    println!("Positions      : {}", ansi!(num, 31, esc));

    for i in 1..schedule.start_epoch {
        schedule.update(i, num_cs, esc)
    }

    let timer = Instant::now();

    let mut gpu_loader = GpuDataLoader::<T>::default();

    device_synchronise();

    for epoch in schedule.start_epoch..=schedule.num_epochs() {
        trainer.prep_for_epoch();
        let epoch_timer = Instant::now();
        let mut finished_batches = 0;
        let loader = DataLoader::new(file, 1_024).unwrap();
        let blend = schedule.wdl(epoch);

        loader.map_batches_threaded_loading(batch_size, |batch| {
            let batch_size = batch.len();

            trainer.clear_data();
            device_synchronise();

            gpu_loader.load(batch, threads, blend, rscale);
            trainer.load_data(&gpu_loader);
            device_synchronise();


            trainer.train_on_batch(0.01, 0.001);

            device_synchronise();

            if finished_batches % 128 == 0 {
                let pct = finished_batches as f32 / batches as f32 * 100.0;
                let positions = finished_batches * batch_size;
                let pos_per_sec = positions as f32 / epoch_timer.elapsed().as_secs_f32();
                print!(
                    "epoch {} [{}% ({}/{} batches, {} pos/sec)]\r",
                    ansi!(epoch, num_cs, esc),
                    ansi!(format!("{pct:.1}"), 35, esc),
                    ansi!(finished_batches, num_cs, esc),
                    ansi!(batches, num_cs, esc),
                    ansi!(format!("{pos_per_sec:.0}"), num_cs, esc),
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

            trainer.save(schedule.net_id(), epoch);

            println!("Saved [{net_path}]");
        }
    }
}
