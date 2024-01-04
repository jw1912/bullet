use crate::{Trainer, TrainingSchedule};

use bullet_core::{inputs::InputType, GpuDataLoader};
use bullet_tensor::{device_name, device_synchronise};
use bulletformat::DataLoader;
use std::{
    io::{stdout, Write},
    sync::atomic::{AtomicBool, Ordering::SeqCst},
    time::Instant,
};

#[macro_export]
macro_rules! ansi {
    ($x:expr, $y:expr) => {
        format!("\x1b[{}m{}\x1b[0m", $y, $x)
    };
    ($x:expr, $y:expr, $esc:expr) => {
        format!("\x1b[{}m{}\x1b[0m{}", $y, $x, $esc)
    };
}

static CBCS: AtomicBool = AtomicBool::new(false);

pub fn set_cbcs(val: bool) {
    CBCS.store(val, SeqCst)
}

#[allow(clippy::too_many_arguments)]
pub fn run<T: InputType>(
    trainer: &mut Trainer<T>,
    schedule: &TrainingSchedule,
    threads: usize,
    file: &str,
    out_dir: &str,
) {
    std::fs::create_dir(out_dir).unwrap_or(());

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

    println!("Net Name       : {}", ansi!(schedule.net_id, "32;1", esc));
    println!("Arch           : {}", ansi!(format!("{trainer}"), 31, esc));
    println!("Batch Size     : {}", ansi!(trainer.batch_size(), 31, esc));
    println!(
        "Scale          : {}",
        ansi!(format!("{:.0}", trainer.eval_scale()), 31, esc)
    );

    println!("Start Epoch    : {}", ansi!(schedule.start_epoch, 31, esc));
    println!("End Epoch      : {}", ansi!(schedule.end_epoch, 31, esc));
    println!("Save Rate      : {}", ansi!(schedule.save_rate, 31, esc));
    println!("WDL Scheduler  : {}", schedule.wdl_scheduler.colourful(esc));
    println!("LR Scheduler   : {}", schedule.lr_scheduler.colourful(esc));

    println!("Device         : {}", ansi!(device_name(), 31, esc));
    println!("Threads        : {}", ansi!(threads, 31, esc));
    println!("Data File Path : {}", ansi!(file, "32;1", esc));
    println!("Positions      : {}", ansi!(num, 31, esc));

    let timer = Instant::now();

    let mut gpu_loader = GpuDataLoader::<T>::new(trainer.input_getter());

    device_synchronise();

    let mut prev_lr = schedule.lr(schedule.start_epoch);

    for epoch in schedule.start_epoch..=schedule.end_epoch {
        trainer.prep_for_epoch();
        let epoch_timer = Instant::now();
        let mut finished_batches = 0;
        let loader = DataLoader::new(file, 1_024).unwrap();
        let blend = schedule.wdl(epoch);
        let lrate = schedule.lr(epoch);

        if lrate != prev_lr {
            println!("LR Dropped to {}", ansi!(lrate, num_cs, esc));
        }

        prev_lr = lrate;

        loader.map_batches_threaded_loading(batch_size, |batch| {
            let batch_size = batch.len();

            trainer.clear_data();
            device_synchronise();

            gpu_loader.load(batch, threads, blend, rscale);
            trainer.load_data(&gpu_loader);
            device_synchronise();

            trainer.train_on_batch(0.01, lrate);

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

        if schedule.should_save(epoch) {
            trainer.save(out_dir, schedule.net_id(), epoch);

            let name = ansi!(format!("{}-epoch{epoch}", schedule.net_id()), "32;1", esc);
            println!("Saved [{name}]");
        }
    }
}
