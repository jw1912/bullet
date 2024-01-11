use crate::{Trainer, TrainingSchedule, LocalSettings};

use bullet_core::{inputs::InputType, GpuDataLoader};
use bullet_tensor::{device_name, device_synchronise};
use bulletformat::DataLoader;
use std::{
    io::{stdout, Write},
    sync::atomic::{AtomicBool, Ordering::SeqCst},
    time::Instant,
};

#[allow(clippy::too_many_arguments)]
pub fn run<T: InputType>(
    trainer: &mut Trainer<T>,
    schedule: &TrainingSchedule,
    settings: &LocalSettings,
) {
    let LocalSettings {
        threads,
        data_file_path: file,
        output_directory: out_dir,
    } = *settings;

    std::fs::create_dir(out_dir).unwrap_or(());

    device_synchronise();

    let esc = esc();
    let rscale = 1.0 / trainer.eval_scale();
    let file_size = std::fs::metadata(file).unwrap().len();
    let num = (file_size / 32) as usize;
    let batch_size = trainer.batch_size();
    let batches = (num + batch_size - 1) / batch_size;

    if device_name() == "CPU" {
        println!("{}", ansi("========== WARNING ==========", 31));
        println!("This backend is not currently");
        println!("   intended to be used for   ");
        println!("  serious training, you may  ");
        println!("  have meant to enable the   ");
        println!("      `cuda` feature.        ");
        println!("{}", ansi("=============================", 31));
    }

    print!("{esc}");
    println!("{}", ansi("Beginning Training", "34;1"));
    println!("Net Name       : {}", ansi(schedule.net_id.clone(), "32;1"));
    trainer.display();
    schedule.display();
    println!("Device         : {}", ansi(device_name(), 31));
    settings.display();
    println!("Positions      : {}", ansi(num, 31));

    let timer = Instant::now();

    let mut gpu_loader = GpuDataLoader::<T>::new(trainer.input_getter());
    trainer.set_threads(threads);

    device_synchronise();

    let mut prev_lr = schedule.lr(schedule.start_epoch);

    for epoch in schedule.start_epoch..=schedule.end_epoch {
        trainer.set_error_zero();
        let epoch_timer = Instant::now();
        let loader = DataLoader::new(file, 1_024).unwrap();
        let blend = schedule.wdl(epoch);
        let lrate = schedule.lr(epoch);

        if lrate != prev_lr {
            println!("LR Dropped to {}", ansi(lrate, num_cs()));
        }

        prev_lr = lrate;

        let mut finished = 0;

        loader.map_batches_threaded_loading(batch_size, |batch| {
            let batch_size = batch.len();

            trainer.clear_data();
            device_synchronise();

            gpu_loader.load(batch, threads, blend, rscale);
            trainer.load_data(&gpu_loader);
            device_synchronise();

            trainer.train_on_batch(0.01, lrate);

            device_synchronise();

            if finished % 128 == 0 {
                report_epoch_progress(epoch, batch_size, batches, finished, &epoch_timer);
            }

            finished += 1;
        });

        let error = trainer.error() / batches as f32;

        report_epoch_finished(schedule, epoch, error, &epoch_timer, &timer, num);

        if schedule.should_save(epoch) {
            trainer.save(out_dir, schedule.net_id(), epoch);

            let name = ansi(format!("{}-epoch{epoch}", schedule.net_id()), "32;1");
            println!("Saved [{name}]");
        }
    }
}

static CBCS: AtomicBool = AtomicBool::new(false);

pub fn ansi<T, U>(x: T, y: U) -> String
where T: std::fmt::Display, U: std::fmt::Display
{
    format!("\x1b[{}m{}\x1b[0m{}", y, x, esc())
}

pub fn set_cbcs(val: bool) {
    CBCS.store(val, SeqCst)
}

fn num_cs() -> i32 {
    if CBCS.load(SeqCst) { 35 } else { 36 }
}

fn esc() -> &'static str {
    if CBCS.load(SeqCst) { "\x1b[38;5;225m" } else { "" }
}

fn report_epoch_progress(
    epoch: usize,
    batch_size: usize,
    batches: usize,
    finished_batches: usize,
    epoch_timer: &Instant,
) {
    let num_cs = num_cs();
    let epoch_time = epoch_timer.elapsed().as_secs_f32();
    let pct = finished_batches as f32 / batches as f32;
    let positions = finished_batches * batch_size;
    let pos_per_sec = positions as f32 / epoch_time;

    let seconds = epoch_time / pct - epoch_time;

    print!(
        "epoch {} [{}% ({}/{} batches, {} pos/sec)]\n\
        Estimated time to end of epoch: {}s     \x1b[F",
        ansi(epoch, num_cs),
        ansi(format!("{:.1}", pct * 100.0), 35),
        ansi(finished_batches, num_cs),
        ansi(batches, num_cs),
        ansi(format!("{pos_per_sec:.0}"), num_cs),
        ansi(format!("{seconds:.1}"), num_cs),
    );
    let _ = stdout().flush();
}

fn report_epoch_finished(
    schedule: &TrainingSchedule,
    epoch: usize,
    error: f32,
    epoch_timer: &Instant,
    timer: &Instant,
    positions: usize,
) {
    let num_cs = num_cs();
    let epoch_time = epoch_timer.elapsed().as_secs_f32();
    let total_time = timer.elapsed().as_secs_f32();
    let pos_per_sec = positions as f32 / epoch_time;

    println!(
        "epoch {} | time {}s | running loss {} | {} pos/sec | total time {}s",
        ansi(epoch, num_cs),
        ansi(format!("{epoch_time:.1}"), num_cs),
        ansi(format!("{error:.6}"), num_cs),
        ansi(format!("{:.0}", pos_per_sec), num_cs),
        ansi(format!("{total_time:.1}"), num_cs),
    );

    let finished_epochs = epoch - schedule.start_epoch + 1;
    let total_epochs = schedule.end_epoch - schedule.start_epoch + 1;
    let pct = finished_epochs as f32 / total_epochs as f32;
    let mut seconds = (total_time / pct - total_time) as u32;
    let mut minutes = seconds / 60;
    let hours = minutes / 60;
    seconds -= minutes * 60;
    minutes -= hours * 60;

    println!(
        "Estimated time remaining in training: {}h {}m {}s",
        ansi(hours, num_cs),
        ansi(minutes, num_cs),
        ansi(seconds, num_cs),
    );
}
