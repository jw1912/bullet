use crate::{Trainer, TrainingSchedule, LocalSettings};

use bullet_core::{inputs::InputType, outputs::OutputBuckets, GpuDataLoader};
use bullet_tensor::{device_name, device_synchronise};
use std::{
    io::{stdout, Write, BufReader, BufRead},
    sync::{atomic::{AtomicBool, Ordering::SeqCst}, mpsc::sync_channel},
    time::Instant, fs::File,
};

#[allow(clippy::too_many_arguments)]
pub fn run<T: InputType, U: OutputBuckets<T::RequiredDataType>>(
    trainer: &mut Trainer<T, U>,
    schedule: &TrainingSchedule,
    settings: &LocalSettings,
) {
    let LocalSettings {
        threads,
        data_file_paths,
        output_directory: out_dir,
    } = settings;

    let threads = *threads;

    std::fs::create_dir(out_dir).unwrap_or(());

    device_synchronise();

    trainer.set_batch_size(schedule.batch_size);

    let esc = esc();
    let rscale = 1.0 / schedule.eval_scale;
    let mut file_size = 0;
    for file in data_file_paths.iter() {
        file_size += std::fs::metadata(file).unwrap_or_else(|_| panic!("Invalid File Metadata: {file}")).len();
    }

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
    println!("Arch           : {}", ansi(format!("{trainer}"), 31));
    schedule.display();
    println!("Device         : {}", ansi(device_name(), 31));
    settings.display();
    println!("Positions      : {}", ansi(num, 31));

    let timer = Instant::now();

    trainer.set_threads(threads);
    device_synchronise();

    let mut prev_lr = schedule.lr(schedule.start_epoch);

    for epoch in schedule.start_epoch..=schedule.end_epoch {
        trainer.set_error_zero();
        let epoch_timer = Instant::now();
        let buffer_size_mb = 256;
        let buffer_size = buffer_size_mb * 1024 * 1024;
        let blend = schedule.wdl(epoch);
        let lrate = schedule.lr(epoch);

        if lrate != prev_lr {
            println!("LR Dropped to {}", ansi(lrate, num_cs()));
        }

        prev_lr = lrate;

        let mut finished = 0;

        let data_size: usize = std::mem::size_of::<T::RequiredDataType>();
        let batches_per_load = buffer_size / data_size / batch_size;
        let cap = data_size * batch_size * batches_per_load;
        let mut loader_files = vec![];
        for file in data_file_paths.iter() {
            loader_files.push(File::open(file).unwrap_or_else(|_| panic!("Invalid File Path: {file}")));
        }

        let (sender, reciever) = sync_channel::<GpuDataLoader<T, U>>(512);
        let x = trainer.input_getter();
        let y = trainer.bucket_getter();

        let dataloader = std::thread::spawn(move || {
            for loader_file in loader_files.iter() {
                let mut file = BufReader::with_capacity(cap, loader_file);
                while let Ok(buf) = file.fill_buf() {
                    if buf.is_empty() {
                        break;
                    }

                    let data: &[T::RequiredDataType] = bullet_core::util::to_slice_with_lifetime(buf);

                    for batch in data.chunks(batch_size) {
                        let mut gpu_loader = GpuDataLoader::<T, U>::new(x, y);
                        gpu_loader.load(batch, threads, blend, rscale);
                        sender.send(gpu_loader).unwrap();
                    }

                    let consumed = buf.len();
                    file.consume(consumed);
                }
            }
        });

        while let Ok(gpu_loader) = reciever.recv() {
            trainer.clear_data();
            device_synchronise();

            trainer.load_data(&gpu_loader);
            device_synchronise();

            trainer.train_on_batch(0.01, lrate);
            device_synchronise();

            if finished % 128 == 0 {
                report_epoch_progress(epoch, batch_size, batches, finished, &epoch_timer);
            }

            finished += 1;
        }

        dataloader.join().unwrap();

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
