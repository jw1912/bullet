use crate::{
    inputs::InputType,
    loader::GpuDataLoader,
    outputs::OutputBuckets,
    tensor::{device_name, device_synchronise},
    util, LocalSettings, Trainer, TrainingSchedule,
};
use std::{
    fs::File,
    io::{stdout, BufRead, BufReader, Write},
    sync::{
        atomic::{AtomicBool, Ordering::SeqCst},
        mpsc::sync_channel,
    },
    time::Instant,
};

#[allow(clippy::too_many_arguments)]
pub fn run<T: InputType, U: OutputBuckets<T::RequiredDataType>>(
    trainer: &mut Trainer<T, U>,
    schedule: &TrainingSchedule,
    settings: &LocalSettings,
) {
    let threads = settings.threads;
    let data_file_paths: Vec<_> = settings.data_file_paths.iter().map(|s| s.to_string()).collect();
    let out_dir = settings.output_directory.to_string();
    let out_dir = out_dir.as_str();

    std::fs::create_dir(out_dir).unwrap_or(());

    device_synchronise();

    trainer.set_batch_size(schedule.batch_size);

    let esc = esc();
    let rscale = 1.0 / schedule.eval_scale;
    let mut file_size = 0;
    for file in data_file_paths.iter() {
        file_size += std::fs::metadata(file)
            .unwrap_or_else(|_| panic!("Invalid File Metadata: {file}"))
            .len();
    }

    let num = (file_size / 32) as usize;
    let batch_size = trainer.batch_size();

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
    println!("Net Name               : {}", ansi(schedule.net_id.clone(), "32;1"));
    println!("Arch                   : {}", ansi(format!("{trainer}"), 31));
    schedule.display();
    println!("Device                 : {}", ansi(device_name(), 31));
    settings.display();
    println!("Positions              : {}", ansi(num, 31));

    let timer = Instant::now();

    trainer.set_threads(threads);
    device_synchronise();

    let x = trainer.input_getter();
    let y = trainer.bucket_getter();
    let sch = schedule.clone();
    let (sender, reciever) = sync_channel::<GpuDataLoader<T, U>>(512);

    let buffer_size_mb = 256;
    let buffer_size = buffer_size_mb * 1024 * 1024;
    let data_size: usize = std::mem::size_of::<T::RequiredDataType>();
    let batches_per_load = buffer_size / data_size / batch_size;
    let cap = data_size * batch_size * batches_per_load;

    let dataloader = std::thread::spawn(move || {
        let mut sb = sch.start_superbatch;
        let mut cb = 0;
        let mut blend = sch.wdl_scheduler.blend(sb, sch.end_superbatch);

        'dataloading: loop {
            let mut loader_files = vec![];
            for file in data_file_paths.iter() {
                loader_files
                    .push(File::open(file).unwrap_or_else(|_| panic!("Invalid File Path: {file}")));
            }

            for loader_file in loader_files.iter() {
                let mut file = BufReader::with_capacity(cap, loader_file);
                while let Ok(buf) = file.fill_buf() {
                    if buf.is_empty() {
                        break;
                    }

                    let data: &[T::RequiredDataType] = util::to_slice_with_lifetime(buf);

                    for batch in data.chunks(batch_size) {
                        let mut gpu_loader = GpuDataLoader::<T, U>::new(x, y);
                        gpu_loader.load(batch, threads, blend, rscale);
                        sender.send(gpu_loader).unwrap();
                        cb += 1;
                        if cb % sch.batches_per_superbatch == 0 {
                            if sb == sch.end_superbatch {
                                break 'dataloading;
                            }

                            cb = 0;
                            sb += 1;
                            blend = sch.wdl_scheduler.blend(sb, sch.end_superbatch);
                        }
                    }

                    let consumed = buf.len();
                    file.consume(consumed);
                }
            }
        }
    });

    let mut prev_lr = schedule.lr(1);
    let mut superbatch = schedule.start_superbatch;
    let mut curr_batch = 0;
    let mut superbatch_timer = Instant::now();
    trainer.set_error_zero();

    while let Ok(gpu_loader) = reciever.recv() {
        let lrate = schedule.lr(superbatch);
        if lrate != prev_lr {
            println!("LR Dropped to {}", ansi(lrate, num_cs()));
        }
        prev_lr = lrate;


        trainer.clear_data();
        device_synchronise();

        trainer.load_data(&gpu_loader);
        device_synchronise();

        let valid = trainer.train_on_batch(0.01, lrate);
        device_synchronise();

        if !valid {
            trainer.save(out_dir, format!("error-nan-batch-{curr_batch}"));
            panic!("Batch {curr_batch} NaN!");
        }

        if curr_batch % 128 == 0 {
            report_superbatch_progress(
                superbatch,
                batch_size,
                schedule.batches_per_superbatch,
                curr_batch,
                &superbatch_timer,
            );
        }

        curr_batch += 1;

        if curr_batch % schedule.batches_per_superbatch == 0 {
            let error = trainer.error() / schedule.batches_per_superbatch as f32;

            report_superbatch_finished(schedule, superbatch, error, &superbatch_timer, &timer, num);

            if schedule.should_save(superbatch) {
                let name = format!("{}-{superbatch}", schedule.net_id());
                trainer.save(out_dir, name.clone());
                println!("Saved [{}]", ansi(name, 31));
            }

            superbatch += 1;
            curr_batch = 0;
            superbatch_timer = Instant::now();
            trainer.set_error_zero();
        }
    }

    dataloader.join().unwrap();
}

static CBCS: AtomicBool = AtomicBool::new(false);

pub fn ansi<T, U>(x: T, y: U) -> String
where
    T: std::fmt::Display,
    U: std::fmt::Display,
{
    format!("\x1b[{}m{}\x1b[0m{}", y, x, esc())
}

pub fn set_cbcs(val: bool) {
    CBCS.store(val, SeqCst)
}

fn num_cs() -> i32 {
    if CBCS.load(SeqCst) {
        35
    } else {
        36
    }
}

fn esc() -> &'static str {
    if CBCS.load(SeqCst) {
        "\x1b[38;5;225m"
    } else {
        ""
    }
}

fn report_superbatch_progress(
    superbatch: usize,
    batch_size: usize,
    batches: usize,
    finished_batches: usize,
    superbatch_timer: &Instant,
) {
    let num_cs = num_cs();
    let superbatch_time = superbatch_timer.elapsed().as_secs_f32();
    let pct = finished_batches as f32 / batches as f32;
    let positions = finished_batches * batch_size;
    let pos_per_sec = positions as f32 / superbatch_time;

    let seconds = superbatch_time / pct - superbatch_time;

    print!(
        "superbatch {} [{}% ({}/{} batches, {} pos/sec)]\n\
        Estimated time to end of superbatch: {}s     \x1b[F",
        ansi(superbatch, num_cs),
        ansi(format!("{:.1}", pct * 100.0), 35),
        ansi(finished_batches, num_cs),
        ansi(batches, num_cs),
        ansi(format!("{pos_per_sec:.0}"), num_cs),
        ansi(format!("{seconds:.1}"), num_cs),
    );
    let _ = stdout().flush();
}

fn report_superbatch_finished(
    schedule: &TrainingSchedule,
    superbatch: usize,
    error: f32,
    superbatch_timer: &Instant,
    timer: &Instant,
    positions: usize,
) {
    let num_cs = num_cs();
    let superbatch_time = superbatch_timer.elapsed().as_secs_f32();
    let total_time = timer.elapsed().as_secs_f32();
    let pos_per_sec = positions as f32 / superbatch_time;

    println!(
        "superbatch {} | time {}s | running loss {} | {} pos/sec | total time {}s",
        ansi(superbatch, num_cs),
        ansi(format!("{superbatch_time:.1}"), num_cs),
        ansi(format!("{error:.6}"), num_cs),
        ansi(format!("{:.0}", pos_per_sec), num_cs),
        ansi(format!("{total_time:.1}"), num_cs),
    );

    let finished_superbatches = superbatch - schedule.start_superbatch + 1;
    let total_superbatches = schedule.end_superbatch - schedule.start_superbatch + 1;
    let pct = finished_superbatches as f32 / total_superbatches as f32;
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
