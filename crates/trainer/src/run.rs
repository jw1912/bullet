pub mod dataloader;
pub mod logger;
pub mod schedule;

use std::{sync::mpsc, thread, time::Instant};

use bullet_compiler::tensor::TValue;
use bullet_gpu::{
    buffer::{Buffer, SyncOnValue},
    runtime::Gpu,
};

use crate::{
    DataLoadingError, Trainer, TrainerError,
    optimiser::OptimiserState,
    run::{
        dataloader::{DataLoader, PreparedBatchHost},
        schedule::TrainingSchedule,
    },
};

pub fn train_custom<G: Gpu, O: OptimiserState<G>, S>(
    trainer: &mut Trainer<G, O, S>,
    schedule: TrainingSchedule,
    dataloader: impl DataLoader,
    mut batch_callback: impl FnMut(&mut Trainer<G, O, S>, usize, usize, f32),
    mut superbatch_callback: impl FnMut(&mut Trainer<G, O, S>, usize),
) -> Result<(), TrainerError<G>> {
    trainer.optimiser.model.set_bwd_batch_size(schedule.steps.batch_size).map_err(TrainerError::Unexpected)?;

    let model = &trainer.optimiser.model;
    let device = model.device();
    let props = device.props();

    logger::clear_colours();
    println!(
        "{}",
        logger::ansi(format!("Training on {} ({})", props.name(), props.arch().unwrap_or("unknown")), "34;1")
    );

    let timer = Instant::now();
    let lr = schedule.lr_schedule;
    let steps = schedule.steps;

    let (sender, receiver) = mpsc::sync_channel::<PreparedBatchHost>(32);

    let dataloader = thread::spawn(move || {
        let mut batch_no = 0;
        let mut superbatch = steps.start_superbatch;

        dataloader.map_batches(steps.batch_size, |batch| {
            if batch.batch_size != steps.batch_size {
                panic!("Dataloader returned a batch with incorrect batch size!");
            }

            sender.send(batch).unwrap();

            batch_no += 1;

            if batch_no % steps.batches_per_superbatch == 0 {
                batch_no = 0;
                superbatch += 1;

                if superbatch > steps.end_superbatch {
                    return true;
                }
            }

            false
        })
    });

    let mut prev_lr = lr(0, 1);
    let mut superbatch = steps.start_superbatch;
    let mut curr_batch = 0;
    let mut superbatch_timer = Instant::now();
    let mut running_loss = 0.0;
    let mut superbatch_positions = 0;

    let first_batch =
        receiver.recv().map_err(|_| TrainerError::DataLoadingError(DataLoadingError::NoBatchesReceived))?;

    let copy_stream = device.new_stream().map_err(TrainerError::Unexpected)?;
    let compute_stream = device.new_stream().map_err(TrainerError::Unexpected)?;

    let outputs = model.make_backward_output_tensors().map_err(TrainerError::Unexpected)?;
    let gradients = model.make_gradient_tensors().map_err(TrainerError::Unexpected)?;
    let tlr = Buffer::from_host(&device, &TValue::F32(vec![0.0])).map_err(TrainerError::Unexpected)?;
    let tgf = Buffer::from_host(&device, &TValue::F32(vec![0.0])).map_err(TrainerError::Unexpected)?;

    let mut next_batch_size = first_batch.batch_size;
    let mut batch_on_device = first_batch.to_device(&device).map_err(TrainerError::Unexpected)?;

    let mut next_on_device = batch_on_device
        .iter()
        .map(|(id, tensor)| {
            let buf = Buffer::zeroed(&device, tensor.dtype(), tensor.size());
            (id.clone(), buf.unwrap())
        })
        .collect();

    let mut batch_queued = true;

    while batch_queued {
        if superbatch > steps.end_superbatch {
            return Err(TrainerError::DataLoadingError(DataLoadingError::TooManyBatchesReceived));
        }

        // ignore startup time from loading the first batch of data
        // because it just poisons the reported pos/sec when reading
        // from binpacked data
        if superbatch == steps.start_superbatch && curr_batch == 0 {
            superbatch_timer = Instant::now();
        }

        let lrate = lr(curr_batch, superbatch);
        let this_batch_size = next_batch_size;

        let lrdrop = TValue::F32(vec![lrate]);
        let lrdrop = tlr.copy_from_host_async(&copy_stream, &lrdrop).map_err(TrainerError::Unexpected)?;
        let gfdrop = TValue::F32(vec![1.0 / this_batch_size as f32]);
        let gfdrop = tgf.copy_from_host_async(&copy_stream, &gfdrop).map_err(TrainerError::Unexpected)?;

        if curr_batch == 0 {
            if lrate < prev_lr {
                println!("LR dropped to {}", logger::ansi(lrate, logger::num_cs()));
            } else if lrate > prev_lr {
                println!("LR increased to {}", logger::ansi(lrate, logger::num_cs()));
            }
        }

        prev_lr = lrate;

        let compute_block1 = trainer
            .optimiser
            .model
            .backward(&compute_stream, &batch_on_device, &outputs, &gradients)
            .map_err(TrainerError::GradientCalculationError)?;

        let compute_block1 = unsafe { compute_block1.detach_value() };

        lrdrop.value().map_err(TrainerError::Unexpected)?;
        gfdrop.value().map_err(TrainerError::Unexpected)?;

        let compute_block2 = trainer
            .optimiser
            .update(&compute_stream, tgf.clone(), tlr.clone(), &gradients)
            .map_err(TrainerError::OptimiserUpdateError)?;

        if let Ok(next_batch_host) = receiver.recv() {
            next_batch_size = next_batch_host.batch_size;
            drop(
                next_batch_host
                    .copy_to_device_async(&copy_stream, &next_on_device)
                    .map_err(TrainerError::Unexpected)?,
            );
            std::mem::swap(&mut batch_on_device, &mut next_on_device);
        } else {
            batch_queued = false;
        }

        compute_block1.sync().map_err(TrainerError::Unexpected)?;
        compute_block2.sync().map_err(TrainerError::Unexpected)?;

        let loss = outputs.get("outputs/loss").expect("`Trainer` must have a \"loss\" output!");
        let TValue::F32(loss) = loss
            .clone()
            .to_host_async(&copy_stream)
            .map(SyncOnValue::value)
            .map_err(TrainerError::Unexpected)?
            .map_err(TrainerError::Unexpected)?
        else {
            panic!()
        };
        let [loss] = loss[..] else { panic!() };
        let error = loss / this_batch_size as f32;

        running_loss += error;
        superbatch_positions += this_batch_size;

        if curr_batch % schedule.log_rate == 0 {
            logger::report_superbatch_progress(
                superbatch,
                steps.batches_per_superbatch,
                curr_batch,
                &superbatch_timer,
                superbatch_positions,
            );
        }

        curr_batch += 1;

        batch_callback(trainer, superbatch, curr_batch, error);

        if curr_batch % steps.batches_per_superbatch == 0 {
            let error = running_loss / steps.batches_per_superbatch as f32;
            running_loss = 0.0;

            let total_time = timer.elapsed().as_secs_f32();
            let sb_time = superbatch_timer.elapsed().as_secs_f32();

            logger::report_superbatch_finished(superbatch, error, sb_time, total_time, superbatch_positions);
            logger::report_time_left(steps, superbatch, total_time);

            superbatch_callback(trainer, superbatch);

            superbatch += 1;
            curr_batch = 0;
            superbatch_positions = 0;
            superbatch_timer = Instant::now();
        }
    }

    let total_time = timer.elapsed().as_secs();
    let (hours, minutes, seconds) = logger::seconds_to_hms(total_time as u32);

    println!(
        "Total Training Time: {}h {}m {}s",
        logger::ansi(hours, logger::num_cs()),
        logger::ansi(minutes, logger::num_cs()),
        logger::ansi(seconds, logger::num_cs()),
    );

    dataloader.join().unwrap()?;

    Ok(())
}
