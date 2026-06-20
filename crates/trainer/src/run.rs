mod dataloader;
pub mod logger;
mod schedule;

pub use dataloader::{DataLoader, DataLoadingError, HostPool, PreparedBatchHost};
pub use schedule::{Step, TrainingSchedule, TrainingSteps};

use std::{collections::BTreeMap, sync::mpsc, thread, time::Instant};

use bullet_compiler::{
    model::Layout,
    tensor::{IRTrace, TValue},
};
use bullet_gpu::{
    buffer::{Buffer, SyncOnValue},
    function::Function,
    runtime::{self, Device, Gpu},
};

use crate::optimiser::{Optimiser, OptimiserState};

#[cfg(not(any(feature = "cuda", feature = "rocm")))]
pub type DefaultDevice = Device<runtime::mock::MockGpu>;

#[cfg(feature = "cuda")]
pub type DefaultDevice = Device<runtime::cuda::Cuda>;

#[cfg(all(feature = "rocm", not(feature = "cuda")))]
pub type DefaultDevice = Device<runtime::rocm::ROCm>;

#[derive(Debug)]
pub enum TrainingError<G: Gpu> {
    DataLoadingError(DataLoadingError),
    GradientCalculationError(G::Error),
    OptimiserUpdateError(G::Error),
    Unexpected(G::Error),
    CompilingBackwards(IRTrace),
    IoError,
}

impl<G: Gpu> From<DataLoadingError> for TrainingError<G> {
    fn from(value: DataLoadingError) -> Self {
        Self::DataLoadingError(value)
    }
}

pub fn measure_max_cpu_throughput(dataloader: impl DataLoader, steps: TrainingSteps) -> Result<(), DataLoadingError> {
    let timer = Instant::now();
    logger::clear_colours();
    println!("{}", logger::ansi("Measuring CPU Throughput", "34;1"));

    let sb_cnt = steps.batches_per_superbatch * steps.batch_size;

    let mut sb_timer = Instant::now();
    let mut step = Step::from(steps);

    dataloader.map_batches(step, steps.batch_size, |_| {
        if step.batch() == steps.batches_per_superbatch - 1 {
            let sb = step.superbatch();
            let total_time = timer.elapsed().as_secs_f32();
            let sb_time = sb_timer.elapsed().as_secs_f32();
            logger::report_superbatch_throughput(sb, sb_time, total_time, sb_cnt);
            logger::report_time_left(steps, sb, total_time);

            sb_timer = Instant::now();
        }

        step.step();
        step.finished()
    })?;

    Ok(())
}

pub fn train<G: Gpu, O: OptimiserState<G>>(
    optimiser: &mut Optimiser<G, O>,
    schedule: TrainingSchedule,
    dataloader: impl DataLoader,
    mut batch_callback: impl FnMut(&mut Optimiser<G, O>, Step, f32),
    mut superbatch_callback: impl FnMut(&mut Optimiser<G, O>, Step),
) -> Result<(), TrainingError<G>> {
    let timer = Instant::now();

    let device = optimiser.device();
    let props = device.props();

    logger::clear_colours();
    println!(
        "{}",
        logger::ansi(format!("Training on {} ({})", props.name(), props.arch().unwrap_or("unknown")), "34;1")
    );

    let steps = schedule.steps;
    let (sender, receiver) = mpsc::sync_channel::<PreparedBatchHost>(32);
    let dataloader = thread::spawn(move || {
        let mut step = Step::from(steps);

        dataloader.map_batches(step, steps.batch_size, |batch| {
            sender.send(batch).unwrap();
            step.step();
            step.finished()
        })
    });

    let defn = optimiser.definition();
    let (func, gmap) =
        defn.lower_backward(&Default::default(), steps.batch_size).map_err(TrainingError::CompilingBackwards)?;
    let map = func.map();
    let mut backwards = Function::new(device.clone(), func.ir().clone()).map_err(TrainingError::CompilingBackwards)?;
    backwards.prealloc().map_err(TrainingError::Unexpected)?;

    let mut tensor_map = BTreeMap::new();

    let mut gradients = BTreeMap::new();
    for (mid, (name, _)) in defn.ir().weights() {
        let tid = *map.get(mid).unwrap();
        let gid = *gmap.get(mid).unwrap();

        let ty = defn.ir().node(*mid).ty();
        let Layout::Dense(dtype) = ty.layout() else { unreachable!() };
        let size = ty.shape().size();
        let grad = Buffer::zeroed(&device, dtype, size).map_err(TrainingError::Unexpected)?;

        tensor_map.insert(tid, optimiser.weights().get(name).unwrap().clone());
        tensor_map.insert(gid, grad.clone());

        gradients.insert(name.clone(), grad);
    }

    let tgf = TValue::F32(vec![1.0 / steps.batch_size as f32]);
    let tgf = Buffer::from_host(&device, &tgf).map_err(TrainingError::Unexpected)?;
    let tlr = Buffer::from_host(&device, &TValue::F32(vec![0.0])).map_err(TrainingError::Unexpected)?;
    let loss = Buffer::from_host(&device, &TValue::F32(vec![0.0])).map_err(TrainingError::Unexpected)?;

    tensor_map.insert(*map.get(&defn.loss().unwrap()).unwrap(), loss.clone());

    let first_batch =
        receiver.recv().map_err(|_| TrainingError::DataLoadingError(DataLoadingError::NoBatchesReceived))?;
    let mut batch_on_device = first_batch.to_device(&device).map_err(TrainingError::Unexpected)?;
    let mut next_on_device = batch_on_device
        .iter()
        .map(|(id, tensor)| {
            let buf = Buffer::zeroed(&device, tensor.dtype(), tensor.size());
            (id.clone(), buf.unwrap())
        })
        .collect();

    let mut input_names = BTreeMap::new();
    for (mid, name) in defn.ir().inputs() {
        let tid = *map.get(mid).unwrap();

        input_names.insert(tid, name.clone());
        tensor_map.insert(tid, batch_on_device.get(name).unwrap().clone());
    }

    let copy_stream = device.new_stream().map_err(TrainingError::Unexpected)?;
    let compute_stream = device.new_stream().map_err(TrainingError::Unexpected)?;
    let lr = schedule.lr_schedule;
    let mut batch_queued = true;
    let mut step = Step::from(steps);
    let mut prev_lr = lr(step);
    let mut superbatch_timer = Instant::now();
    let mut running_loss = 0.0;
    let mut superbatch_positions = 0;

    while batch_queued {
        if step.finished() {
            return Err(TrainingError::DataLoadingError(DataLoadingError::TooManyBatchesReceived));
        }

        let lrate = lr(step);
        let lrdrop = TValue::F32(vec![lrate]);
        let lrdrop = tlr.copy_from_host_async(&copy_stream, &lrdrop).map_err(TrainingError::Unexpected)?;

        if step.batch() == 0 {
            if lrate < prev_lr {
                println!("LR dropped to {}", logger::ansi(lrate, logger::num_cs()));
            } else if lrate > prev_lr {
                println!("LR increased to {}", logger::ansi(lrate, logger::num_cs()));
            }
        }

        prev_lr = lrate;

        let compute_block1 =
            backwards.execute(compute_stream.clone(), &tensor_map).map_err(TrainingError::GradientCalculationError)?;

        lrdrop.value().map_err(TrainingError::Unexpected)?;

        let compute_block2 = optimiser
            .update(&compute_stream, tgf.clone(), tlr.clone(), &gradients)
            .map_err(TrainingError::OptimiserUpdateError)?;

        if let Ok(next_batch_host) = receiver.recv() {
            drop(
                next_batch_host
                    .copy_to_device_async(&copy_stream, &next_on_device)
                    .map_err(TrainingError::Unexpected)?,
            );
            std::mem::swap(&mut batch_on_device, &mut next_on_device);

            for (id, name) in &input_names {
                *tensor_map.get_mut(id).unwrap() = batch_on_device.get(name).unwrap().clone();
            }
        } else {
            batch_queued = false;
        }

        let _ = compute_block1.value().map_err(TrainingError::Unexpected)?;
        compute_block2.sync().map_err(TrainingError::Unexpected)?;

        let TValue::F32(loss) = loss
            .to_host_async(&copy_stream)
            .map(SyncOnValue::value)
            .map_err(TrainingError::Unexpected)?
            .map_err(TrainingError::Unexpected)?
        else {
            panic!()
        };
        let [loss] = loss[..] else { panic!() };
        let error = loss / steps.batch_size as f32;

        running_loss += error;
        superbatch_positions += steps.batch_size;

        if step.batch().is_multiple_of(schedule.log_rate) {
            logger::report_superbatch_progress(step, &superbatch_timer, superbatch_positions);
        }

        batch_callback(optimiser, step, error);

        if step.batch() == step.batches_per_superbatch() - 1 {
            let error = running_loss / steps.batches_per_superbatch as f32;
            running_loss = 0.0;

            let total_time = timer.elapsed().as_secs_f32();
            let sb_time = superbatch_timer.elapsed().as_secs_f32();

            let sb = step.superbatch();
            logger::report_superbatch_finished(sb, error, sb_time, total_time, superbatch_positions);
            logger::report_time_left(steps, sb, total_time);

            superbatch_callback(optimiser, step);

            superbatch_positions = 0;
            superbatch_timer = Instant::now();
        }

        step.step();
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
