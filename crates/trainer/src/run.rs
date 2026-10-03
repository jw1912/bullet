mod dataloader;
pub mod events;
pub mod logger;
mod schedule;

pub use dataloader::{DataLoader, DataLoadingError, HostPool, PreparedBatchHost};
pub use events::TrainingEvent;
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
    InvalidSchedule,
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

    // Batches are never sent to a device, so there is no need for real pinned memory
    let device = Device::<runtime::mock::MockGpu>::new(0).map_err(DataLoadingError::Message)?;
    let pool = HostPool::new(device);

    dataloader.map_batches(&pool, step, steps.batch_size, |_| {
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
    batch_callback: impl FnMut(&mut Optimiser<G, O>, Step, f32),
    superbatch_callback: impl FnMut(&mut Optimiser<G, O>, Step),
) -> Result<(), TrainingError<G>> {
    let mut console = logger::ConsoleObserver::new(schedule.steps, schedule.log_rate);

    train_with_observer(optimiser, schedule, dataloader, batch_callback, superbatch_callback, &mut |event| {
        console.on_event(event);
    })
}

pub fn train_with_observer<G: Gpu, O: OptimiserState<G>>(
    optimiser: &mut Optimiser<G, O>,
    schedule: TrainingSchedule,
    dataloader: impl DataLoader,
    mut batch_callback: impl FnMut(&mut Optimiser<G, O>, Step, f32),
    mut superbatch_callback: impl FnMut(&mut Optimiser<G, O>, Step),
    observer: &mut (impl FnMut(&TrainingEvent) + ?Sized),
) -> Result<(), TrainingError<G>> {
    let timer = Instant::now();
    let device = optimiser.device();
    let props = device.props();

    let steps = schedule.steps;

    if steps.batch_size == 0
        || steps.batches_per_superbatch == 0
        || steps.start_superbatch == 0
        || steps.start_superbatch > steps.end_superbatch
        || steps.end_superbatch == usize::MAX
        || steps
            .end_superbatch
            .checked_mul(steps.batches_per_superbatch)
            .and_then(|batches| batches.checked_mul(steps.batch_size))
            .is_none()
    {
        return Err(TrainingError::InvalidSchedule);
    }

    observer(&TrainingEvent::RunStarted {
        steps,
        device_name: props.name().to_owned(),
        device_arch: props.arch().map(str::to_owned),
    });

    let (sender, receiver) = mpsc::sync_channel::<PreparedBatchHost<G>>(32);

    let pool = HostPool::new(device.clone());
    let dataloader = thread::spawn(move || {
        let mut step = Step::from(steps);

        dataloader.map_batches(&pool, step, steps.batch_size, |batch| {
            if sender.send(batch).is_err() {
                return true;
            }

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

    let first_batch = match receiver.recv() {
        Ok(batch) => batch,
        Err(_) => {
            dataloader.join().map_err(|_| DataLoadingError::Message("Data loader panicked".into()))??;

            return Err(DataLoadingError::NoBatchesReceived.into());
        }
    };

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
    let mut superbatch_timer = Instant::now();
    let mut running_loss = 0.0;
    let mut completed_batches = 0;

    observer(&TrainingEvent::TrainingReady { setup_time: timer.elapsed() });

    while batch_queued {
        if step.finished() {
            return Err(TrainingError::DataLoadingError(DataLoadingError::TooManyBatchesReceived));
        }

        let lrate = lr(step);
        let lrdrop = TValue::F32(vec![lrate]);
        let lrdrop = tlr.copy_from_host_async(&copy_stream, &lrdrop).map_err(TrainingError::Unexpected)?;

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
        completed_batches += 1;

        observer(&TrainingEvent::BatchCompleted {
            step,
            loss: error,
            learning_rate: lrate,
            completed_batches,
            completed_positions: completed_batches * steps.batch_size,
            elapsed: timer.elapsed(),
            superbatch_elapsed: superbatch_timer.elapsed(),
        });

        batch_callback(optimiser, step, error);

        if step.batch() == step.batches_per_superbatch() - 1 {
            let error = running_loss / steps.batches_per_superbatch as f32;
            running_loss = 0.0;

            observer(&TrainingEvent::SuperbatchCompleted {
                step,
                loss: error,
                completed_batches,
                completed_positions: completed_batches * steps.batch_size,
                elapsed: timer.elapsed(),
                superbatch_elapsed: superbatch_timer.elapsed(),
            });

            superbatch_callback(optimiser, step);

            superbatch_timer = Instant::now();
        }

        step.step();
    }

    dataloader.join().map_err(|_| DataLoadingError::Message("Data loader panicked".into()))??;

    if !step.finished() {
        return Err(DataLoadingError::Message("Data loader ended before all scheduled batches completed".into()).into());
    }

    observer(&TrainingEvent::RunCompleted {
        completed_batches,
        completed_positions: completed_batches * steps.batch_size,
        elapsed: timer.elapsed(),
    });

    Ok(())
}
