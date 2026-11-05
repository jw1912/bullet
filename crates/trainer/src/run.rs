mod dataloader;
pub mod events;
pub mod logger;
mod schedule;

pub use dataloader::{DataLoader, DataLoadingError, HostPool, PreparedBatchHost};
pub use events::TrainingEvent;
pub use schedule::{Step, TrainingSchedule, TrainingSteps};

use std::{sync::mpsc, thread, time::Instant};

use bullet_compiler::{model::Layout, tensor::IRTrace};
use bullet_gpu::runtime::{self, Device, Gpu};

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
    MoreDevicesThanBatchSize,
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

    if steps.batch_size < optimiser.ranks() {
        return Err(TrainingError::MoreDevicesThanBatchSize);
    }

    observer(&TrainingEvent::RunStarted {
        steps,
        device_name: props.name().to_owned(),
        device_arch: props.arch().map(str::to_owned),
        device_count: optimiser.ranks(),
    });

    let (sender, receiver) = mpsc::sync_channel::<PreparedBatchHost<G>>(32);

    let ir = optimiser.definition().ir();
    let sparse = ir.inputs().iter().filter_map(|(id, name)| match ir.node(*id).ty().layout() {
        Layout::Sparse(nnz) => Some((name.clone(), nnz)),
        Layout::Dense(_) => None,
    });
    let sparse = sparse.collect::<Vec<_>>();
    let threads = thread::available_parallelism().map_or(1, usize::from).min(16);

    let pool = HostPool::new(device.clone());
    let dataloader = thread::spawn(move || {
        let mut step = Step::from(steps);
        let mut error = None;

        let result = dataloader.map_batches(&pool, step, steps.batch_size, |mut batch| {
            if let Err(e) = batch.compact(&sparse, threads) {
                error = Some(DataLoadingError::Message(format!("{e:?}")));
                return true;
            }

            if sender.send(batch).is_err() {
                return true;
            }

            step.step();
            step.finished()
        });

        error.map_or(result, Err)
    });

    optimiser.prepare(steps.batch_size, schedule.track_grad_norms)?;

    let first_batch = match receiver.recv() {
        Ok(batch) => batch,
        Err(_) => {
            dataloader.join().map_err(|_| DataLoadingError::Message("Data loader panicked".into()))??;

            return Err(DataLoadingError::NoBatchesReceived.into());
        }
    };

    optimiser.load_batch(&first_batch).map_err(TrainingError::Unexpected)?;
    drop(first_batch);

    let lr = schedule.lr_schedule;
    let mut batch_queued = true;
    let mut step = Step::from(steps);
    let mut superbatch_timer = Instant::now();
    let mut running_loss = 0.0;

    observer(&TrainingEvent::TrainingReady { setup_time: timer.elapsed() });

    while batch_queued {
        if step.finished() {
            return Err(TrainingError::DataLoadingError(DataLoadingError::TooManyBatchesReceived));
        }

        let lrate = lr(step);
        let output = optimiser.step(lrate, &mut || receiver.recv().ok())?;
        batch_queued = output.next_queued;

        let error = output.loss / steps.batch_size as f32;

        running_loss += error;

        let grad_norm = output.grad_sqnorms.map(|sqnorms| {
            let norms = sqnorms.into_iter().map(|(name, sq)| (name, sq.sqrt() / steps.batch_size as f32));
            norms.collect()
        });

        observer(&TrainingEvent::BatchCompleted {
            step,
            loss: error,
            learning_rate: lrate,
            elapsed: timer.elapsed(),
            superbatch_elapsed: superbatch_timer.elapsed(),
            grad_norm,
        });

        batch_callback(optimiser, step, error);

        if step.batch() == step.batches_per_superbatch() - 1 {
            let error = running_loss / steps.batches_per_superbatch as f32;
            running_loss = 0.0;

            observer(&TrainingEvent::SuperbatchCompleted {
                step,
                loss: error,
                elapsed: timer.elapsed(),
                superbatch_elapsed: superbatch_timer.elapsed(),
            });

            superbatch_callback(optimiser, step);

            superbatch_timer = Instant::now();
        }

        step.step();
    }

    optimiser.finish();

    dataloader.join().map_err(|_| DataLoadingError::Message("Data loader panicked".into()))??;

    if !step.finished() {
        return Err(DataLoadingError::Message("Data loader ended before all scheduled batches completed".into()).into());
    }

    observer(&TrainingEvent::RunCompleted { elapsed: timer.elapsed() });

    Ok(())
}
