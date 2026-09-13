use std::{
    sync::mpsc::{Receiver, sync_channel},
    thread::{self, JoinHandle},
    time::Instant,
};

use bullet_trainer::{
    model::{LossEvaluator, ModelInputsMapper},
    optimiser::{Optimiser, OptimiserState},
    reader::DataReader,
    run::{Step, TrainingSteps},
};

use crate::{nn::ExecutionContext, trainer::settings::TestDataset};

pub struct ValidationResult {
    pub loss: f32,
    pub positions: usize,
    pub batches: usize,
    pub seconds: f32,
}

pub struct ValidationRunner<T> {
    receiver: Receiver<Vec<T>>,
    handle: JoinHandle<()>,
    mapper: ModelInputsMapper<T>,
    evaluator: LossEvaluator<ExecutionContext>,
    freq: usize,
    batches: usize,
    batch_size: usize,
    threads: u8,
}

impl<T> ValidationRunner<T>
where
    T: Copy + Send + Sync + 'static,
{
    pub fn new<D, O>(
        reader: &D,
        mapper: ModelInputsMapper<T>,
        optimiser: &Optimiser<ExecutionContext, O>,
        steps: TrainingSteps,
        test: TestDataset<'_>,
        queue_size: usize,
        threads: u8,
    ) -> Self
    where
        D: DataReader<T>,
        O: OptimiserState<ExecutionContext>,
    {
        assert!(test.batches > 0);
        let batch_size = steps.batch_size;

        let validation_events_per_superbatch = 
        if test.freq > 0 { steps.batches_per_superbatch.div_ceil(test.freq) } else { 1 };

        let validation_batches_per_superbatch = validation_events_per_superbatch * test.batches;
        let skip_count = (steps.start_superbatch - 1) * validation_batches_per_superbatch * batch_size;

        let reader = reader.clone();
        let (sender, receiver) = sync_channel(queue_size);

        let handle = thread::spawn(move || {
            let mut incomplete = Vec::with_capacity(batch_size);

            reader.read_chunks(skip_count, |chunk| {
                let mut remaining = chunk;

                if !incomplete.is_empty() {
                    let needed = batch_size - incomplete.len();
                    let take = needed.min(remaining.len());

                    incomplete.extend_from_slice(&remaining[..take]);
                    remaining = &remaining[take..];

                    if incomplete.len() == batch_size {
                        let batch = std::mem::replace(&mut incomplete, Vec::with_capacity(batch_size));

                        if sender.send(batch).is_err() {
                            return true;
                        }
                    }
                }

                let mut chunks = remaining.chunks_exact(batch_size);
                for batch in &mut chunks {
                    if sender.send(batch.to_vec()).is_err() {
                        return true;
                    }
                }

                incomplete.extend_from_slice(chunks.remainder());
                false
            });
        });

        let evaluator = LossEvaluator::new(optimiser.definition(), optimiser.device(), batch_size).unwrap();

        Self { receiver, handle, mapper, evaluator, freq: test.freq, batches: test.batches, batch_size, threads }
    }

    pub fn should_run(&self, step: Step) -> bool {
        step.batch() == step.batches_per_superbatch() - 1
        || (step.batch() > 0 && step.batch().is_multiple_of(self.freq))
    }

    pub fn evaluate<O>(&mut self, optimiser: &Optimiser<ExecutionContext, O>, step: Step) -> ValidationResult
    where
        O: OptimiserState<ExecutionContext>,
    {
        let timer = Instant::now();
        self.evaluator.load_device_weights(optimiser.weights()).unwrap();

        let mut loss_sum = 0.0;
        for _ in 0..self.batches {
            let raw_batch = self.receiver.recv().expect("Validation data loader ended early");

            // validation positions are mapped using the CURRENT TRAINING
            // STEP, so WDL scheduling is identical to the training mapper.
            let host_batch = self.mapper.map(&raw_batch, step, self.threads);
            let device_batch = host_batch.to_device(&optimiser.device()).unwrap();

            let loss = self.evaluator.evaluate(&device_batch).unwrap();
            loss_sum += loss / self.batch_size as f32;
        }

        ValidationResult {
            loss: loss_sum / self.batches as f32,
            positions: self.batch_size * self.batches,
            batches: self.batches,
            seconds: timer.elapsed().as_secs_f32(),
        }
    }

    pub fn finish(self) {
        let Self { receiver, handle, .. } = self;
        drop(receiver);
        handle.join().unwrap();
    }
}
