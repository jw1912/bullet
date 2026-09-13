use std::time::Instant;
use crate::value::loader::rng::seeded_rng;

use bullet_trainer::{
    model::{LossEvaluator, ModelInputsMapper},
    optimiser::{Optimiser, OptimiserState},
    reader::{DataReader, DataReaderOnce},
    run::Step,
};

use crate::{nn::ExecutionContext, trainer::settings::TestDataset};

pub struct ValidationResult {
    pub loss: f32,
    pub positions: usize,
    pub batches: usize,
    pub seconds: f32,
}

pub struct ValidationRunner<T> {
    data: Vec<T>,
    mapper: ModelInputsMapper<T>,
    // full sized batches
    evaluator: LossEvaluator<ExecutionContext>,
    // cache for if dataset doesnt divide cleanly by batch_size
    remainder_evaluator: Option<LossEvaluator<ExecutionContext>>,
    freq: usize,
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
        batch_size: usize,
        test: TestDataset<'_>,
        threads: u8,
    ) -> Self
    where
        D: DataReader<T> + DataReaderOnce<T>,
        O: OptimiserState<ExecutionContext>,
    {
        // cache fixed set once
        let data = Self::load_validation_data(reader, test.positions);

        assert!(!data.is_empty(), "Validation dataset contains no positions!");

        let evaluator = LossEvaluator::new(optimiser.definition(), optimiser.device(), batch_size).unwrap();
        
        let remainder_size = data.len() % batch_size;
        let remainder_evaluator = if remainder_size > 0 {
            Some(LossEvaluator::new(optimiser.definition(), optimiser.device(), remainder_size).unwrap())
        } else {
            None
        };
        
        Self { data, mapper, evaluator, remainder_evaluator, freq: test.freq, batch_size, threads }
    }

    pub fn positions(&self) -> usize {
        self.data.len()
    }

    pub fn batches(&self) -> usize {
        self.data.len().div_ceil(self.batch_size)
    }

    pub fn should_run(&self, step: Step) -> bool {
        step.batch() == step.batches_per_superbatch() - 1
            || (self.freq > 0 && step.batch() > 0 && step.batch().is_multiple_of(self.freq))
    }

    pub fn evaluate<O>(&mut self, optimiser: &Optimiser<ExecutionContext, O>, step: Step) -> ValidationResult
    where
        O: OptimiserState<ExecutionContext>,
    {
        let timer = Instant::now();
        self.evaluator.load_device_weights(optimiser.weights()).unwrap();

        if let Some(evaluator) = self.remainder_evaluator.as_mut() {
            evaluator.load_device_weights(optimiser.weights()).unwrap();
        }

        let mut total_loss = 0.0;
        let mut positions = 0usize;
        let mut batches = 0usize;

        for raw_batch in self.data.chunks(self.batch_size) {
            // targets are remapped at the current training step
            // so WDL scheduling remaings synced with training
            let host_batch = self.mapper.map(raw_batch, step, self.threads);
            let device_batch = host_batch.to_device(&optimiser.device()).unwrap();

            let loss = if raw_batch.len() == self.batch_size {
                self.evaluator.evaluate(&device_batch).unwrap()
            } else {
                self.remainder_evaluator.as_mut().expect("Missing remainder evaluator").evaluate(&device_batch).unwrap()
            };

            total_loss += loss;
            positions += raw_batch.len();
            batches += 1;
        }

        ValidationResult {
            loss: total_loss / positions as f32,
            positions,
            batches,
            seconds: timer.elapsed().as_secs_f32(),
        }
    }

    fn load_validation_data<D>(
        loader: &D,
        position_count: Option<usize>,
    ) -> Vec<T>
    where
        D: DataReaderOnce<T>,
    {
        match position_count {
            Some(count) => Self::sample_validation_data(loader, count),

            None => {
                let mut data = Vec::new();

                loader.read_once(|chunk| {
                    data.extend_from_slice(chunk);
                    false
                });

                data
            }
        }
    }

    // reservoir sampling (sampled once at init)
    fn sample_validation_data<D>(
        loader: &D,
        target: usize,
    ) -> Vec<T>
    where
        D: DataReaderOnce<T>,
    {
        assert!(target > 0);

        let mut sample = Vec::with_capacity(target);
        let mut seen = 0usize;
        let mut rng = seeded_rng();

        loader.read_once(|chunk| {
            for &entry in chunk {
                if seen < target {
                    sample.push(entry);
                } else {
                    let idx = rng.rand_range(0..seen as u64 + 1) as usize;

                    if idx < target {
                        sample[idx] = entry;
                    }
                }

                seen += 1;
            }

            false
        });

        sample
    }
}
