use std::{f32::consts::PI, fmt::Debug};

use crate::trainer::logger::ansi;

/// Learning rate scheduling. Types implementing this trait output a learning rate
/// at each point in training, indexed by batch and superbatch.
pub trait LrScheduler: Clone + Debug + Send + Sync {
    /// The learning rate for the current batch and superbatch.
    /// Most schedulers do not depend on the batch index.
    fn lr(&self, batch: usize, superbatch: usize) -> f32;
    /// A colourful display representation of the learning rate scheduler.
    fn colourful(&self) -> String;
}

/// Constant learning rate.
#[derive(Clone, Debug)]
pub struct ConstantLR {
    pub value: f32,
}

impl LrScheduler for ConstantLR {
    fn lr(&self, _batch: usize, _superbatch: usize) -> f32 {
        self.value
    }

    fn colourful(&self) -> String {
        format!("constant {}", ansi(self.value, 31))
    }
}

/// Drop once at superbatch `drop`, by a factor of `gamma`.
#[derive(Clone, Debug)]
pub struct DropLR {
    pub start: f32,
    pub gamma: f32,
    pub drop: usize,
}

impl LrScheduler for DropLR {
    fn lr(&self, _batch: usize, superbatch: usize) -> f32 {
        if superbatch > self.drop {
            self.start * self.gamma
        } else {
            self.start
        }
    }

    fn colourful(&self) -> String {
        format!(
            "start {} gamma {} drop at {} superbatches",
            ansi(self.start, 31),
            ansi(self.gamma, 31),
            ansi(self.drop, 31)
        )
    }
}

/// Drop every `step` superbatches by a factor of `gamma`.
#[derive(Clone, Debug)]
pub struct StepLR {
    pub start: f32,
    pub gamma: f32,
    pub step: usize,
}

impl LrScheduler for StepLR {
    fn lr(&self, _batch: usize, superbatch: usize) -> f32 {
        let steps = superbatch.saturating_sub(1) / self.step;
        self.start * self.gamma.powi(steps as i32)
    }

    fn colourful(&self) -> String {
        format!(
            "start {} gamma {} drop every {} superbatches",
            ansi(self.start, 31),
            ansi(self.gamma, 31),
            ansi(self.step, 31),
        )
    }
}

#[derive(Clone, Debug)]
pub struct LinearDecayLR {
    pub initial_lr: f32,
    pub final_lr: f32,
    pub final_superbatch: usize,
}

impl LrScheduler for LinearDecayLR {
    fn lr(&self, _batch: usize, superbatch: usize) -> f32 {
        if superbatch >= self.final_superbatch {
            return self.final_lr;
        }

        let lambda = superbatch as f32 / self.final_superbatch as f32;
        self.initial_lr + lambda * (self.final_lr - self.initial_lr)
    }

    fn colourful(&self) -> String {
        format!(
            "start at {} and linearly decay to {} at superbatch {}",
            ansi(self.initial_lr, 31),
            ansi(self.final_lr, 31),
            ansi(self.final_superbatch, 31),
        )
    }
}

#[derive(Clone, Debug)]
pub struct CosineDecayLR {
    pub initial_lr: f32,
    pub final_lr: f32,
    pub final_superbatch: usize,
}

impl LrScheduler for CosineDecayLR {
    fn lr(&self, _batch: usize, superbatch: usize) -> f32 {
        if superbatch >= self.final_superbatch {
            return self.final_lr;
        }

        let progress = superbatch as f32 / self.final_superbatch as f32;
        let lambda = 1.0 - 0.5 * (1.0 + (PI * progress).cos());
        self.initial_lr + lambda * (self.final_lr - self.initial_lr)
    }

    fn colourful(&self) -> String {
        format!(
            "start at {} and cosine decay to {} at superbatch {}",
            ansi(self.initial_lr, 31),
            ansi(self.final_lr, 31),
            ansi(self.final_superbatch, 31),
        )
    }
}

#[derive(Clone, Debug)]
pub struct ExponentialDecayLR {
    pub initial_lr: f32,
    pub final_lr: f32,
    pub final_superbatch: usize,
}

impl LrScheduler for ExponentialDecayLR {
    fn lr(&self, _batch: usize, superbatch: usize) -> f32 {
        if superbatch >= self.final_superbatch {
            return self.final_lr;
        }

        let lambda = superbatch as f32 / self.final_superbatch as f32;
        self.initial_lr * (self.final_lr / self.initial_lr).powf(lambda)
    }

    fn colourful(&self) -> String {
        format!(
            "start at {} and exponentially decay to {} at superbatch {}",
            ansi(self.initial_lr, 31),
            ansi(self.final_lr, 31),
            ansi(self.final_superbatch, 31),
        )
    }
}

/// Warm up to a sub-scheduler over `warmup_batches` batches.
#[derive(Clone, Debug)]
pub struct Warmup<LR: LrScheduler> {
    pub inner: LR,
    pub warmup_batches: usize,
}

impl<LR: LrScheduler> LrScheduler for Warmup<LR> {
    fn lr(&self, batch: usize, superbatch: usize) -> f32 {
        let base_lr = self.inner.lr(batch, superbatch);
        // batch loops within superbatches, so we must check we're
        // actually at the start of training to correctly implement
        // warmup.
        if superbatch == 1 && batch < self.warmup_batches {
            // linearly interpolate up from base_lr / warmup_batches
            base_lr / (self.warmup_batches - batch) as f32
        } else {
            base_lr
        }
    }

    fn colourful(&self) -> String {
        // < BASE_SCHEDULER_TEXT >, warmup over {} batches
        format!("{}, warmup over {} batches", self.inner.colourful(), ansi(self.warmup_batches, 31))
    }
}

/// Sequence two sub-schedulers, switching over at `first_scheduler_final_superbatch`
#[derive(Clone, Debug)]
pub struct Sequence<First: LrScheduler, Second: LrScheduler> {
    pub first: First,
    pub second: Second,
    pub first_scheduler_final_superbatch: usize,
}

impl<First: LrScheduler, Second: LrScheduler> LrScheduler for Sequence<First, Second> {
    fn lr(&self, batch: usize, superbatch: usize) -> f32 {
        let midpoint = self.first_scheduler_final_superbatch;

        if superbatch <= midpoint {
            self.first.lr(batch, superbatch)
        } else {
            self.second.lr(batch, superbatch - midpoint)
        }
    }

    fn colourful(&self) -> String {
        // < LEFT_SCHEDULER_TEXT >, then after {} superbatches, < RIGHT_SCHEDULER_TEXT>
        format!(
            "{}, then after {} superbatches, {}",
            self.first.colourful(),
            ansi(self.first_scheduler_final_superbatch, 32),
            self.second.colourful()
        )
    }
}
