mod loader;
mod logger;
pub mod schedule;
pub mod settings;

use crate::{backend::util, optimiser::Optimiser};

pub trait NetworkTrainer {
    type PreparedData;
    type Optimiser: Optimiser;

    /// Load prepared data onto the GPU.
    fn load_batch(&mut self, loader: &Self::PreparedData);

    /// Trains for a single step on a batch that has been previously
    /// loaded using `load_batch`.
    fn train_on_batch(&mut self, gf: f32, lr: f32) -> f32 {
        self.optimiser_mut().graph_mut().zero_grads();
        util::device_synchronise();

        let error = self.optimiser_mut().graph_mut().forward();

        self.optimiser_mut().graph_mut().backward();

        self.optimiser_mut().update(gf, lr);

        util::panic_if_device_error("Something went wrong!");

        error
    }

    fn optimiser(&self) -> &Self::Optimiser;

    fn optimiser_mut(&mut self) -> &mut Self::Optimiser;
}
